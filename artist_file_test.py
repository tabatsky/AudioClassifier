import sys
import os
import time

import torch
import random
import numpy as np
import numpy.core.defchararray as np_f
import csv
import math

from pydub import AudioSegment

from artist_net import make_actual_nn, version_name
from debug import _print



def _print(*args):
    # if True:
    if False:
        print(args)


working_dir = '.'

weights_dir = f'{working_dir}/{version_name}_3_400_16_weights'

FRAME_RATE = 8000
CHUNK_SIZE = 24000

last_epoch = 3200

S = 24000
N = 3
L = 500

ktan = 0.03
USE_IPEX = True

if USE_IPEX:
    import intel_extension_for_pytorch as ipex

rnd = random.Random()


print('cuda available:', torch.cuda.is_available())
print('xpu available:', torch.xpu.is_available())

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
if torch.xpu.is_available() and USE_IPEX:
    device = torch.device('xpu:0')

print('device:', device)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.xpu.manual_seed(0)
torch.backends.cudnn.deterministic = True

print('preparing neural networking')
artist_net = make_actual_nn()

if last_epoch >= 0:
    fn_weights = f'{weights_dir}/model_weights_epoch_{last_epoch}.pth'
    artist_net.load_state_dict(torch.load(fn_weights, map_location=device))
    artist_net.eval()

artist_net = artist_net.to(device)

if USE_IPEX:
    artist_net = ipex.optimize(artist_net, dtype=torch.float32)

print('preparing neural networking done')


def detect_file_type(fn_in_mp3):
    print(fn_in_mp3)

    sound = AudioSegment.from_mp3(fn_in_mp3)

    print(sound.frame_rate, sound.sample_width, sound.channels)
    print(len(sound.raw_data), len(sound.raw_data) / (sound.frame_rate * sound.sample_width * sound.channels))
    sound = sound.set_frame_rate(FRAME_RATE)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(1)
    print(len(sound.raw_data), len(sound.raw_data) / FRAME_RATE)

    total_len = len(sound.raw_data)

    audio_raw_data = bytes(0)
    i = 0
    while i < L:
        start = 0.9 * rnd.random() + 0.05
        start = round(start * total_len)
        if total_len - start < 3 * CHUNK_SIZE:
            continue
        chunk_raw_data = sound.raw_data[start:start+CHUNK_SIZE]
        audio_raw_data += chunk_raw_data
        i += 1

    audio_data = np.frombuffer(audio_raw_data, dtype=np.uint8).astype(float)
    _print(audio_data)
    _print(np.min(audio_data), np.max(audio_data), np.mean(audio_data))
    audio_data = np.reshape(audio_data, (L, S))
    _print(audio_data)
    _print(audio_data.shape)

    X = torch.FloatTensor(audio_data).to(device)
    X = X.divide(255.0).subtract(0.5)

    pred = artist_net.inference(X)
    pp = []
    for i in range(N):
        p = pred[:, i].mean().item() * 100
        print(f'artist {i+1}: {p}')
        pp.append(p)
    pp = np.array(pp)

    return [np.argmax(pp), np.max(pp)]


if len(sys.argv) > 1:
    the_fn_in_mp3 = sys.argv[1]
    print(the_fn_in_mp3)
    detect_file_type(the_fn_in_mp3)
else:
    result = []
    total_plus = 0
    total_minus = 0

    for artist in range(N):
        filenames = os.listdir(f'artist{artist+1}_train')
        # for i in range(12, len(filenames)):
        for i in range(len(filenames)):
            the_fn_in_mp3 = f'artist{artist+1}_train/{filenames[i]}'
            pred = detect_file_type(the_fn_in_mp3)
            result.append([i, artist, pred, filenames[i]])
            if pred[0] == artist:
                total_plus += 1
            else:
                total_minus += 1

    for item in result:
        print(item)

    print(total_plus, total_minus)

