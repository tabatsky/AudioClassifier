import sys

import glob

import torch
import random
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from pydub import AudioSegment

from scipy import signal

from artist_net import ArtistNetSpectrogramV9
from debug import _print

version_name = 'spectrogram_v9'

working_dir = '.'

# suffix = '3_400_16'
# suffix = '3_300_24'
suffix = '3_400_24'

weights_dir = f'{working_dir}/weights/{version_name}_{suffix}_weights'

FRAME_RATE = 8000
CHUNK_SIZE = 24000

last_epoch = 94

S = 24000
N = 3
L = 500

sample_rate = 8000

ktan = 0.03

rnd = random.Random()

print('cuda available:', torch.cuda.is_available())
print('xpu available:', torch.xpu.is_available())

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
if torch.xpu.is_available():
    device = torch.device('xpu:0')

print('device:', device)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.xpu.manual_seed(0)
torch.backends.cudnn.deterministic = True

print('preparing neural networking')
artist_net = ArtistNetSpectrogramV9()

if last_epoch >= 0:
    fn_weights = f'{weights_dir}/model_weights_epoch_{last_epoch}.pth'
    artist_net.load_state_dict(torch.load(fn_weights))
    artist_net.eval()

artist_net = artist_net.to(device)


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

    audio_data_spectrograms = []
    for i in range(L):
        sample = audio_data[i, :]
        frequencies, times, spectrogram = signal.spectrogram(sample, sample_rate)
        audio_data_spectrograms.append(spectrogram)
    audio_data_spectrograms = np.array(audio_data_spectrograms)
    print(audio_data_spectrograms.shape)

    X = torch.FloatTensor(audio_data_spectrograms).to(device)
    # X = X.divide(255.0).subtract(0.5)

    pred = artist_net.inference(X)
    pp = []
    for i in range(N):
        p = pred[:, i].mean().item() * 100
        print(f'artist {i+1}: {p}')
        pp.append(p)
    pp = np.array(pp)

    return [np.argmax(pp), np.max(pp)]

def do_it(folder, artist):
    result = np.array([0] * 3)
    # total_plus = 0
    # total_minus = 0
    fn_in_list = glob.glob(f'{folder}\\*.mp3') + glob.glob(f'{folder}\\*\\*.mp3')
    for i in range(len(fn_in_list)):
        the_fn_in_mp3 = fn_in_list[i]
        print(i, the_fn_in_mp3)
        pred = detect_file_type(the_fn_in_mp3)
        # result.append([i, artist, pred, the_fn_in_mp3])
        # if pred[0] == artist:
        #     total_plus += 1
        # else:
        #     total_minus += 1
        result[pred[0]] += 1
    # for item in result:
    #     print(item)
    # print(total_plus, total_minus, total_plus + total_minus)
    # return [total_plus, total_minus, total_plus + total_minus]
    return result


if len(sys.argv) > 1:
    folder = sys.argv[1]
    artist = int(sys.argv[2])
else:
    folders = [
        "N:\\Немного Нервно MP3",
        "N:\\music\\MP3\\The Cranberries",
        "N:\\music\\MP3\\Любэ"
    ]
    result = []
    for artist in range(3):
        result.append(do_it(folders[artist], artist))
    result = np.array(result)
    print(result)
    plt.figure(figsize=(10, 8))
    sns.heatmap(result, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1', '2'],
                yticklabels=['0', '1', '2'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_folder_test_{version_name}_{suffix}.png", dpi=150)
    plt.close()

