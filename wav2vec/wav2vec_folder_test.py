import sys
import os
import time

import glob

import torch
import random
import numpy as np
import numpy.core.defchararray as np_f
import csv
import math
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pydub import AudioSegment

from deepseek_wav2vec_test import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# from artist_net import make_actual_nn, version_name
# from debug import _print

# working_dir = '.'

# weights_dir = f'{working_dir}/{version_name}_3_400_16_weights'
# weights_dir = f'{working_dir}/{version_name}_3_300_24_weights'

FRAME_RATE = 8000
CHUNK_SIZE = 24000

last_epoch = 250

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

S = 24000
N = 3
L = 40
target_sr = 16000


def get_random_batch_from_file(fn_in_mp3):
    print(fn_in_mp3)

    sound = AudioSegment.from_mp3(fn_in_mp3)

    print(sound.frame_rate, sound.sample_width, sound.channels)
    print(len(sound.raw_data), len(sound.raw_data) / (sound.frame_rate * sound.sample_width * sound.channels))
    sound = sound.set_frame_rate(FRAME_RATE)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(1)
    print(len(sound.raw_data), len(sound.raw_data) / FRAME_RATE)

    total_len = len(sound.raw_data)

    all_input_values = []
    i = 0
    while i < L:
        start = 0.9 * rnd.random() + 0.05
        start = round(start * total_len)
        if total_len - start < 3 * CHUNK_SIZE:
            continue
        chunk_raw_data = sound.raw_data[start:start + CHUNK_SIZE]
        chunk_raw_data = np.frombuffer(chunk_raw_data, dtype=np.uint8).astype(float)
        chunk_raw_data = np.reshape(chunk_raw_data, (1, S))
        inputs = processor(
            chunk_raw_data,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding="max_length",
            max_length=24000,
            truncation=True
        )
        input_values = inputs['input_values'].squeeze(0).numpy()
        all_input_values.append(input_values)
        i += 1
    all_input_values = np.array(all_input_values)
    # print(all_input_values.shape)
    all_input_values = torch.tensor(all_input_values)
    return all_input_values


model_path="wav2vec2_finetuned"

try:
    # Загружаем процессор и модель
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    # model = Wav2Vec2ForSequenceClassification.from_pretrained(
    #     "facebook/wav2vec2-base",
    #     num_labels=3,
    #     attention_dropout=0.1,
    #     hidden_dropout=0.1,
    #     classifier_proj_size=256,
    #     ignore_mismatched_sizes=True
    # ).cpu()
    # model.load_state_dict(torch.load("model.pth"))
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path, device_map="cpu")
    model.eval()
    model = model.to(device)
    if USE_IPEX:
        model = ipex.optimize(model)
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    raise


def detect_file_type(fn_in_mp3):
    torch.xpu.empty_cache()

    all_input_values = get_random_batch_from_file(fn_in_mp3).to(device)

    outputs = model(input_values=all_input_values)
    logits = outputs.logits

    # Вероятности
    probabilities = torch.softmax(logits, dim=1)

    pp = []
    for i in range(N):
        p = probabilities.mean(dim=0)[i].item() * 100
        print(f'artist {i + 1}: {p}')
        pp.append(p)
    pp = np.array(pp)

    print(np.argmax(pp), np.max(pp))

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
    plt.savefig("confusion_matrix_folder_test.png", dpi=150)
    plt.close()

