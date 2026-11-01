import time
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy import signal
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from artist_net import ArtistNetSpectrogramV2, sample_len
from debug import _print

version_name = 'spectrogram_v2'

artist_count = 3
samples_per_file = 1000
files_per_artist_train = 18
files_per_artist_validate = 6
files_per_artist_total = files_per_artist_train + files_per_artist_validate

count_per_artist_train = files_per_artist_train * samples_per_file
count_per_artist_validate = files_per_artist_validate * samples_per_file
count_train = count_per_artist_train * artist_count
count_validate = count_per_artist_validate * artist_count

working_dir = '.'
os.makedirs(working_dir, exist_ok=True)

weights_dir = f'{working_dir}/weights/{version_name}_{artist_count}_{samples_per_file}_{files_per_artist_total}_weights'
os.makedirs(weights_dir, exist_ok=True)

audio_data_raw_dir = f'{working_dir}/audio_data_raw'
accuracy_log = f'{working_dir}/accuracy/{version_name}_{artist_count}_{samples_per_file}_{files_per_artist_total}_accuracy.csv'

lr = 3e-4
lr_gamma = 0.9
the_batch_size = 400
sample_rate = 8000

last_epoch = -1
end_epoch = 50
accum_coeff = 0.9

# True  -> весь датасет лежит на GPU/XPU (быстрее, но занимает видеопамять)
# False -> датасет в оперативной памяти, на устройство уходит только батч (экономит видеопамять)
KEEP_DATA_ON_DEVICE = False
# Сколько сэмплов одновременно переводить в спектрограммы (ограничивает пиковую память)
SPECTROGRAM_CHUNK = 500

if last_epoch == -1:
    score_accumulator = 1.0 / artist_count
    accuracy_accumulator_validate = 1.0 / artist_count
    loss_value_accumulator = 0.0
else:
    score_data = pd.read_csv(accuracy_log, delimiter=';').to_numpy().astype(float)
    epoch_data = score_data[:, 0]
    score_data = score_data[np.where(epoch_data == last_epoch)]
    score_accumulator = score_data[-1, 2]
    accuracy_accumulator_validate = score_data[-1, 1]
    loss_value_accumulator = score_data[-1, 3]

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
torch.set_num_threads(6)

data_device = device if KEEP_DATA_ON_DEVICE else torch.device('cpu')


def load_spectrograms(path, count):
    """Читает сырые uint8-сэмплы через memmap (без копии в памяти) и считает
    спектрограммы чанками сразу в float32. Промежуточные float64-массивы
    живут только на размер одного чанка."""
    raw = np.memmap(path, dtype=np.uint8, mode='r').reshape(count, sample_len)
    out = None
    for s in tqdm(range(0, count, SPECTROGRAM_CHUNK)):
        chunk = np.asarray(raw[s:s + SPECTROGRAM_CHUNK], dtype=np.float64)
        _, _, spec = signal.spectrogram(chunk, sample_rate, axis=-1)
        if out is None:
            out = np.empty((count,) + spec.shape[1:], dtype=np.float32)
        out[s:s + len(chunk)] = spec
    del raw
    return out


print('loading data')
X_train = load_spectrograms(
    f'{audio_data_raw_dir}/artist_{artist_count}_{samples_per_file}_{files_per_artist_total}_train.raw', count_train)
print(X_train.shape)
X_validate = load_spectrograms(
    f'{audio_data_raw_dir}/artist_{artist_count}_{samples_per_file}_{files_per_artist_total}_validate.raw', count_validate)
print(X_validate.shape)

# Метки — индексы классов (int64) вместо one-hot float-матриц.
# CrossEntropyLoss с индексами математически эквивалентен one-hot таргетам.
y_train = torch.from_numpy(np.repeat(np.arange(artist_count), count_per_artist_train)).long()
y_validate = torch.from_numpy(np.repeat(np.arange(artist_count), count_per_artist_validate)).long()
print('loading data done')

(N_train, W, H) = X_train.shape
(N_validate, W, H) = X_validate.shape
print(W, H, N_train, N_validate)

print('making tensors')
# from_numpy не копирует память; при data_device == cpu копии нет вообще
X_train = torch.from_numpy(X_train).to(data_device)
X_validate = torch.from_numpy(X_validate).to(data_device)
y_train = y_train.to(data_device)
y_validate = y_validate.to(data_device)
print(X_train.min().item(), X_train.max().item(), X_train.mean().item())
print(X_validate.min().item(), X_validate.max().item(), X_validate.mean().item())
print('making tensors done')

print('preparing neural networking')
artist_net = ArtistNetSpectrogramV2()

epoch = last_epoch
if epoch >= 0:
    fn_weights = f'{weights_dir}/model_weights_epoch_{epoch}.pth'
    artist_net.load_state_dict(torch.load(fn_weights, map_location='cpu'))

artist_net = artist_net.to(device)

optimizer = torch.optim.Adam(artist_net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_gamma)
for i in range(last_epoch):
    scheduler.step()
print('preparing neural networking done')

loss = torch.nn.CrossEntropyLoss()


@torch.no_grad()
def calc_score(pred, target):
    """Средняя вероятность правильного класса (без построения графа)."""
    global score_accumulator
    score = torch.softmax(pred.detach(), dim=1).gather(1, target.unsqueeze(1)).mean().item()
    score_accumulator = score_accumulator * accum_coeff + (1 - accum_coeff) * score


epochs = []
accuracies = []
t0 = time.time()
max_accuracy = 0.0

print('starting training')
while epoch < end_epoch:
    epoch += 1
    order = torch.from_numpy(np.random.permutation(len(X_train)))

    artist_net.train()
    for start_index in tqdm(range(0, len(X_train), the_batch_size)):
        optimizer.zero_grad(set_to_none=True)

        batch_indexes = order[start_index:start_index + the_batch_size]
        X_batch = X_train[batch_indexes].to(device)
        y_batch = y_train[batch_indexes].to(device)

        y_pred = artist_net.forward(X_batch)
        loss_value = loss(y_pred, y_batch)
        calc_score(y_pred, y_batch)
        loss_value_accumulator = loss_value_accumulator * accum_coeff + loss_value.item() * (1 - accum_coeff)

        loss_value.backward()
        optimizer.step()

    scheduler.step()

    artist_net.eval()
    with torch.no_grad():
        order = torch.from_numpy(np.random.permutation(len(X_validate)))
        all_preds, all_labels = [], []
        for i in tqdm(range(samples_per_file * files_per_artist_validate * artist_count // the_batch_size)):
            idx = order[the_batch_size * i:the_batch_size * (i + 1)]
            x_val = X_validate[idx].to(device)
            all_preds.append(artist_net.forward(x_val).argmax(dim=1).cpu())
            all_labels.append(y_validate[idx].cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        accuracy = f1_score(all_labels, all_preds, average='macro')
        accuracy_accumulator_validate = accuracy_accumulator_validate * accum_coeff + accuracy * (1 - accum_coeff)
        print(epoch, accuracy_accumulator_validate, score_accumulator, loss_value_accumulator, scheduler.get_last_lr())
        epochs.append(epoch)
        accuracies.append(accuracy)
        print(accuracy, max_accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            torch.save(artist_net.state_dict(), f'{weights_dir}/model_weights_epoch_{epoch}.pth')
        t1 = time.time()
        print('time', t1 - t0)
        os.makedirs(os.path.dirname(accuracy_log), exist_ok=True)
        with open(accuracy_log, "a") as f:
            f.write(f'{epoch};{accuracy_accumulator_validate};{score_accumulator};{loss_value_accumulator};{t1 - t0}\n')
        t0 = t1

# Финальный отчёт — без графа, батчами
artist_net.eval()
with torch.no_grad():
    idx = torch.from_numpy(np.random.permutation(len(X_validate))[:1000])
    preds = artist_net.forward(X_validate[idx].to(device)).argmax(dim=1).cpu()
    print(classification_report(y_validate[idx].cpu(), preds))
