import time

import torch
import random
import numpy as np
import numpy.core.defchararray as np_f
import csv

import os

from artist_net import make_actual_nn, sample_len, version_name
from debug import _print

artist_count = 3

samples_per_file = 400
# samples_per_file = 300

# files_per_artist_train = 18
# files_per_artist_validate = 6
files_per_artist_train = 12
files_per_artist_validate = 4
# files_per_artist_train = 5
# files_per_artist_validate = 2

files_per_artist_total = files_per_artist_train + files_per_artist_validate

count_per_artist_train = files_per_artist_train * samples_per_file
count_per_artist_validate = files_per_artist_validate * samples_per_file

count_train = count_per_artist_train * artist_count
count_validate = count_per_artist_validate * artist_count

working_dir = '.'
os.makedirs(working_dir, exist_ok=True)

weights_dir = f'{working_dir}/{version_name}_{artist_count}_{samples_per_file}_{files_per_artist_total}_weights'
os.makedirs(weights_dir, exist_ok=True)

accuracy_log = f'{working_dir}/{version_name}_{artist_count}_{samples_per_file}_{files_per_artist_total}_accuracy.csv'

lr = 1e-3
the_batch_size = 100

last_epoch = -1
end_epoch = 1000

accum_coeff = 0.9

if last_epoch == -1:
    score_accumulator = 1.0 / artist_count
    accuracy_accumulator_validate = 1.0 / artist_count
else:
    score_accumulator = 0.9334167838096619
    accuracy_accumulator_validate = 0.9349722132006357

USE_IPEX = True

if USE_IPEX:
    import intel_extension_for_pytorch as ipex

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
# torch.set_num_threads(1)

score_accumulator = torch.FloatTensor([score_accumulator]).to(device)

print('loading data')

# arr_1d = np.array(range(count_per_type_validate * S))
# print(np.reshape(arr_1d, (count_per_type_validate, S)))

with open(f'{working_dir}/artist_{artist_count}_{samples_per_file}_{files_per_artist_total}_train.raw', "rb") as file:
    raw_artist_data_train = file.read()

_print(len(raw_artist_data_train))
artist_data_train = np.frombuffer(raw_artist_data_train, dtype=np.uint8).astype(float)
_print(artist_data_train)
_print(np.min(artist_data_train), np.max(artist_data_train), np.mean(artist_data_train))
artist_data_train = np.reshape(artist_data_train, (count_train, sample_len))
# print('max delta', np.max(artist_data_train_2 - artist_data_train))
_print(artist_data_train)
_print(artist_data_train.shape)

with open(f'{working_dir}/artist_{artist_count}_{samples_per_file}_{files_per_artist_total}_validate.raw', "rb") as file:
    raw_artist_data_validate = file.read()

_print(len(raw_artist_data_validate))
artist_data_validate = np.frombuffer(raw_artist_data_validate, dtype=np.uint8).astype(float)
_print(artist_data_validate)
_print(np.min(artist_data_validate), np.max(artist_data_validate), np.mean(artist_data_validate))
artist_data_validate = np.reshape(artist_data_validate, (count_validate, sample_len))
_print(artist_data_validate)
_print(artist_data_validate.shape)

artist_labels_train = []
artist_labels_validate = []

for artist in range(artist_count):
    row = []
    for _artist in range(artist_count):
        row.append(1.0 if artist == _artist else 0.0)
    row = np.array(row)
    artist_labels_train += [row] * count_per_artist_train
    artist_labels_validate += [row] * count_per_artist_validate

artist_labels_train = np.array(artist_labels_train)
artist_labels_validate = np.array(artist_labels_validate)

_print(artist_labels_train.shape)

print('loading data done')

print('making train and test')

_print(np.mean(artist_data_train), np.mean(artist_labels_train))

X_train = artist_data_train
y_train = artist_labels_train
X_validate = artist_data_validate
y_validate = artist_labels_validate

print('making train and test done')

(H_train, W) = X_train.shape
(H_validate, W) = X_validate.shape

print(W, H_train, H_validate)

print('making tensors')

X_train = torch.FloatTensor(X_train).to(device)
X_validate = torch.FloatTensor(X_validate).to(device)
y_train = torch.FloatTensor(y_train).to(device)
y_validate = torch.FloatTensor(y_validate).to(device)
_print('ok here')

X_train = X_train.divide(255.0).subtract(0.5)
X_validate = X_validate.divide(255.0).subtract(0.5)
_print('ok here')

print(X_train.cpu().min(), X_train.cpu().max(), X_train.cpu().mean())
print(X_validate.cpu().min(), X_validate.cpu().max(), X_validate.cpu().mean())

print('making tensors done')

print('preparing neural networking')
artist_net = make_actual_nn()

epoch = last_epoch

if epoch >= 0:
    fn_weights = f'{weights_dir}/model_weights_epoch_{epoch}.pth'
    artist_net.load_state_dict(torch.load(fn_weights)) #, map_location=device))
    # artist_net.eval()

artist_net = artist_net.to(device)

optimizer = torch.optim.Adam(artist_net.parameters(), lr=lr)

if USE_IPEX:
    artist_net, optimizer = ipex.optimize(artist_net, optimizer=optimizer, dtype=torch.float32)

print('preparing neural networking done')

# loss = torch.nn.CrossEntropyLoss()

col_summator = torch.ones(artist_count, 1).to(device)
score_pow_scale = np.log10(2) / np.log10(artist_count)


def loss(pred, target, train_mode):
    _print(pred.shape)
    _print(target.shape)
    # print(pred.shape, pred.min(), pred.max())
    # print(target)
    count = float(list(target.shape)[0])
    score = (pred * target) @ col_summator
    # print(score.shape, score.min(), score.max())
    score = score.sum().divide(count)
    # print(score)
    if train_mode:
        global score_accumulator
        # print(score_accumulator.cpu())
        score_accumulator = score_accumulator.multiply(accum_coeff).detach()
        score_accumulator += (1 - accum_coeff) * score
        # return (1.0 - score_accumulator + 1e-8).log10()
        return (0.5 - score_accumulator ** score_pow_scale).multiply(1.999).atanh().multiply(10000)
    else:
        # return (1.0 - score + 1e-8).log10()
        return (0.5 - score ** score_pow_scale).multiply(1.999).atanh().multiply(10000)


epochs = []
accuracies = []

t0 = time.time()

print('starting training')
while epoch < end_epoch:
    epoch += 1
    order = np.random.permutation(len(X_train))

    artist_net.train()

    for start_index in range(0, len(X_train), the_batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index+the_batch_size]

        X_batch = X_train[batch_indexes].to(device)
        y_batch = y_train[batch_indexes].to(device)

        y_pred = artist_net.inference(X_batch)

        loss_value = loss(y_pred, y_batch, True)

        _print(loss_value.cpu())
        loss_value.backward()
        optimizer.step()

    artist_net.eval()
    with torch.no_grad():
        # if epoch % 5 == 0:
        if True:
            order = np.random.permutation(len(X_validate))
            validate_batch_indexes = order[0:400]
            x_validate_batch = X_validate[validate_batch_indexes].to(device)
            y_validate_batch = y_validate[validate_batch_indexes].to(device)
            test_preds = artist_net.inference(x_validate_batch).to(device)
            # loss_value = loss(test_preds, y_validate_batch, False).cpu().item()
            loss_value = loss_value.cpu().item()
            test_preds_numbers = test_preds.argmax(dim=1).cpu()
            y_validate_numbers = y_validate_batch.argmax(dim=1).cpu()
            accuracy = (test_preds_numbers == y_validate_numbers).float().mean().item()
            _print((test_preds_numbers == 0).sum())
            _print((test_preds_numbers == 1).sum())
            _print((test_preds_numbers == 2).sum())
            accuracy_accumulator_validate = accuracy_accumulator_validate * accum_coeff + accuracy * (1 - accum_coeff)
            print(epoch, accuracy_accumulator_validate, score_accumulator.item(), loss_value)
            epochs.append(epoch)
            accuracies.append(accuracy)
            t1 = time.time()
            print('time', t1 - t0)
            with open(accuracy_log, "a") as f:
                f.write(f'{epoch};{accuracy_accumulator_validate};{score_accumulator.item()};{loss_value};{t1 - t0}\n')
            t0 = t1

        if epoch % 10 == 0:
            fn_weights = f'{weights_dir}/model_weights_epoch_{epoch}.pth'
            torch.save(artist_net.state_dict(), fn_weights)
