import time

import torch
import random
import numpy as np
import numpy.core.defchararray as np_f
import csv
import math

import os


def _print(*args):
    # if True:
    if False:
        print(args)


count_per_type_train = 2000
count_per_type_validate = 500

working_dir = '.'
os.makedirs(working_dir, exist_ok=True)

weights_dir = f'{working_dir}/weights_2d_label'
os.makedirs(weights_dir, exist_ok=True)

accuracy_log = f'{working_dir}/accuracy_2d_label.csv'

lr = 3e-5
ktan = 0.03

S = 24000
the_batch_size = 100
N = 2

last_epoch = -1
end_epoch = 100

USE_IPEX = True

if USE_IPEX:
    import intel_extension_for_pytorch as ipex

print('loading data')

# arr_1d = np.array(range(count_per_type_validate * S))
# print(np.reshape(arr_1d, (count_per_type_validate, S)))

with open(f'{working_dir}/music_train.raw', "rb") as file:
    raw_music_data_train = file.read()

_print(len(raw_music_data_train))
music_data_train = np.frombuffer(raw_music_data_train, dtype=np.uint8).astype(float)
_print(music_data_train)
_print(np.min(music_data_train), np.max(music_data_train), np.mean(music_data_train))
music_data_train = np.reshape(music_data_train, (count_per_type_train, S))
# print('max delta', np.max(music_data_train_2 - music_data_train))
_print(music_data_train)
_print(music_data_train.shape)
music_labels_train = np.array([np.array([1.0, 0.0])] * count_per_type_train)

with open(f'{working_dir}/music_validate.raw', "rb") as file:
    raw_music_data_validate = file.read()

_print(len(raw_music_data_validate))
music_data_validate = np.frombuffer(raw_music_data_validate, dtype=np.uint8).astype(float)
_print(music_data_validate)
_print(np.min(music_data_validate), np.max(music_data_validate), np.mean(music_data_validate))
music_data_validate = np.reshape(music_data_validate, (count_per_type_validate, S))
_print(music_data_validate)
_print(music_data_validate.shape)
music_labels_validate = np.array([np.array([1.0, 0.0])] * count_per_type_validate)

with open(f'{working_dir}/abooks_train.raw', "rb") as file:
    raw_abooks_data_train = file.read()

_print(len(raw_abooks_data_train))
abooks_data_train = np.frombuffer(raw_abooks_data_train, dtype=np.uint8).astype(float)
_print(abooks_data_train)
_print(np.min(abooks_data_train), np.max(abooks_data_train), np.mean(abooks_data_train))
abooks_data_train = np.reshape(abooks_data_train, (count_per_type_train, S))
_print(abooks_data_train)
_print(abooks_data_train.shape)
abooks_labels_train = np.array([np.array([0.0, 1.0])] * count_per_type_train)

with open(f'{working_dir}/abooks_validate.raw', "rb") as file:
    raw_abooks_data_validate = file.read()

_print(len(raw_abooks_data_validate))
abooks_data_validate = np.frombuffer(raw_abooks_data_validate, dtype=np.uint8).astype(float)
_print(abooks_data_validate)
_print(np.min(abooks_data_validate), np.max(abooks_data_validate), np.mean(abooks_data_validate))
abooks_data_validate = np.reshape(abooks_data_validate, (count_per_type_validate, S))
_print(abooks_data_validate)
_print(abooks_data_validate.shape)
abooks_labels_validate = np.array([np.array([0.0, 1.0])] * count_per_type_validate)

print('loading data done')

print('making train and test')

_print(np.mean(music_data_train), np.mean(abooks_data_train), np.mean(music_labels_train), np.mean(abooks_labels_validate))

X_train = np.row_stack((music_data_train, abooks_data_train))
y_train = np.concatenate((music_labels_train, abooks_labels_train))
X_validate = np.row_stack((music_data_validate, abooks_data_validate))
y_validate = np.concatenate((music_labels_validate, abooks_labels_validate))

print('making train and test done')


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

(H_train, W) = X_train.shape
(H_validate, W) = X_validate.shape

print(W, H_train, H_validate)

print('making tensors')

X_train = torch.FloatTensor(X_train).to(device)
X_validate = torch.FloatTensor(X_validate).to(device)
y_train = torch.FloatTensor(y_train).to(device)
y_validate = torch.FloatTensor(y_validate).to(device)

X_train = X_train.divide(255.0).subtract(0.5)#.multiply(ktan)
X_validate = X_validate.divide(255.0).subtract(0.5)#.multiply(ktan)
print(X_train.cpu().min(), X_train.cpu().max(), X_train.cpu().mean())
print(X_validate.cpu().min(), X_validate.cpu().max(), X_validate.cpu().mean())

print('making tensors done')

# S = 24000
# conv_kernel = 999
# conv_stride = 111
# conv_padding = 99
# => Sc = 209 + 1 = 210


class FCNet(torch.nn.Module):
    def __init__(self, S=S,
                 conv1_channels=128, conv1_padding=12,
                 conv1_kernel=99, conv1_stride=11,
                 pool1_kernel=128,
                 conv2_channels=128, conv2_kernel=19,
                 conv2_padding=15, conv2_stride=9,
                 pool2_kernel=128):
        super(FCNet, self).__init__()

        # self.conv1_channels = conv1_channels
        self.pool1_kernel = pool1_kernel
        # self.conv2_channels = conv2_channels
        self.pool2_kernel = pool2_kernel

        self.S = S

        self.Sc1 = (S - conv1_kernel + 2 * conv1_padding) // conv1_stride + 1
        print('Sc1', self.Sc1)
        self.S1 = conv1_channels * self.Sc1 // pool1_kernel
        print('S1', self.S1)
        self.Sc2 = (self.S1 - conv2_kernel + 2 * conv2_padding) // conv2_stride + 1
        print('Sc2', self.Sc2)
        self.S2 = conv2_channels * self.Sc2 // pool2_kernel
        print('S2', self.S2)
        self.n_hidden_neurons = self.S2
        # self.n_hidden_neurons = 240
        # self.n_hidden_neurons = self.S1
        print('n_hidden_neurons', self.n_hidden_neurons)

        # self.fc0 = torch.nn.Linear(self.S, self.n_hidden_neurons)
        # self.activ0 = torch.nn.Tanh()
        self.conv1 = torch.nn.Conv1d(1, conv1_channels, conv1_kernel, stride=conv1_stride, padding=conv1_padding, groups=1)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=pool1_kernel)
        self.conv2 = torch.nn.Conv1d(1, conv2_channels, conv2_kernel, stride=conv2_stride, padding=conv2_padding, groups=1)
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=pool2_kernel)
        self.fc1 = torch.nn.Linear(self.S2, self.n_hidden_neurons)
        self.activ1 = torch.nn.Tanh()
        # self.conv2 = torch.nn.Conv2d(1, conv_channels, conv_kernel, stride=conv_stride, padding=conv_padding, groups=1)
        # self.maxpool2 = torch.nn.MaxPool2d(kernel_size=pool_kernel)
        # self.fc2 = torch.nn.Linear(self.Sl, self.n_hidden_neurons)
        # self.activ2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(self.n_hidden_neurons, self.n_hidden_neurons)
        self.activ3 = torch.nn.Tanh()
        self.fc4 = torch.nn.Linear(self.n_hidden_neurons, N)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        _print('forward')
        batch_size = list(x.shape)[0]
        # x = self.fc0(x)
        # x = self.activ0(x * ktan) / ktan
        v0 = x[0, :]
        x = x.reshape(batch_size, 1, S)
        v1 = x[0, 0, :]
        _print(torch.sum(v1.cpu() - v0.cpu()))
        _print(x.shape)
        x = self.conv1(x)
        _print(x.shape)
        v0 = x[0, 0, :]
        # x = x.reshape(-1, 1, self.pool1_kernel)
        x = x.transpose(1, 2)
        v1 = x[0, :, 0]
        _print(torch.sum(v1.cpu() - v0.cpu()))
        _print(x.shape)
        x = self.maxpool1(x)
        _print(x.shape)
        x = x.reshape(batch_size, 1, self.S1)
        x = self.conv2(x)
        _print(x.shape)
        v0 = x[0, 0, :]
        # x = x.reshape(-1, 1, self.pool2_kernel)
        x = x.transpose(1, 2)
        v1 = x[0, :, 0]
        _print(torch.sum(v1.cpu() - v0.cpu()))
        _print(x.shape)
        x = self.maxpool2(x)
        _print(x.shape)
        x = x.reshape(batch_size, 1, self.S2)
        _print(x.shape)
        x = self.fc1(x)
        x = self.activ1(x * ktan) / ktan
        x = self.fc3(x)
        x = self.activ3(x * ktan) / ktan
        x = self.fc4(x)
        x = x.reshape(batch_size, N)
        _print(x.shape)
        # x = self.sm(x)
        # _print(x.shape)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


print('preparing neural networking')
fc_net = FCNet().to(device)

epoch = last_epoch

if epoch >= 0:
    fn_weights = f'{working_dir}/weights/model_weights_epoch_{epoch}.pth'
    fc_net.load_state_dict(torch.load(fn_weights, map_location=device))
    fc_net.eval()

optimizer = torch.optim.Adam(fc_net.parameters(), lr=lr)

if USE_IPEX:
    fc_net, optimizer = ipex.optimize(fc_net, optimizer=optimizer, dtype=torch.float32)

print('preparing neural networking done')

# loss = torch.nn.CrossEntropyLoss()

col_summator = torch.ones(2, 1).to(device)


def loss(pred, target):
    _print(pred.shape)
    _print(target.shape)
    count = float(list(target.shape)[0])
    score = (pred * target) @ col_summator
    _print(score.shape, score.min(), score.max())
    score = score.sum().divide(count)
    return (1.0 - score + 1e-8).log10()


epochs = []
accuracies = []

t0 = time.time()

print('starting training')
while epoch < end_epoch:
    epoch += 1
    order = np.random.permutation(len(X_train))
    for start_index in range(0, len(X_train), the_batch_size):
        # t00 = time.time()

        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index + the_batch_size]

        x_batch = X_train[batch_indexes].to(device)
        y_batch = y_train[batch_indexes].to(device)

        # preds = fc_net.forward(x_batch)
        preds = fc_net.inference(x_batch)

        loss_value = loss(preds, y_batch)
        _print(loss_value.cpu())
        loss_value.backward()
        optimizer.step()

        # print(start_index, loss_value)
        # t01 = time.time()
        # print('batch time', t01 - t00)

        # if epoch % 5 == 0:
    if True:
        test_preds = fc_net.forward(X_validate)
        test_preds = test_preds.argmax(dim=1)
        y_validate_numbers = y_validate.argmax(dim=1)
        _print((test_preds.cpu() == 0.0).sum())
        _print((test_preds.cpu() == 1.0).sum())
        _print((test_preds.cpu() == 2.0).sum())
        _print((test_preds.cpu() == 0).sum())
        _print((test_preds.cpu() == 1).sum())
        _print((test_preds.cpu() == 2).sum())
        accuracy = (test_preds.cpu() == y_validate_numbers.cpu()).float().mean().item()
        print(epoch, accuracy, loss_value.cpu().item())
        epochs.append(epoch)
        accuracies.append(accuracy)
        t1 = time.time()
        print('time', t1 - t0)
        with open(accuracy_log, "a") as f:
            f.write(f'{epoch};{accuracy};{loss_value.cpu()};{t1 - t0}\n')
        t0 = t1

    if epoch % 10 == 0:
        fn_weights = f'{weights_dir}/model_weights_epoch_{epoch}.pth'
        torch.save(fc_net.state_dict(), fn_weights)