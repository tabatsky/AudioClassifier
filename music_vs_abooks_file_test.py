import sys
import os
import time

import torch
import random
import numpy as np
import numpy.core.defchararray as np_f
import csv
import math

from sklearn.model_selection import train_test_split

from pydub import AudioSegment



def _print(*args):
    # if True:
    if False:
        print(args)


working_dir = '.'

FRAME_RATE = 8000
CHUNK_SIZE = 24000

last_epoch = 100

S = 24000
N = 2

ktan = 0.03
USE_IPEX = False

if USE_IPEX:
    import intel_extension_for_pytorch as ipex

rnd = random.Random()


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
        _print('Sc1', self.Sc1)
        self.S1 = conv1_channels * self.Sc1 // pool1_kernel
        _print('S1', self.S1)
        self.Sc2 = (self.S1 - conv2_kernel + 2 * conv2_padding) // conv2_stride + 1
        _print('Sc2', self.Sc2)
        self.S2 = conv2_channels * self.Sc2 // pool2_kernel
        _print('S2', self.S2)
        self.n_hidden_neurons = self.S2
        # self.n_hidden_neurons = 240
        # self.n_hidden_neurons = self.S1
        _print('n_hidden_neurons', self.n_hidden_neurons)

        # self.fc0 = torch.nn.Linear(self.S, self.n_hidden_neurons)
        # self.activ0 = torch.nn.Tanh()
        self.conv1 = torch.nn.Conv1d(1, conv1_channels, conv1_kernel, stride=conv1_stride, padding=conv1_padding,
                                     groups=1)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=pool1_kernel)
        self.conv2 = torch.nn.Conv1d(1, conv2_channels, conv2_kernel, stride=conv2_stride, padding=conv2_padding,
                                     groups=1)
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
        _print(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


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
fc_net = FCNet().to(device)

if last_epoch >= 0:
    fn_weights = f'{working_dir}/weights_2d_label/model_weights_epoch_{last_epoch}.pth'
    fc_net.load_state_dict(torch.load(fn_weights, map_location=device))
    fc_net.eval()

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
    while i < 40:
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
    audio_data = np.reshape(audio_data, (40, S))
    _print(audio_data)
    _print(audio_data.shape)

    X = torch.FloatTensor(audio_data).to(device)
    X = X.divide(255.0).subtract(0.5)

    test_preds = fc_net.forward(X)
    _print(test_preds)
    test_preds = test_preds.argmax(dim=1)
    _print(test_preds)
    test_preds = test_preds.sum().item() / 40 * 100

    print(f'(method 1) music: {100 - test_preds}%, audiobook: {test_preds}%')

    pred = fc_net.inference(X)
    p0 = pred[:, 0]
    p1 = pred[:, 1]
    # print(p0.cpu().min(), p1.cpu().min(), (p1 - p0).cpu().min(), (p1 + p0).cpu().min())
    y = (p1 - p0).multiply(0.5).add(0.5).multiply(100).mean().item()
    print(f'(method 2) music: {100 - y}%, audiobook: {y}%')

    p0 = pred[:, 0].cpu().mean().item() * 100
    p1 = pred[:, 1].cpu().mean().item() * 100
    # print(p0.cpu().min(), p1.cpu().min(), (p1 - p0).cpu().min(), (p1 + p0).cpu().min())
    print(f'(method 3) music: {p0}%, audiobook: {p1}%')

if len(sys.argv) > 1:
    the_fn_in_mp3 = sys.argv[1]
    print(the_fn_in_mp3)
    detect_file_type(the_fn_in_mp3)
else:
    music_filenames = os.listdir('music_test')
    for filename in music_filenames:
        the_fn_in_mp3 = f'music_test/{filename}'
        detect_file_type(the_fn_in_mp3)
    abooks_filenames = os.listdir('abooks_test')
    for filename in abooks_filenames:
        the_fn_in_mp3 = f'abooks_test/{filename}'
        detect_file_type(the_fn_in_mp3)