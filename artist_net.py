import torch
import random
import numpy as np
import numpy.core.defchararray as np_f
import csv
import math

from debug import _print

version_name = 'artist_v2'

artist_count = 3

lr = 3e-3
ktan = 0.3

sample_len = 24000

# S = 24000
# conv_kernel = 999
# conv_stride = 111
# conv_padding = 99
# => Sc = 209 + 1 = 210


def make_actual_nn():
    if version_name == 'artist':
        return ArtistNet()
    elif version_name == 'artist_v2':
        return ArtistNetV2()
    elif version_name == 'artist_v3':
        return ArtistNetV3()
    else:
        return None


class ArtistNet(torch.nn.Module):
    def __init__(self, S=sample_len, N=artist_count,
                 conv1_channels=36, conv1_padding=12,
                 conv1_kernel=99, conv1_stride=11,
                 pool1_kernel=17,
                 conv2_channels=36, conv2_kernel=8,
                 conv2_padding=4, conv2_stride=4,
                 pool2_kernel=33):
        super(ArtistNet, self).__init__()

        # self.conv1_channels = conv1_channels
        self.pool1_kernel = pool1_kernel
        # self.conv2_channels = conv2_channels
        self.pool2_kernel = pool2_kernel

        self.S = S
        self.N = N

        self.Sc1 = (S - conv1_kernel + 2 * conv1_padding) // conv1_stride + 1
        print('Sc1', self.Sc1)
        # self.S1 = conv1_channels * self.Sc1 // pool1_kernel
        # print('S1', self.S1)
        self.Sf1 = self.Sc1 // pool1_kernel
        print('Sf1', self.Sf1)
        self.Sc2 = (self.Sf1 - conv2_kernel + 2 * conv2_padding) // conv2_stride + 1
        print('Sc2', self.Sc2)
        self.Sf2 = self.Sc2 // pool2_kernel
        print('Sf2', self.Sf2)
        self.S2 = conv2_channels * self.Sf2
        print('S2', self.S2)
        self.n_hidden_neurons = self.S2 * 6
        # self.n_hidden_neurons = 240
        # self.n_hidden_neurons = self.S1
        print('n_hidden_neurons', self.n_hidden_neurons)

        # self.fc0 = torch.nn.Linear(self.S, self.n_hidden_neurons)
        # self.activ0 = torch.nn.Tanh()
        self.conv1 = torch.nn.Conv1d(1, conv1_channels, conv1_kernel, stride=conv1_stride, padding=conv1_padding, groups=1)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=pool1_kernel)
        self.conv2 = torch.nn.Conv1d(conv1_channels, conv2_channels, conv2_kernel, stride=conv2_stride, padding=conv2_padding, groups=conv1_channels)
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=pool2_kernel)
        self.fc1 = torch.nn.Linear(self.S2, self.n_hidden_neurons)
        self.activ1 = torch.nn.Tanh()
        # self.conv2 = torch.nn.Conv2d(1, conv_channels, conv_kernel, stride=conv_stride, padding=conv_padding, groups=1)
        # self.maxpool2 = torch.nn.MaxPool2d(kernel_size=pool_kernel)
        # self.fc2 = torch.nn.Linear(self.Sl, self.n_hidden_neurons)
        # self.activ2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(self.n_hidden_neurons, self.n_hidden_neurons)
        self.activ3 = torch.nn.Tanh()
        self.fc4 = torch.nn.Linear(self.n_hidden_neurons, 2*N)
        self.maxpool3 = torch.nn.MaxPool1d(kernel_size=2)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        _print('forward')
        batch_size = list(x.shape)[0]
        x = x.reshape(batch_size, 1, self.S)
        x = self.conv1(x)
        _print(x.shape)
        x = self.maxpool1(x)
        _print(x.shape)
        x = self.conv2(x)
        _print(x.shape)
        x = self.maxpool2(x)
        _print(x.shape)
        x = x.transpose(1, 2)
        _print(x.shape)
        x = self.fc1(x)
        _print(x.shape)
        x = self.activ1(x * ktan) / ktan
        x = self.fc3(x)
        _print(x.shape)
        x = self.activ3(x * ktan) / ktan
        x = self.fc4(x)
        _print(x.shape)
        x = self.maxpool3(x)
        _print(x.shape)
        x = x.reshape(batch_size, self.N)
        _print(x.shape)
        # x = self.sm(x)
        # _print(x.shape)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


class ArtistNetV2(torch.nn.Module):
    def __init__(self, S=sample_len, N=artist_count,
                 conv1_channels=36, conv1_padding=12,
                 conv1_kernel=99, conv1_stride=11,
                 pool1_kernel=17,
                 conv2_channels=36, conv2_kernel=8,
                 conv2_padding=4, conv2_stride=4,
                 pool2_kernel=33):
        super(ArtistNetV2, self).__init__()

        # self.conv1_channels = conv1_channels
        self.pool1_kernel = pool1_kernel
        # self.conv2_channels = conv2_channels
        self.pool2_kernel = pool2_kernel

        self.S = S
        self.N = N

        self.Sc1 = (S - conv1_kernel + 2 * conv1_padding) // conv1_stride + 1
        print('Sc1', self.Sc1)
        # self.S1 = conv1_channels * self.Sc1 // pool1_kernel
        # print('S1', self.S1)
        self.Sf1 = self.Sc1 // pool1_kernel
        print('Sf1', self.Sf1)
        self.Sc2 = (self.Sf1 - conv2_kernel + 2 * conv2_padding) // conv2_stride + 1
        print('Sc2', self.Sc2)
        self.Sf2 = self.Sc2 // pool2_kernel
        print('Sf2', self.Sf2)
        self.S2 = conv2_channels * self.Sf2
        print('S2', self.S2)
        self.n_hidden_neurons = self.S2 * 6
        # self.n_hidden_neurons = 240
        # self.n_hidden_neurons = self.S1
        print('n_hidden_neurons', self.n_hidden_neurons)

        # self.fc0 = torch.nn.Linear(self.S, self.n_hidden_neurons)
        # self.activ0 = torch.nn.Tanh()
        self.conv1 = torch.nn.Conv1d(1, conv1_channels, conv1_kernel, stride=conv1_stride, padding=conv1_padding, groups=1)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=pool1_kernel)
        self.conv2 = torch.nn.Conv1d(conv1_channels, conv2_channels, conv2_kernel, stride=conv2_stride, padding=conv2_padding, groups=conv1_channels)
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=pool2_kernel)
        self.fc1 = torch.nn.Linear(self.S2, self.n_hidden_neurons)
        self.activ1 = torch.nn.Tanh()
        # self.conv2 = torch.nn.Conv2d(1, conv_channels, conv_kernel, stride=conv_stride, padding=conv_padding, groups=1)
        # self.maxpool2 = torch.nn.MaxPool2d(kernel_size=pool_kernel)
        # self.fc2 = torch.nn.Linear(self.Sl, self.n_hidden_neurons)
        # self.activ2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(self.n_hidden_neurons, self.n_hidden_neurons)
        self.activ3 = torch.nn.Tanh()
        self.fc4 = torch.nn.Linear(self.n_hidden_neurons, N * (N + 2))
        self.maxpool3 = torch.nn.MaxPool1d(kernel_size=N+2)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        _print('forward')
        batch_size = list(x.shape)[0]
        x = x.reshape(batch_size, 1, self.S)
        x = self.conv1(x)
        _print(x.shape)
        x = self.maxpool1(x)
        _print(x.shape)
        x = self.conv2(x)
        _print(x.shape)
        x = self.maxpool2(x)
        _print(x.shape)
        x = x.transpose(1, 2)
        _print(x.shape)
        x = self.fc1(x)
        _print(x.shape)
        x = self.activ1(x * ktan) / ktan
        x = self.fc3(x)
        _print(x.shape)
        x = self.activ3(x * ktan) / ktan
        x = self.fc4(x)
        _print(x.shape)
        x = self.maxpool3(x)
        _print(x.shape)
        x = x.reshape(batch_size, self.N)
        # x = self.sm(x)
        # _print(x.shape)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


class ArtistNetV3(torch.nn.Module):
    def __init__(self, S=sample_len, N=artist_count,
                 conv1_channels=36, conv1_padding=12,
                 conv1_kernel=99, conv1_stride=11,
                 pool1_kernel=17,
                 conv2_channels=36, conv2_kernel=8,
                 conv2_padding=4, conv2_stride=4,
                 pool2_kernel=33):
        super(ArtistNetV3, self).__init__()

        # self.conv1_channels = conv1_channels
        self.pool1_kernel = pool1_kernel
        # self.conv2_channels = conv2_channels
        self.pool2_kernel = pool2_kernel

        self.S = S
        self.N = N

        self.Sc1 = (S - conv1_kernel + 2 * conv1_padding) // conv1_stride + 1
        print('Sc1', self.Sc1)
        # self.S1 = conv1_channels * self.Sc1 // pool1_kernel
        # print('S1', self.S1)
        self.Sf1 = self.Sc1 // pool1_kernel
        print('Sf1', self.Sf1)
        self.Sc2 = (self.Sf1 - conv2_kernel + 2 * conv2_padding) // conv2_stride + 1
        print('Sc2', self.Sc2)
        self.Sf2 = self.Sc2 // pool2_kernel
        print('Sf2', self.Sf2)
        self.S2 = conv2_channels * self.Sf2
        print('S2', self.S2)
        self.n_hidden_neurons = self.S2 * 6
        # self.n_hidden_neurons = 240
        # self.n_hidden_neurons = self.S1
        print('n_hidden_neurons', self.n_hidden_neurons)

        # self.fc0 = torch.nn.Linear(self.S, self.n_hidden_neurons)
        # self.activ0 = torch.nn.Tanh()
        self.bn = torch.nn.BatchNorm1d(1)
        self.conv1 = torch.nn.Conv1d(1, conv1_channels, conv1_kernel, stride=conv1_stride, padding=conv1_padding, groups=1)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=pool1_kernel)
        self.conv2 = torch.nn.Conv1d(conv1_channels, conv2_channels, conv2_kernel, stride=conv2_stride, padding=conv2_padding, groups=conv1_channels)
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=pool2_kernel)
        self.fc1 = torch.nn.Linear(self.S2, self.n_hidden_neurons)
        self.activ1 = torch.nn.Tanh()
        # self.conv2 = torch.nn.Conv2d(1, conv_channels, conv_kernel, stride=conv_stride, padding=conv_padding, groups=1)
        # self.maxpool2 = torch.nn.MaxPool2d(kernel_size=pool_kernel)
        # self.fc2 = torch.nn.Linear(self.Sl, self.n_hidden_neurons)
        # self.activ2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(self.n_hidden_neurons, self.n_hidden_neurons)
        self.activ3 = torch.nn.Tanh()
        self.fc4 = torch.nn.Linear(self.n_hidden_neurons, N * (N + 2))
        self.maxpool3 = torch.nn.MaxPool1d(kernel_size=N+2)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        _print('forward')
        batch_size = list(x.shape)[0]
        x = x.reshape(batch_size, 1, self.S)
        x = self.bn(x)
        x = self.conv1(x)
        _print(x.shape)
        x = self.maxpool1(x)
        _print(x.shape)
        x = self.conv2(x)
        _print(x.shape)
        x = self.maxpool2(x)
        _print(x.shape)
        x = x.transpose(1, 2)
        _print(x.shape)
        x = self.fc1(x)
        _print(x.shape)
        x = self.activ1(x * ktan) / ktan
        x = self.fc3(x)
        _print(x.shape)
        x = self.activ3(x * ktan) / ktan
        x = self.fc4(x)
        _print(x.shape)
        x = self.maxpool3(x)
        _print(x.shape)
        x = x.reshape(batch_size, self.N)
        # x = self.sm(x)
        # _print(x.shape)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x