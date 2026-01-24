#!/usr/bin/env python

#   Copyright (C) 2025-2026 Simon Crase

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''
    This module defines an Autoencoder class
'''

from abc import ABC, abstractmethod
from unittest import TestCase, main
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as tr
from torch.utils.data import DataLoader

class AutoEncoder(nn.Module, ABC):
    '''
    This class represents a network that acts as an autoencoder

    Data members:
        encoder      The encoder network
        decoder      Decoder nwtwork
        bottleneck   Size of bottleneck
    '''

    def __init__(self, width=28, height=28, encoder=nn.Sequential(), decoder=nn.Sequential(),bottleneck=3):
        super().__init__()
        self.input_size = width * height
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self,x):
        '''
        Encode data down to bottleneck
        '''
        return self.encoder(x.reshape(-1, 784))

    def decode(self,x):
        '''
        Expand data from bottleneck
        '''
        return self.decoder(x)

    def get_batch_loss(self, batch):
        '''

        Use encode + decode to predict loss.

        Parameters:
            batch       One batch of data

        I'm following https://www.geeksforgeeks.org/machine-learning/auto-encoders/
        and using MSE Loss (we want output to match input)
        '''
        images, _ = batch
        out = self(images)
        return F.mse_loss(out, torch.reshape(images, out.shape))

    def save(self, name):
        '''
        Used to save weights
        '''
        torch.save(self.state_dict(), f'{name}.pth')

    def load(self, file):
        '''
        Used to recall a previous set of weights
        '''
        self.load_state_dict(torch.load(file))


class SimpleAutoEncoder(AutoEncoder):
    '''
    This class represets an autoencoder based on a perceptron
    '''

    def __init__(self, width=28, height=28,bottleneck=3):
        super().__init__(width=width,
                         height=height,
                         encoder=nn.Sequential(
                             nn.Linear(784, 148),
                             nn.ReLU(),
                             nn.Linear(148, 28),
                             nn.ReLU(),
                             nn.Linear(28, bottleneck),
                             nn.ReLU()
                         ),
                         decoder=nn.Sequential(
                            nn.Linear(bottleneck, 28),
                            nn.ReLU(),
                             nn.Linear(28, 148),
                             nn.ReLU(),
                             nn.Linear(148, 784),
                             nn.Sigmoid()
                         ),
                         bottleneck=bottleneck)

class ShallowAutoEncoder(AutoEncoder):
    '''
    This class represets an autoencoder with minimal encoder and decoder
    '''

    def __init__(self, width=28, height=28,bottleneck=14):
        super().__init__(width=width,
                         height=height,
                         encoder=nn.Sequential(
                             nn.Linear(width*height, bottleneck),
                             nn.ReLU()
                         ),
                         decoder=nn.Sequential(
                            nn.Linear(bottleneck, width*height),
                            nn.ReLU(),
                            nn.Sigmoid()
                         ),
                         bottleneck=bottleneck)

class CNNAutoEncoder(AutoEncoder):
    '''
    This class represets an autoencoder using Convolutional layers
    '''
    def __init__(self, width=28, height=28):
        super().__init__(width=width,
                         height=height,
                         encoder=nn.Sequential(
                             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             nn.ReLU(),
                             nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             nn.ReLU(),
                             nn.Sigmoid() #Convergence is poor if this is left out
                         ),
                         decoder=nn.Sequential(
                             nn.ConvTranspose2d(in_channels=1, out_channels=4, kernel_size=11, stride=2),
                             nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=12, stride=2),
                             nn.MaxPool2d(kernel_size=2,stride=2),
                             nn.ReLU()
                         ))



    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoEncoderFactory:
    def get_choices(self):
        return ['perceptron', 'cnn', 'shallow']

    def get_default(self):
        return 'perceptron'

    def instantiate(self, args):
        match args.implementation:
            case 'perceptron':
                return SimpleAutoEncoder(width=args.width, height=args.height,bottleneck=args.bottleneck)
            case 'cnn':
                return CNNAutoEncoder(width=args.width, height=args.height)
            case 'shallow':
                return ShallowAutoEncoder(width=args.width, height=args.height,bottleneck=args.bottleneck)

    def create(self, args):
        '''
        Instantiate an autoencoder, and, optionally, reload weights from a file

        Parameters:
            restart   Optional name for a file from which to load weights
        '''
        product = self.instantiate(args)

        if args.restart:
            restart_path = Path(args.restart).with_suffix('.pth')
            product.load(restart_path)
            print(f'Reloaded parameters from {restart_path}')

        return product


class TestAutoEncoder(TestCase):
    class Args:
        def __init__(self):
            self.implementation = 'cnn'
            self.width = 28
            self.height = 28

    def setUp(self):
        self.factory = AutoEncoderFactory()
        self.args = self.Args()

    def test_encode_decode(self):
        '''
        Verify that encoder/decoder preserves shape
        '''
        dataset = MNIST(root='./data', download=True, transform=tr.ToTensor())
        train_loader = DataLoader(dataset, 128)
        ae = self.factory.instantiate(self.args)

        for batch in train_loader:
            images, _ = batch
            encoded = ae.encode(images)
            batch_size,channels,h,w = encoded.shape
            self.assertEqual(128,batch_size)
            self.assertEqual(1,channels)
            self.assertEqual(7,h)
            self.assertEqual(7,w)
            decoded = ae.decode(encoded)
            self.assertEqual(images.shape,decoded.shape)
            return

if __name__ == '__main__':
    main()
