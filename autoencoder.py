#!/usr/bin/env python

#   Copyright (C) 2025 Simon Crase

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


class AutoEncoder(nn.Module, ABC):
    '''
    This class represets a network that acts as an autoencoder

    Data members:
        encoder
        decoder
    '''

    def __init__(self, width=28, height=28, encoder=nn.Sequential(), decoder=nn.Sequential()):
        super().__init__()
        self.input_size = width * height
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        x1 = x.shape
        x = self.encoder(x)
        x2 = x.shape
        x = self.decoder(x)
        x3 = x.shape
        return x

    def get_batch_loss(self, batch):
        '''
        I'm following https://www.geeksforgeeks.org/machine-learning/auto-encoders/
        and using MSE Loss
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

    def __init__(self, width=28, height=28):
        super().__init__(width=width,
                         height=height,
                         encoder=nn.Sequential(
                             nn.Linear(784, 148),
                             nn.ReLU(),
                             nn.Linear(148, 28),
                             nn.ReLU()
                         ),
                         decoder=nn.Sequential(
                             nn.Linear(28, 148),
                             nn.ReLU(),
                             nn.Linear(148, 784),
                             nn.Sigmoid()
                         ))


class CNNAutoEncoder(AutoEncoder):
    def __init__(self, width=28, height=28):
        super().__init__(width=width,
                         height=height,
                         encoder=nn.Sequential(
                             nn.Conv2d(1, 16, kernel_size=3, padding=1),
                             nn.Conv2d(16, 4, kernel_size=3, padding=1),
                             nn.MaxPool2d(2, 2),
                             nn.ReLU(),
                             nn.Sigmoid()
                         ),
                         decoder=nn.Sequential(
                             nn.ConvTranspose2d(16, 4, 2, stride=2),
                             nn.ConvTranspose2d(16, 1, 2, stride=2)
                         ))


class AutoEncoderFactory:
    def get_choices(self):
        return ['perceptron', 'cnn']

    def get_default(self):
        return 'perceptron'

    def instantiate(self, args):
        match args.implementation:
            case 'perceptron':
                return SimpleAutoEncoder(width=args.width, height=args.height)
            case 'cnn':
                return CNNAutoEncoder(width=args.width, height=args.height)

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

    def test_instantiation(self):
        ae = self.factory.instantiate(self.args)
        print (ae)


if __name__ == '__main__':
    main()
