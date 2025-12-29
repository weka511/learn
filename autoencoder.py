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

'''Train an autoencoder against MNIST data'''

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from os.path import splitext, join
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as tr
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.optim import SGD, Adam
from utils import Logger, get_seed, user_has_requested_stop, ensure_we_can_save


class AutoEncoder(nn.Module,ABC):
    '''
    This class represets a network that acts as an autoencoder
    '''
    @staticmethod
    def create_sizes(input_size, bottleneck, nlayers):
        '''
        Determine number of nodes in each layer

        Parameters:
            input_size
            bottleneck
            nlayers
        '''
        factor = (bottleneck / input_size)**(1 / nlayers)
        product = [input_size]
        for i in range(nlayers):
            product.append(int(product[-1] * factor))
        product[-1] = bottleneck
        product += product[::-1][1:]
        return product

    @staticmethod
    def create_layers(sizes):
        '''
        Create list of layers for Autoencoder

        Parameters:
            sizes
        '''
        product = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            product.append(nn.Linear(a, b))
            product.append(nn.ReLU())
        return product



    def __init__(self, width=28, height=28, bottleneck=28, nlayers=2):
        super().__init__()
        self.input_size = width * height
        self.model = nn.Sequential(*AutoEncoder.create_layers(AutoEncoder.create_sizes(self.input_size, bottleneck, nlayers)))

    def forward(self, xb):
        return self.model(xb.reshape(-1, self.input_size))

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
    def __init__(self, width=28, height=28, bottleneck=28, nlayers=2):
        super().__init__(width=width, height=height, bottleneck=bottleneck, nlayers=nlayers)

class AutoEncoderFactory:
    @staticmethod
    def create(width=28, height=28, bottleneck=28, nlayers=2,restart=None):
        '''
        Instantiate an autoencoder, and, optionally, reload weights from a file

        Parameters:
            restart   Optional name for a file from which to load weights
        '''
        product = SimpleAutoEncoder(width=width, height=height, bottleneck=bottleneck, nlayers=nlayers)
        if restart:
            restart_path = Path(restart).with_suffix('.pth')
            product.load(restart_path)
            print(f'Reloaded parameters from {restart_path}')
        return product

class OptimizerFactory:
    '''
    This class instantiates optimizers as specified by the command line parameters
    '''

    choices = [
        'SGD',
        'Adam'
    ]

    @staticmethod
    def get_default():
        return OptimizerFactory.choices[1]

    @staticmethod
    def create(model, args):
        match args.optimizer:
            case 'SGD':
                return SGD(model.parameters(), lr=args.lr)
            case 'Adam':
                return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['train', 'test'],
                        default='train', help='Chooses between training or testing')

    training_group = parser.add_argument_group('Parameters for --action train')
    training_group.add_argument('--batch_size', default=128, type=int, help='Number of images per batch')
    training_group.add_argument('--N', default=5, type=int, help='Number of epochs')
    training_group.add_argument('--n', default=5, type=int, help='Number of steps to an epoch')
    training_group.add_argument('--params', default='./params', help='Location for storing plot files')
    training_group.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    training_group.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    training_group.add_argument('--optimizer', choices=OptimizerFactory.choices, default=OptimizerFactory.get_default(),
                                help='Optimizer to be used for training')
    training_group.add_argument('--restart', default=None, help='Restart from saved parameters')
    training_group.add_argument('--nlayers', default=2, type=int, help='Number of layers in encoder (or decoder)')
    training_group.add_argument('--bottleneck', default=28, type=int, help='Number of cells in bottleneck')
    training_group.add_argument('--width', default=28, type=int, help='Width of each image in pixels')
    training_group.add_argument('--height', default=28, type=int, help='Height of each image in pixels')

    test_group = parser.add_argument_group('Parameters for --action test')
    test_group.add_argument('--file', default=None, help='Used to load weights')

    shared_group = parser.add_argument_group('General Parameters')
    shared_group.add_argument('--data', default='./data', help='Location of data files')
    shared_group.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    shared_group.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    shared_group.add_argument('--figs', default='./figs', help='Location for storing plot files')
    shared_group.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
    return parser.parse_args()



def training_step(batch, optimizer):
    loss = auto_encoder.get_batch_loss(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def get_file_name(args):
    return f'{Path(__file__).stem}-{args.bottleneck}-{args.nlayers}'


def get_moving_average(xs, ys, window_size=11):
    '''
    Calculate a moving average

    Parameters:
         xs            Indices of data for plotting
         ys            Data to be plotted
         window_size   Number of points to be included

    Returns:
         x1s    A subset of xs, chosen so average can be plotted on the same scale as xs,ys
         y1s    The moving average
    '''
    kernel = np.ones(window_size) / window_size
    y1s = np.convolve(ys, kernel, mode='valid')
    skip = (len(ys) - len(y1s)) // 2
    x1s = xs[skip:]
    tail_count = len(x1s) - len(y1s)
    x1s = x1s[:-tail_count]
    return x1s, y1s


if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    fig = figure(figsize=(24, 12))
    start = time()
    args = parse_args()
    seed = get_seed(args.seed)
    rng = np.random.default_rng(seed)
    auto_encoder = AutoEncoderFactory.create(width=args.width, height=args.height, bottleneck=args.bottleneck, nlayers=args.nlayers,restart=args.restart)
    optimizer = OptimizerFactory.create(auto_encoder, args)
    dataset = MNIST(root=args.data, download=True, transform=tr.ToTensor())
    train_data, validation_data = random_split(dataset, [50000, 10000])
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, args.batch_size, shuffle=False)
    history = []
    for epoch in range(args.N):
        for _ in range(args.n):
            for batch in train_loader:
                training_step(batch, optimizer)

            validation_losses = [float(auto_encoder.get_batch_loss(batch).detach()) for batch in validation_loader]
            history += validation_losses
        print(f'Epoch {epoch} of {args.N}. Average validation loss = {np.mean(validation_losses)}')

        checkpoint_file_name = join(args.params, get_file_name(args))
        ensure_we_can_save(checkpoint_file_name)
        auto_encoder.save(checkpoint_file_name)
        if user_has_requested_stop():
            break

    xs = np.arange(0, len(history))
    x1s, moving_average = get_moving_average(xs, history)

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, history, c='xkcd:blue', label='Loss')
    ax.plot(x1s, moving_average, c='xkcd:red', label='Average Loss')
    ax.legend()

    ax.set_title(f'{Path(__file__).stem.title()}: bottleneck = {args.bottleneck}, nlayers={args.nlayers}')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Step')

    fig.savefig(join(args.figs, get_file_name(args)))

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
