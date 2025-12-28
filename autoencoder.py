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

from argparse import ArgumentParser
from os.path import splitext,join
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

class AutoEncoder(nn.Module):

    '''
    Determine number of nodes in each layer

    Parameters:
        input_size
        reduced
        nsteps
    '''
    @staticmethod
    def create_sizes(input_size,reduced,nsteps):
        factor = (reduced/input_size)**(1/nsteps)
        product = [input_size]
        for i in range(nsteps):
            product.append(int(product[-1]*factor))
        product[-1] = reduced
        product += product[::-1][1:]
        return product

    '''
    Create list of layers for Autoencoder

    Parameters:
        sizes
    '''
    @staticmethod
    def create_layers(sizes):
        product = []
        for a,b in zip(sizes[:-1],sizes[1:]):
            product.append(nn.Linear(a,b))
            product.append(nn.ReLU())
        return product

    def __init__(self, width=28, height=28,reduced = 28,nsteps=2):
        super().__init__()
        self.input_size = width*height
        self.model = nn.Sequential(*AutoEncoder.create_layers(AutoEncoder.create_sizes(self.input_size,reduced,nsteps)))

    def forward(self, xb):
        return self.model(xb.reshape(-1, self.input_size))

    def get_batch_loss(self, batch):
        images, _ = batch
        out = self(images)
        return F.cross_entropy(out, torch.reshape(images,out.shape))

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
    training_group.add_argument('--steps', default=5, type=int, help='Number of steps to an epoch')
    training_group.add_argument('--params', default='./params', help='Location for storing plot files')
    training_group.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    training_group.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    training_group.add_argument('--optimizer', choices=OptimizerFactory.choices, default=OptimizerFactory.get_default(),
                                help='Optimizer to be used for training')
    training_group.add_argument('--restart', default=None, help='Restart from saved parameters')
    training_group.add_argument('--nsteps', default=2, type=int, help='Number of steps to an epoch')
    training_group.add_argument('--reduced', default=28, type=int, help='Number of steps to an epoch')

    test_group = parser.add_argument_group('Parameters for --action test')
    test_group.add_argument('--file', default=None, help='Used to load weights')

    shared_group = parser.add_argument_group('General Parameters')
    shared_group.add_argument('--data', default='./data', help='Location of data files')
    shared_group.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    shared_group.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    shared_group.add_argument('--figs', default='./figs', help='Location for storing plot files')
    shared_group.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
    return parser.parse_args()

def training_step(batch,optimizer):
    loss = auto_encoder.get_batch_loss(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def get_file_name(args):
    return f'{Path(__file__).stem}-{args.reduced}-{args.nsteps}'

def get_moving_average(history,window_size = 11):
    kernel = np.ones(window_size) / window_size
    return np.convolve(history, kernel, mode='valid')

if __name__=='__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    fig = figure(figsize=(24, 12))
    start = time()
    args = parse_args()
    seed = get_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    auto_encoder = AutoEncoder()
    optimizer = OptimizerFactory.create(auto_encoder, args)
    dataset = MNIST(root=args.data, download=True, transform=tr.ToTensor())
    train_data, validation_data = random_split(dataset, [50000, 10000])
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, args.batch_size, shuffle=False)
    history = []
    for epoch in range(args.N):
        print (f'Epoch {epoch} of {args.N}')
        for batch in train_loader:
            training_step(batch,optimizer)
        for batch in validation_loader:
            history.append(float(auto_encoder.get_batch_loss(batch).detach()))
        checkpoint_file_name = join(args.params,get_file_name(args))
        ensure_we_can_save(checkpoint_file_name)
        auto_encoder.save(checkpoint_file_name)

    moving_average= get_moving_average(history,window_size = 11)
    xs = np.arange(0,len(history))
    skip = (len(history) - len(moving_average))//2
    x1s = xs[skip:]
    tail_count = len(x1s) - len(moving_average)
    x1s = x1s[:-tail_count]

    ax = fig.add_subplot(1,1,1)
    ax.plot(xs,history,c='xkcd:blue',label='Loss')
    ax.plot(x1s,moving_average,c='xkcd:blue',linestyle = 'dotted',label='Average Loss')
    ax.legend()

    ax.set_title(f'reduced = {args.reduced}, nsteps={args.nsteps}')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Step')

    fig.savefig(join(args.figs, get_file_name(args)))

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
