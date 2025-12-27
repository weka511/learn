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

'''Template for python script using pytorch'''

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
from utils import Logger, get_seed, user_has_requested_stop

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

    test_group = parser.add_argument_group('Parameters for --action test')
    test_group.add_argument('--file', default=None, help='Used to load weights')

    shared_group = parser.add_argument_group('General Parameters')
    shared_group.add_argument('--data', default='./data', help='Location of data files')
    shared_group.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    shared_group.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    shared_group.add_argument('--figs', default='./figs', help='Location for storing plot files')
    shared_group.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
    return parser.parse_args()

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

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
