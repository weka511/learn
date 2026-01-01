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

'''Train an autoencoder against MNIST data'''

from argparse import ArgumentParser
from os.path import splitext, join
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc
import numpy as np
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as tr
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam
from autoencoder import AutoEncoderFactory
from utils import Logger, get_seed, user_has_requested_stop, ensure_we_can_save


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


def parse_args(factory):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['train', 'test'],
                        default='train', help='Chooses between training or testing')
    parser.add_argument('--implementation', choices=factory.get_choices(), default=factory.get_default())
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
    training_group.add_argument('--width', default=28, type=int, help='Width of each image in pixels')
    training_group.add_argument('--height', default=28, type=int, help='Height of each image in pixels')

    test_group = parser.add_argument_group('Parameters for --action test')
    test_group.add_argument('--file', default=None, help='Used to load weights')
    test_group.add_argument('--nrows',default=7,type=int, help = 'Number of rows to display')
    test_group.add_argument('--ncols',default=7,type=int, help = 'Number of images to display in each row')

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
    return f'{Path(__file__).stem}'


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

def generate_samples(images,nrows=4,ncols=3):
    '''
    Used to draw samples from a collection of images
    '''
    m,_,_,_ = images.shape
    samples = rng.choice(m,nrows*ncols,replace=False)
    image_index = 0
    for i in range(len(samples)):
        yield samples[image_index]
        image_index += 1

if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    fig = figure(figsize=(24, 12))
    start = time()
    auto_encoder_factory = AutoEncoderFactory()
    args = parse_args(auto_encoder_factory)
    seed = get_seed(args.seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    match args.action:
        case 'train':
            auto_encoder = auto_encoder_factory.create(args)
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

            ax.set_title(f'{Path(__file__).stem.title()}')
            ax.set_ylabel('Loss')
            ax.set_xlabel('Step')

            fig.savefig(join(args.figs, get_file_name(args)))

        case test:
            auto_encoder = auto_encoder_factory.create(args)
            auto_encoder.load('./params/aetrain.pth')
            dataset = MNIST(root=args.data, download=True, train=False, transform=tr.ToTensor())
            loader = DataLoader(dataset, 128)
            m = rng.choice(len(loader)-1)
            k = 0
            for batch in loader:
                if k == m:
                    images, _ = batch
                    processed = auto_encoder(images)
                    samples = generate_samples(images,nrows=args.nrows,ncols=args.ncols)
                    for i in range(args.nrows):
                        for j in range(args.ncols):
                            sample = next(samples)
                            subplot_index = 2*args.ncols*i + 2*j
                            ax1 = fig.add_subplot(args.nrows,2*args.ncols,subplot_index+1)
                            ax1.imshow(images[sample].squeeze(), cmap='gray')
                            ax1.get_xaxis().set_visible(False)
                            ax1.get_yaxis().set_visible(False)
                            ax2 = fig.add_subplot(args.nrows,2*args.ncols,subplot_index+2)
                            ax2.imshow(processed[sample].detach().numpy().squeeze(), cmap='gray')
                            ax2.get_xaxis().set_visible(False)
                            ax2.get_yaxis().set_visible(False)

                    fig.suptitle(f'Batch {m}')
                    break
                else:
                    k += 1

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
