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
    Train an autoencoder against MNIST data
'''

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
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LinearLR
import torch.nn.functional as F
from autoencoder import AutoEncoderFactory
from utils import Logger, get_seed, user_has_requested_stop, ensure_we_can_save, get_moving_average, create_xkcd_colours, sort_labels


class Perceptron(nn.Module):
    '''
    A simple multi layer perceptron for use with encoded data
    '''
    name = 'perceptron'

    def __init__(self, width=7, height=7, n_classes=10):
        super().__init__()
        self.input_size = width * height
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU())

    def forward(self, xb):
        xb = xb.reshape(-1, self.input_size)
        return self.model(xb)

    def training_step(self, batch, encoder, optimizer):
        '''
        Compute loss for one batch of training data and use optimizer to update weights

        Parameters:
            batch      Batch of training data
            encoder    Encode part of autoencoder
            optimizer  Used to reduce loss
        '''
        images, labels = batch
        encoded = encoder.encode(images)
        out = self(encoded)
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


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
        '''
        Instantiate optimizer
        '''
        match args.optimizer:
            case 'SGD':
                text = (f'optimizer=SGD, lr={args.lr}, weight_decay={args.weight_decay},momentum={args.momentum}')
                return SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum), text
            case 'Adam':
                text = f'optimizer=Adam, lr={args.lr}, weight_decay={args.weight_decay}'
                return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay), text


def parse_args(factory):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['train1', 'train2', 'test', 'plot_encoded'],
                        default='train2',
                        help='Chooses between training auto encoder (train1), training main network (train2), or testing')
    parser.add_argument('--implementation', choices=factory.get_choices(), default=factory.get_default())
    parser.add_argument('--widths', default=[600,400,200], type=int,nargs='+',help='Widths used by default autoencoder')

    training_group = parser.add_argument_group('Parameters for --action train')
    training_group.add_argument('--batchsize', default=128, type=int, help='Number of images per batch')
    training_group.add_argument('--N', default=5, type=int, help='Number of epochs')
    training_group.add_argument('--n', default=5, type=int, help='Number of steps to an epoch')
    training_group.add_argument('--params', default='./params', help='Location for storing plot files')
    training_group.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    training_group.add_argument('--lr_end_factor', type=float, default=0.9,
                                help='Controls Learning Rate. After total_lr_iters epochs, learning rate will be reduced by this factor')
    training_group.add_argument('--total_lr_iters', default=None, type=int, help='Number of steps to fully decay learning rate')
    training_group.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    training_group.add_argument('--momentum', type=float, default=0, help='Momentum for SGD')
    training_group.add_argument('--optimizer', choices=OptimizerFactory.choices, default=OptimizerFactory.get_default(),
                                help='Optimizer to be used for training')
    training_group.add_argument('--restart', default=None, help='Restart from saved parameters')
    training_group.add_argument('--width', default=28, type=int, help='Width of each image in pixels')
    training_group.add_argument('--height', default=28, type=int, help='Height of each image in pixels')
    training_group.add_argument('--bottleneck', default=14, type=int, help='Number of neurons in bottleneck')
    training_group.add_argument('--dataset_fraction',default=None,type=float,
                                help='Proportion of training data that will be used.')
    training_group.add_argument('--validation_fraction',default=0.1,type=float,
                                help='Propertion of data for validation (after dataset_fraction)')

    test_group = parser.add_argument_group('Parameters for --action test')
    test_group.add_argument('--file', default=None, help='Used to load weights')
    test_group.add_argument('--nrows', default=7, type=int, help='Number of rows to display')
    test_group.add_argument('--ncols', default=7, type=int, help='Number of images to display in each row')

    shared_group = parser.add_argument_group('General Parameters')
    shared_group.add_argument('--data', default='./data', help='Location of data files')
    shared_group.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    shared_group.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    shared_group.add_argument('--figs', default='./figs', help='Location for storing plot files')
    shared_group.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
    return parser.parse_args()

def create_dataset(data,fraction=None,rng = np.random.default_rng()):
    '''
    Read the dataset; optionally extract a subset

    Parameters:
        data       Path to stored data
        fraction   Optional parameter for a fraction of data
        rng        Random number generator
    '''
    dataset = MNIST(root=args.data, download=True, transform=tr.ToTensor())
    if args.dataset_fraction == None: return dataset
    return Subset(dataset, rng.choice(len(dataset),replace=False,size=int(fraction*len(dataset))))

def encoder_training_step(model, batch, optimizer):
    '''
    Perform training step. Calculate loss for one batch of data, and its gradient,
    then use optimizer to update weights

    Parameters:
        model      The model that we are training
        batch      Current batch of data
        optimizer  Used to update weights
    '''
    loss = model.get_batch_loss(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def get_validation_loss(model, loader):
    '''
    Used during traning to compute loss from validation data

    Parameters:
        model     The model we are training
        loader    Source for batches of data
    '''
    return [float(model.get_batch_loss(batch).detach()) for batch in loader]


def get_file_name(args):
    '''
    Used to save plots and weights.

    Parameters:
        args     Command line arguments used as part of file name
    '''
    return f'{Path(__file__).stem}-{args.bottleneck}'


def generate_samples(images, n=12):
    '''
    Used to draw samples from a collection of images

    Parameters:
        images     Collection of images
        n          Number of images to retrieve
    '''
    m, _, _, _ = images.shape
    samples = rng.choice(m, n, replace=False)

    for image_index in range(n):
        yield samples[image_index]


def display_images(model, loader, bottleneck, nrows=4, ncols=2, fig=None):
    '''
    Display a grid filled with pairs of images. Each pair comprises
    one randomly selected MNIST image, accompanied by the results of
    processing it through the autoencoder. All images are taken
    from the same randomly selected batch.

    Parameters:
        model      The autoencoder
        loader     Used to load data
        bottleneck Used in suptitle
        nrows      Number of rows to display
        ncols      Number of columns to display
        fig        Figure fo displaying images
    '''
    def display_one_image(image, ax=None):
        '''
        Display one image without axis decorations

        Parameters:
            img        Image for display
            ax         Axis for displaying image
        '''
        ax.imshow(image.squeeze(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    batch_number = rng.choice(len(loader) - 1)
    for k, batch in enumerate(loader):
        if k != batch_number:
            continue
        images, _ = batch
        processed = model(images)
        samples = generate_samples(images, n=nrows * ncols)
        subplot_index = 0
        for i in range(nrows):
            for j in range(ncols):
                sample = next(samples)
                subplot_index += 1
                display_one_image(images[sample], ax=fig.add_subplot(nrows, 2 * ncols, subplot_index))
                img = np.reshape(processed[sample].detach().numpy(), (28, 28))
                subplot_index += 1
                display_one_image(img, ax=fig.add_subplot(nrows, 2 * ncols, subplot_index))
        fig.suptitle(f'Batch {batch_number}, bottleneck={bottleneck}')
        return


def plot_losses(history, lr_history, ax=None, bottleneck=3, window_size=11, optimizer_text=''):
    '''
    Plot history: losses, moving average of losses, and learning rate

    Parameters:
        history          Losses for entire run
        lr_history       Learning rate for entire run
        ax               Axis for plotting
        bottleneck       Number of nodes in bottleneck
        window_size      Number of points in moving average
        optimizer_text   Text identifying optimizer, for use in title
    '''
    xs = np.arange(0, len(history))
    x1s, moving_average = get_moving_average(xs, history, window_size=window_size)
    ax.plot(xs, history, c='xkcd:blue', label='Loss')
    ax.plot(x1s, moving_average, c='xkcd:red', label=f'Average Loss, last={moving_average[-1]:.5e}')
    ax.set_title(f'{Path(__file__).stem.title()}, bottleneck={bottleneck}, {optimizer_text}')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Step')
    ax.set_ylim(bottom=0)
    ax_twin = ax.twinx()
    ax_twin.plot(xs,lr_history,c='xkcd:green',label='Learning Rate')
    ax_twin.set_ylim(bottom=0)
    ax_twin.set_ylabel('Learning Rate')
    legend_handles1, legend_labels1 = ax.get_legend_handles_labels()
    legend_handles2, legend_labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(legend_handles1 + legend_handles2, legend_labels1 + legend_labels2, loc='lower left')

class ManifoldDisplayer:
    def __init__(self,auto_encoder):
        self.auto_encoder = auto_encoder

    def can_display(self):
        return self.auto_encoder.bottleneck in [2,3]

    def display(self, loader, fig):
        '''
        Display points in manifold defined by bottleneck

        Parameters:
            loader         Data source
            fig            Figure in which to place points
        '''
        ax = None
        number_of_classes = 10
        colours = create_xkcd_colours(number_of_classes)
        needs_text_label = [True for _ in range(number_of_classes)]
        for k, batch in enumerate(loader):
            images, labels = batch
            for i in range(len(labels)):
                img = self.auto_encoder.encode(images[i]).detach().numpy()[0]
                text_label = None
                if needs_text_label[labels[i]]:
                    text_label = str(int(labels[i].detach().numpy()))
                    needs_text_label[labels[i]] = False
                match len(img):
                    case 2:
                        if ax == None:
                            ax = fig.add_subplot(1, 1, 1)
                        ax.scatter(img[0], img[1], c=colours[labels[i]], label=text_label, s=1)
                    case 3:
                        if ax == None:
                            ax = fig.add_subplot(1, 1, 1, projection='3d')
                        ax.scatter(img[0], img[1], img[2], c=colours[labels[i]], label=text_label, s=1)
                    case _:
                        return

        sorted_handles, sorted_labels = sort_labels(ax)
        ax.legend(sorted_handles, sorted_labels, title='Labels', loc='upper right', markerscale=3)


def create_scheduler(optimizer, end_factor=0.01,total_iters=None,N=50):
    '''
    Create a learning rate scheduler

    Parameters:
        optimizer     The scheduler whose learning rate is to be controlled
        end_factor    Learning rate will be reduced to this value time original
        total_iters   Learning rate will decrease by this amount per iteration
        N             Total number of iterations for training,
                      used to establish default for total_iters

    Returns:
       A scheduler associated with optimizer
    '''
    return LinearLR(optimizer, start_factor=1.0,end_factor=end_factor,
                    total_iters = total_iters if total_iters != None else N)


if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    fig = figure(figsize=(24, 12))
    start = time()
    auto_encoder_factory = AutoEncoderFactory()
    args = parse_args(auto_encoder_factory)
    with Logger(join(args.logfiles,get_file_name(args))) as logger:
        seed = get_seed(args.seed,notify=lambda s: logger.log(f'Created new seed {s}'))
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        match args.action:
            case 'train1':
                auto_encoder = auto_encoder_factory.create(args)
                logger.log(str(auto_encoder))
                optimizer, optimizer_text = OptimizerFactory.create(auto_encoder, args)
                scheduler = create_scheduler(optimizer, end_factor=args.lr_end_factor,
                                             total_iters=args.total_lr_iters,N=args.N)

                dataset = create_dataset(args.data,fraction=args.dataset_fraction,rng=rng)

                train_data, validation_data = random_split(dataset, [1-args.validation_fraction, args.validation_fraction])
                train_loader = DataLoader(train_data, args.batchsize, shuffle=True)
                validation_loader = DataLoader(validation_data, args.batchsize, shuffle=False)
                history = []
                lr_history = []
                for epoch in range(args.N):
                    for _ in range(args.n):
                        for batch in train_loader:
                            encoder_training_step(auto_encoder, batch, optimizer)

                    lr_previous = optimizer.param_groups[0]["lr"]
                    scheduler.step()

                    validation_losses = get_validation_loss(auto_encoder, validation_loader)
                    history += validation_losses
                    lr_history += len(validation_losses) * [lr_previous]
                    logger.log(f'Epoch {epoch + 1} of {args.N}. Loss = {np.mean(validation_losses):.6f}, lr={lr_previous:.6f}')

                    checkpoint_file_name = join(args.params, get_file_name(args))
                    ensure_we_can_save(checkpoint_file_name)
                    auto_encoder.save(checkpoint_file_name)
                    if user_has_requested_stop():
                        break

                displayer = ManifoldDisplayer(auto_encoder)
                if displayer.can_display():
                    subfigs = fig.subfigures(2, 1, wspace=0.07)
                    plot_losses(history, lr_history, ax=subfigs[0].add_subplot(1, 1, 1),
                                bottleneck=args.bottleneck, optimizer_text=optimizer_text)
                    displayer.display(validation_loader, fig=subfigs[1])
                else:
                    plot_losses(history, lr_history, ax=fig.add_subplot(1, 1, 1),
                                bottleneck=args.bottleneck, optimizer_text=optimizer_text)
                fig.savefig(join(args.figs, get_file_name(args)))

            case 'train2':
                auto_encoder = auto_encoder_factory.create(args)
                auto_encoder.load('./params/aetrain.pth')
                perceptron = Perceptron()
                optimizer = OptimizerFactory.create(auto_encoder, args)
                scheduler = create_scheduler(optimizer, end_factor=args.lr_end_factor,
                                   total_iters=args.total_lr_iters,N=args.N)
                dataset = MNIST(root=args.data, download=True, transform=tr.ToTensor())
                train_data, validation_data = random_split(dataset, [50000, 10000])
                train_loader = DataLoader(train_data, args.batchsize, shuffle=True)
                validation_loader = DataLoader(validation_data, args.batchsize, shuffle=False)
                history = []
                lr_history = []

                for epoch in range(args.N):
                    for _ in range(args.n):
                        for batch in train_loader:
                            perceptron.training_step(batch, auto_encoder, optimizer)

                        lr_previous = optimizer.param_groups[0]["lr"]
                        scheduler.step()

                        validation_losses = get_validation_loss(auto_encoder, validation_loader)
                        history += validation_losses
                        lr_history += len(validation_losses) * [lr_previous]
                        logger.log(f'Epoch {epoch + 1} of {args.N}. Loss = {np.mean(validation_losses):.6f}, lr={lr_previous:.6f}')


                plot_losses(history, lr_history, ax=fig.add_subplot(1, 1, 1))

            case 'test':
                auto_encoder = auto_encoder_factory.create(args)
                auto_encoder.load(args.file)
                dataset = MNIST(root=args.data, download=True, train=False, transform=tr.ToTensor())
                loader = DataLoader(dataset, args.batchsize)
                display_images(auto_encoder, loader, args.bottleneck, nrows=args.nrows, ncols=args.ncols, fig=fig)
                fig.savefig(join(args.figs, f'{Path(args.file).stem}-test'))

            case 'plot_encoded':
                auto_encoder = auto_encoder_factory.create(args)
                auto_encoder.load(args.file)
                dataset = MNIST(root=args.data, download=True, train=False, transform=tr.ToTensor())
                loader = DataLoader(dataset, args.batchsize)
                displayer = ManifoldDisplayer(auto_encoder)
                displayer.display(loader, fig=fig)

        elapsed = time() - start
        minutes = int(elapsed / 60)
        seconds = elapsed - 60 * minutes
        logger.log(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
