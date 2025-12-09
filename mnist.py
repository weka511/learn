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
    Testbed for deep neural networks, using ideas from Goodfellow et al.
    Learn to recognize images from the NIST dataset

    Snarfed from https://www.kaggle.com/code/geekysaint/solving-mnist-using-pytorch
'''

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from array import array
from os.path import join
from pathlib import Path
from struct import unpack
from time import time
import numpy as np
from matplotlib.pyplot import figure, show, cm
from matplotlib import rc
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as tr
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.optim import SGD, Adam


class MnistModel(nn.Module, ABC):
    '''
    This class implements a neural network model
    '''

    def __init__(self, width=28, height=28, n_classes=10):
        super().__init__()
        self.input_size = width * height
        self.n_classes = n_classes

    @abstractmethod
    def forward(self, xb):
        pass

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        return F.cross_entropy(out, labels)

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return ({'val_loss': loss, 'val_acc': acc})

    def get_loss_and_accuracy(self, outputs):
        '''
        Used to evaluate goodness of fit
        '''
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return ({'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()})

    def log_loss_and_accuracy(self, step, result, logger=None):
        '''
        Used while training to record loss and accuracy
        '''
        logger.log(f'Step: {step}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}')

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

    def predict(self, img):
        '''
        Used while calculating performance with test dataset
        '''
        xb = img.unsqueeze(0)
        yb = self(xb)
        _, preds = torch.max(yb, dim=1)
        return preds[0].item()


class LinearRegressionModel(MnistModel):
    '''
    Perform a simple linear regression
    '''
    name = 'linear'

    def __init__(self, width=28, height=28, n_classes=10):
        super().__init__(width=width, height=height, n_classes=n_classes)
        self.model = nn.Linear(self.input_size, n_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, self.input_size)
        return self.model(xb)


class PerceptronModel(MnistModel):
    '''
    A simple multi layer perceptron
    '''
    name = 'perceptron'

    def __init__(self, width=28, height=28, n_classes=10):
        super().__init__(width=width, height=height, n_classes=n_classes)
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 90),
            nn.ReLU(),
            nn.Linear(90, 10),
            nn.ReLU())

    def forward(self, xb):
        xb = xb.reshape(-1, self.input_size)
        return self.model(xb)


class ModelFactory:
    '''
    This class instantiates models as specified by the command line parameters
    '''
    choices = [
        LinearRegressionModel.name,
        PerceptronModel.name
    ]

    @staticmethod
    def create(name):
        '''
        Create model

        Parameters:
            name
        '''
        match name:
            case LinearRegressionModel.name:
                return LinearRegressionModel()
            case PerceptronModel.name:
                return PerceptronModel()


class OptimizerFactory:
    '''
    This class instantiates optimizers as specified by the command line parameters
    '''

    choices = [
        'SGD',
        'Adam'
    ]

    @staticmethod
    def create(model, args):
        match args.optimizer:
            case 'SGD':
                return SGD(model.parameters(), lr=args.lr)
            case 'Adam':
                return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


class Logger(object):
    '''
    This class records text in a logfile
    '''

    def __init__(self, name):
        self.name = name + '.log'
        self.file = None

    def __enter__(self):
        self.file = open(self.name, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file != None:
            self.file.close()

    def log(self, line):
        '''
        Output one line of text to console and file, flushing as we go
        '''
        print(line, flush=True)
        self.file.write(line + '\n')
        self.file.flush()


def parse_args(factory):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['train', 'test'], default='train', help='Chooses between training or testint')

    training_group = parser.add_argument_group('Parameters for --action train')
    training_group.add_argument('--batch_size', default=128, type=int, help='Number of images per batch')
    training_group.add_argument('--N', default=5, type=int, help='Number of epochs')
    training_group.add_argument('--steps', default=5, type=int, help='Number of steps to an epoch')
    training_group.add_argument('--model', choices=ModelFactory.choices, default=ModelFactory.choices[0],
                                help='Type of model to train')
    training_group.add_argument('--params', default='./params', help='Location for storing plot files')
    training_group.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    training_group.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    training_group.add_argument('--optimizer', choices=OptimizerFactory.choices, default=OptimizerFactory.choices[0],
                                help='Optimizer to be used for training')

    test_group = parser.add_argument_group('Parameters for --action test')
    test_group.add_argument('--file', default=None)
    test_group.add_argument('--n', default=12, type=int, help='Number of images for test')

    shared_group = parser.add_argument_group('General Parameters')
    shared_group.add_argument('--data', default='./data', help='Location of data files')
    shared_group.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    shared_group.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    shared_group.add_argument('--figs', default='./figs', help='Location for storing plot files')
    shared_group.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')

    return parser.parse_args()


def create_short_name(args, checkpoint=False):
    '''
    Used as the stem for file names
    '''
    seed = '' if args.seed == None else f'-{args.seed}'
    action = args.action if checkpoint == False else 'checkpoint'
    name = f'{args.model}-{action}-{args.N}-{args.batch_size}{seed}-{args.optimizer}-{args.lr}-{args.weight_decay}'
    return name.replace('.', '_')


def create_long_name(args):
    '''
    Used for titles
    '''
    seed = '' if args.seed == None else f', seed={args.seed}'
    title = (f'Model={args.model}: {args.action}, N={args.N}, batch_size={args.batch_size}{seed}, '
             f'optimizer={args.optimizer}, lr={args.lr}, weight decay={args.weight_decay}')
    return title if args.file == None else title + name


def accuracy(outputs, labels):
    '''
    Calculate accuracy, i.e. how frequently prediction matches labels
    '''
    _, preds = torch.max(outputs, dim=1)
    return (torch.tensor(torch.sum(preds == labels).item() / len(preds)))


def evaluate(model, loader):
    '''
    Used to evaluate goodness of fit
    '''
    outputs = [model.validation_step(batch) for batch in loader]
    return model.get_loss_and_accuracy(outputs)


def fit(epoch, n_steps, model, train_loader, val_loader, optimizer=None, logger=None):
    '''
    Fit parameters to training data
    '''
    history = []

    for i in range(n_steps):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        model.log_loss_and_accuracy(n_steps * epoch + i, result, logger=logger)
        history.append(result)
    return (history)


def killed(killfile='kill.txt'):
    '''
    Used to verify that there is a killfile, so the program can shut down gracefully
    '''
    killfile_path = Path(killfile)
    killed = killfile_path.is_file()
    if killed:
        print(f'{killfile} detected')
        killfile_path.unlink()
    return killed


def ensure_we_can_save(checkpoint_file_name):
    '''
    If there is already a checkpoint file, we need to make it
    into a backup. But if there is already a backup, delete it first
    '''
    checkpoint_path = Path(checkpoint_file_name + '.pth')
    if not checkpoint_path.is_file():
        return
    checkpoint_path_bak = Path(checkpoint_file_name + '.bak')
    if checkpoint_path_bak.is_file():
        checkpoint_path_bak.unlink()
    checkpoint_path.rename(checkpoint_path_bak)


if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    factory = ModelFactory()
    fig = figure(figsize=(12, 12))

    start = time()
    args = parse_args(factory)
    rng = np.random.default_rng(args.seed)
    model = ModelFactory.create(args.model)
    match args.action:
        case 'train':
            with Logger(join(args.logfiles, create_short_name(args))) as logger:
                optimizer = OptimizerFactory.create(model, args)
                dataset = MNIST(root=args.data, download=True, transform=tr.ToTensor())
                train_data, validation_data = random_split(dataset, [50000, 10000])
                train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
                val_loader = DataLoader(validation_data, args.batch_size, shuffle=False)
                history = [evaluate(model, val_loader)]
                for i in range(args.N):
                    history += fit(i, args.steps, model, train_loader, val_loader,
                                   optimizer=optimizer, logger=logger)
                    checkpoint_file_name = join(args.params, create_short_name(args, checkpoint=True))
                    ensure_we_can_save(checkpoint_file_name)
                    model.save( checkpoint_file_name)
                    if killed():
                        break
                accuracies = [result['val_acc'] for result in history]
                losses = [result['val_loss'] for result in history]
                model.save(join(args.params, create_short_name(args)))

                ax = fig.add_subplot(1, 1, 1)
                ax.plot(accuracies, '-x', label='Accuracy')
                ax.plot(losses, '-o', label='Loss')
                ax.legend()
                ax.set_xlabel('epoch')
                ax.set_title('Accuracy Vs. No. of epochs')
                fig.suptitle(create_long_name(args), fontsize=12)
                fig.tight_layout(pad=3, h_pad=4, w_pad=3)
                fig.savefig(join(args.figs, create_short_name(args)))

        case 'test':
            dataset = MNIST(root=args.data, download=True, train=False, transform=tr.ToTensor())
            model.load(args.file)
            selection = rng.integers(len(dataset), size=args.n)
            n_cols = 4
            n_rows = args.n // n_cols
            while n_rows * n_cols < args.n:
                n_rows += 1

            for i in range(args.n):
                img, label = dataset[i]
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                ax.imshow(img[0], cmap='gray')
                ax.set_title(f'Label: {label}, Predicted: {model.predict(img)}')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            fig.suptitle(create_long_name(args), fontsize=12)
            fig.tight_layout(pad=3, h_pad=9, w_pad=3)
            fig.savefig(join(args.figs, Path(args.file).stem.replace('train', 'test')))

            test_loader = DataLoader(dataset, batch_size=256)
            print(evaluate(model, test_loader))

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
