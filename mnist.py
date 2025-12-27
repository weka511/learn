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
from utils import Logger, get_seed, user_has_requested_stop

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
        acc = get_accuracy(out, labels)
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


class CNN(MnistModel):
    '''
    Convolution Neural Network
    Snarfed from https://www.kaggle.com/code/sdelecourt/cnn-with-pytorch-for-mnist
    '''
    name = 'CNN'

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelFactory:
    '''
    This class instantiates models as specified by the command line parameters
    '''
    choices = [
        LinearRegressionModel.name,
        PerceptronModel.name,
        CNN.name
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
            case CNN.name:
                return CNN()

    @staticmethod
    def create_from_file_name(name):
        for choice in ModelFactory.choices:
            if name.find(choice) > -1:
                return ModelFactory.create(choice)

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
    training_group.add_argument('--model', choices=ModelFactory.choices, default=ModelFactory.choices[0],
                                help='Type of model to train')
    training_group.add_argument('--params', default='./params', help='Location for storing plot files')
    training_group.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    training_group.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    training_group.add_argument('--optimizer', choices=OptimizerFactory.choices, default=OptimizerFactory.get_default(),
                                help='Optimizer to be used for training')
    training_group.add_argument('--restart', default=None, help='Restart from saved parameters')

    test_group = parser.add_argument_group('Parameters for --action test')
    test_group.add_argument('--file', default=None, help='Used to load weights')
    test_group.add_argument('--rows', default=6, type=int, help='Number of rows of images to display')
    test_group.add_argument('--cols', default=12, type=int, help='Number of columns of images to display')

    shared_group = parser.add_argument_group('General Parameters')
    shared_group.add_argument('--data', default='./data', help='Location of data files')
    shared_group.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    shared_group.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    shared_group.add_argument('--figs', default='./figs', help='Location for storing plot files')
    shared_group.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')

    return parser.parse_args()


class NameFactory:
    def __init__(self, args, seed):
        self.seed_text = '' if seed == None else f'-{seed}'
        self.action = args.action
        self.model = args.model
        self.N = args.N
        self.batch_size = args.batch_size
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.file = args.file

    def create_short_name(self, checkpoint=False):
        '''
        Used as the stem for file names
        '''
        action = self.action if checkpoint == False else 'checkpoint'
        name = f'{self.model}-{action}-{self.N}-{self.batch_size}-{self.seed_text}-{self.optimizer}-{self.lr}-{self.weight_decay}'
        return name.replace('.', '_')

    def create_long_name(self):
        '''
        Used for titles
        '''
        title = (f'Model={self.model}: {self.action}, N={self.N}, batch_size={self.batch_size}, {self.seed_text}, '
                 f'optimizer={self.optimizer}, lr={self.lr}, weight decay={self.weight_decay}')
        return title if self.file == None else title + self.file


def get_accuracy(prediction, labels):
    '''
    Calculate accuracy, i.e. how frequently prediction matches labels

    Parameters:
        prediction    Output from model
        labels        Expected output
    '''
    _, predicted_labels = torch.max(prediction, dim=1)
    return (torch.tensor(torch.sum(predicted_labels == labels).item() / len(predicted_labels)))


def evaluate(model, loader):
    '''
    Used to evaluate goodness of fit

    Parameters:
        model    The model we are using to predict labels
        loader    Used to read data
    '''
    prediction = [model.validation_step(batch) for batch in loader]
    return model.get_loss_and_accuracy(prediction)


def fit(epoch, n_steps, model, train_loader, val_loader, optimizer=None, logger=None):
    '''
    Fit parameters to training data

    Parameters:
        epoch
        n_steps
        model
        train_loader
        val_loader
        optimizer
        logger
    '''
    history = []

    for i in range(n_steps):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        logger.log(f'Step: {n_steps * epoch + i}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}')
        history.append(result)
    return (history)

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


def generate_mismatches(dataset, n, rng=np.random.default_rng()):
    '''
    Used when testing: iterate through images where
    prediction and label don't match

    Parameters:
        dataset
        n
        rng
    '''
    indices = rng.permutation(len(dataset))
    j = 0
    for i in range(n):
        prediction = None
        label = None
        while prediction == label:
            k = indices[j]
            img, label = dataset[k]
            prediction = model.predict(img)
            j += 1

        yield i + 1, img, label, prediction


def create_model(restart, model_name):
    '''
    Allows model to in initialzed from scratch or loaded from saved weights

    Parameters:
        restart
        model_name
    '''
    if restart:
        model = ModelFactory.create_from_file_name(restart)
        model.load(restart)
        print(f'Reloaded parameters from {restart}')
        return model
    else:
        return ModelFactory.create(model_name)


if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    fig = figure(figsize=(24, 12))
    start = time()
    args = parse_args()
    seed = get_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    name_factory = NameFactory(args, seed)
    match args.action:
        case 'train':
            with Logger(join(args.logfiles, name_factory.create_short_name())) as logger:
                model = create_model(args.restart, args.model)
                optimizer = OptimizerFactory.create(model, args)
                dataset = MNIST(root=args.data, download=True, transform=tr.ToTensor())
                train_data, validation_data = random_split(dataset, [50000, 10000])
                train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
                val_loader = DataLoader(validation_data, args.batch_size, shuffle=False)
                history = [evaluate(model, val_loader)]
                for i in range(args.N):
                    history += fit(i, args.steps, model, train_loader, val_loader,
                                   optimizer=optimizer, logger=logger)
                    checkpoint_file_name = join(args.params, name_factory.create_short_name(checkpoint=True))
                    ensure_we_can_save(checkpoint_file_name)
                    model.save(checkpoint_file_name)
                    if user_has_requested_stop():
                        break
                accuracies = [result['val_acc'] for result in history]
                losses = [result['val_loss'] for result in history]
                model.save(join(args.params, name_factory.create_short_name()))

                ax = fig.add_subplot(1, 1, 1)
                ax.plot(accuracies, '-x', label='Accuracy')
                ax.plot(losses, '-o', label='Loss')
                ax.legend()
                ax.set_xlabel('epoch')
                ax.set_title('Accuracy Vs. No. of epochs')
                fig.suptitle(name_factory.create_long_name(), fontsize=12)
                fig.tight_layout(pad=3, h_pad=4, w_pad=3)
                fig.savefig(join(args.figs, name_factory.create_short_name()))

        case 'test':
            model = ModelFactory.create_from_file_name(args.file)
            model.load(args.file)
            dataset = MNIST(root=args.data, download=True, train=False, transform=tr.ToTensor())
            test_loader = DataLoader(dataset, batch_size=256)
            score = evaluate(model, test_loader)

            for pos, img, label, prediction in generate_mismatches(dataset, args.rows * args.cols, rng=rng):
                ax = fig.add_subplot(args.rows, args.cols, pos)
                ax.imshow(img[0], cmap='gray')
                ax.set_title(f'{label}[{prediction}]')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            fig.suptitle(rf'Testing {args.file}: Loss = {score['val_loss']:.4}, Accuracy = {100 * score['val_acc']:.2f}\%',
                         fontsize=12)
            fig.tight_layout(pad=3, h_pad=9, w_pad=3)
            fig.savefig(join(args.figs, Path(args.file).stem.replace('train', 'test')))


    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
