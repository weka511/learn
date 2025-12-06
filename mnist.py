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


class MnistModel(nn.Module):
    def __init__(self, width=28, height=28, n_classes=10):
        super().__init__()
        self.input_size = width * height
        self.n_classes = n_classes
        self.linear = nn.Linear(self.input_size, self.n_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, self.input_size)
        return self.linear(xb)

    def training_step(self, batch):
        images, labels = batch
        out = self(images) ## Generate predictions
        loss = F.cross_entropy(out, labels) ## Calculate the loss
        return (loss)

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return ({'val_loss': loss, 'val_acc': acc})

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return ({'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()})

    def epoch_end(self, epoch, result):
        print('Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}'.format(epoch, result['val_loss'], result['val_acc']))

    def get_path(self):
        return 'mnist-logistic.pth'

    def save(self):
        torch.save(self.state_dict(), self.get_path())

    def load(self):
        self.load_state_dict(torch.load(self.get_path()))

    def predict(self, img):
        xb = img.unsqueeze(0)
        yb = self(xb)
        _, preds = torch.max(yb, dim=1)
        return preds[0].item()


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='./data')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--N', default=5, type=int)
    parser.add_argument('--n', default=12, type=int)
    parser.add_argument('--action', choices=['train', 'test'], default='train')
    return parser.parse_args()


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return (torch.tensor(torch.sum(preds == labels).item() / len(preds)))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return (history)


if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    fig = figure(figsize=(12, 12))

    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model = MnistModel()
    match args.action:
        case 'train':
            dataset = MNIST(root=args.data, download=True, transform=tr.ToTensor())
            train_data, validation_data = random_split(dataset, [50000, 10000])
            train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
            val_loader = DataLoader(validation_data, args.batch_size, shuffle=False)
            history = [evaluate(model, val_loader)]
            for i in range(args.N):
                history += fit(5, 0.001, model, train_loader, val_loader)
            accuracies = [result['val_acc'] for result in history]
            losses = [result['val_loss'] for result in history]
            model.save()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(accuracies, '-x', label='Accuracy')
            ax.plot(losses, '-o', label='Loss')
            ax.legend()
            ax.set_xlabel('epoch')
            ax.set_title('Accuracy Vs. No. of epochs')
            fig.suptitle('Training', fontsize=12)
            fig.tight_layout(pad=3, h_pad=4, w_pad=3)
            fig.savefig(join(args.figs, Path(__file__).stem))

        case 'test':
            dataset = MNIST(root=args.data, download=True, train=False, transform=tr.ToTensor())
            model.load()
            selection = rng.integers(len(dataset),size=args.n)
            n_cols = 4
            n_rows = args.n // n_cols
            while n_rows*n_cols < args.n:
                n_rows +=1

            for i in range(args.n):
                img, label = dataset[i]
                ax = fig.add_subplot(n_rows, n_cols, i+1)
                ax.imshow(img[0], cmap='gray')
                ax.set_title(f'Label: {label}, Predicted: {model.predict(img)}')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            fig.suptitle('Testing', fontsize=12)
            fig.tight_layout(pad=3, h_pad=9, w_pad=3)
            # fig.savefig(join(args.figs, Path(__file__).stem))

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
