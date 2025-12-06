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
# from torch.optim import Adam
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
        print(xb)
        out = self.linear(xb)
        print(out)
        return out


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='./data')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--action', choices=['train', 'test'], default='train')
    return parser.parse_args()

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return(torch.tensor(torch.sum(preds == labels).item()/ len(preds)))

if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    match args.action:
        case 'train':
            dataset = MNIST(root=args.data, download=True, transform=tr.ToTensor())
            print(dataset)
            image_tensor, label = dataset[0]
            print(image_tensor.shape, label)
            train_data, validation_data = random_split(dataset, [50000, 10000])
            print(f'length of Train Datasets: {len(train_data)}')
            print(f'length of Validation Datasets: {len(validation_data)}')
            batch_size = 128
            train_loader = DataLoader(train_data, batch_size, shuffle=True)
            val_loader = DataLoader(validation_data, batch_size, shuffle=False)
            model = MnistModel()
            print(model.linear.weight.shape, model.linear.bias.shape)
            list(model.parameters())
            for images, labels in train_loader:
                outputs = model(images)
                break
            print('outputs shape: ', outputs.shape)
            print('Sample outputs: \n', outputs[:2].data)
            probs = F.softmax(outputs, dim = 1)
            print("Sample probabilities:\n", probs[:2].data)
            print("Sum: ", torch.sum(probs[0]).item())
            max_probs, preds = torch.max(probs, dim = 1)
            print("\n")
            print(preds)
            print("\n")
            print(max_probs)
            print("Accuracy: ",accuracy(outputs, labels))
            print("\n")
            loss_fn = F.cross_entropy
            print("Loss Function: ",loss_fn)
            print("\n")
            ## Loss for the current batch
            loss = loss_fn(outputs, labels)
            print (loss)

        case 'test':
            dataset = MNIST(root=args.data, download=True, train=False, transform=tr.ToTensor())
            print(len(dataset))

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
