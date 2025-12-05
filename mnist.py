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

'''Learn to recognize images from the NIST dataset'''

# https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/

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
from torch.optim import Adam

class MnistDataloader(object):
    '''
    This class loads the MNIST dataset

    Snarfed from https://www.kaggle.com/code/talhaahmed121/mnist-simple-cnn
    '''
    def __init__(self, training_images,training_labels,
                 test_images, test_labels):
        self.training_images = training_images
        self.training_labels = training_labels
        self.test_images = test_images
        self.test_labels = test_labels

    def read_images_labels(self, images, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = unpack('>II', file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array('B', file.read())

        with open(images, 'rb') as file:
            magic, size, rows, cols = unpack('>IIII', file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array('B', file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images, self.training_labels)
        x_test, y_test = self.read_images_labels(self.test_images, self.test_labels)
        return (x_train, y_train),(x_test, y_test)

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='./data/mnist')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--seed', default=None,type=int)
    parser.add_argument('--action', choices=['display','train'], default='train')
    return parser.parse_args()

def show_images(images, title_texts,figs='./figs'):
    cols = 5
    rows = int(len(images)/cols) + 1
    fig = figure(figsize=(30,20))

    for i,x in enumerate(zip(images, title_texts)):
        image = x[0]
        title_text = x[1]
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(image, cmap=cm.gray)
        if (title_text != ''):
            ax.set_title(title_text, fontsize = 15);
    fig.suptitle(f'From {Path(__file__).stem}')
    fig.tight_layout(pad=5, h_pad=4, w_pad=3)
    fig.savefig(join(figs, 'mnist-images'))

if __name__=='__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    start  = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    training_images = join(args.data, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels = join(args.data, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images = join(args.data, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels = join(args.data, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    mnist_dataloader = MnistDataloader(training_images, training_labels,test_images, test_labels)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    match args.action:
        case 'display':
            images_2_show = []
            titles_2_show = []
            for i in range(0, 10):
                r = rng.integers(1, 60000)
                images_2_show.append(x_train[r])
                titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

            for i in range(0, 5):
                r = rng.integers(1, 10000)
                images_2_show.append(x_test[r])
                titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

            show_images(images_2_show, titles_2_show,figs=args.figs)

        case 'train':
            print ('TBP')

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
