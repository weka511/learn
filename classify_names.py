#!/usr/bin/env python

#   Copyright (C) 2026 Simon Crase

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
NLP From Scratch: Classifying Names with a Character-Level RNN
Sean Robertson
https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

We will train on a few thousand surnames from 18 languages of origin,
and predict which language a name is from based on the spelling.
'''

from argparse import ArgumentParser
from glob import glob
from os.path import splitext, join, basename
from pathlib import Path
from string import ascii_letters
from time import time, localtime
import unicodedata
from matplotlib.pyplot import figure, show
from matplotlib import rc
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.optim import SGD, Adam
from utils import get_seed, get_device


class CharacterSet:
    '''
    This class contains knowledge of the chracter set. Ir constructs one-hot vectors representing characters
    '''
    def __init__(self):
        self.allowed_characters = ascii_letters + ' .,;\'' + '_'
        self.all_letters = ascii_letters + " .,;'-"
        self.n_letters = len(self.all_letters) + 1 # Plus EOS marker

    def __len__(self):
        return len(self.allowed_characters)

    def unicodeToAscii(self, s):
        '''
        Turn a Unicode string to plain ASCII

        Thanks to https://stackoverflow.com/a/518232/2809427
        '''
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn' and c in self.allowed_characters
        )

    def letterToIndex(self, letter):
        '''
        Find  index from a letters, e.g. 'a' = 0

        Parameters:
            letter     Letter whose index is desired

        Returns:
            Index, or our out-of-vocabulary character if we encounter a letter unknown to our model

        '''
        index = self.allowed_characters.find(letter)
        return index if index > -1 else self.allowed_characters.find('_')

    def createInputTensor(self,line):
        '''
        Factory method to construct one-hot matrix of first to last letters (not including EOS) for input

        Parameters:
            line    A line of text to be used to make matrix of one-hot vectors
        '''
        product = torch.zeros(len(line), 1, self.n_letters)
        for i in range(len(line)):
            letter = line[i]
            product[i][0][self.all_letters.find(letter)] = 1
        return product

    def createTargetTensor(self,line):
        '''
        Factory Method to create LongTensor of second letter to end (EOS) for target

        Parameters:
            line    A line of text to be used to make matrix of one-hot vectors
        '''
        letter_indexes = [self.all_letters.find(line[i]) for i in range(1, len(line))]
        letter_indexes.append(self.n_letters - 1) # EOS
        return torch.LongTensor(letter_indexes)

    def createLineTensor(self,line):
        '''
        Turn a line into a <line_length x 1 x n_letters>,
        or an array of one-hot letter vectors

         Parameters:
            line    A line of text to be used to make tensoe
        '''
        product = torch.zeros(len(line), 1, len(self))
        for i, letter in enumerate(line):
            product[i][0][self.letterToIndex(letter)] = 1
        return product

class NamesDataset(Dataset):
    '''
    This class holds the traring and test data for out network.
    It contains names from the dataset

    Data members:
        data_dir        Folder where data was loaded from
        load_time       Identifies when data was loaded
        data            List of names in dataset
        data_tensors    List of names in dataset converted to tensors, one for each language
        labels          List of labels (languages) one for each word (so there are duplicates)
        labels_tensors  List of labels  converted to tensors
    '''
    def __init__(self, data_dir, character_set=CharacterSet()):
        self.data_dir = data_dir
        self.load_time = localtime
        labels_set = set()
        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []
        text_files = glob(join(data_dir, '*.txt'))

        for filename in text_files:
            label = splitext(basename(filename))[0]
            labels_set.add(label)
            for name in open(filename, encoding='utf-8').read().strip().split('\n'):
                self.data.append(name)
                self.data_tensors.append(character_set.createLineTensor(name))
                self.labels.append(label)

        #Cache the tensor representation of the labels
        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        '''
        Get number of languges
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Retrieve data for one language

        Parameters:
            idx      Index indentifying language

        Returns:
            label_tensor
            data_tensor
            data_label     The language corresponding to the name
            data_item      Name corresponding to index
        '''
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]
        return label_tensor, data_tensor, data_label, data_item


class CharRNN(nn.Module):
    '''
    Recurrent Neural Network

    This CharRNN class implements an RNN with three components.
    First, we use the nn.RNN implementation. Next, we define a layer
    that maps the RNN hidden layers to our output.
    And finally, we apply a softmax function.
    Using nn.RNN leads to a significant improvement in performance,
    such as cuDNN-accelerated kernels, versus implementing each layer as a nn.Linear.
    It also simplifies the implementation in forward()
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)
        return output


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

    parser.add_argument('--batch_size', default=64, type=int, help='Number of images per batch')
    parser.add_argument('--N', default=5, type=int, help='Number of epochs')
    parser.add_argument('--hidden', default=128, type=int, help='Number of hidden nodes')
    parser.add_argument('--params', default='./params', help='Location for storing plot files')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--optimizer', choices=OptimizerFactory.choices, default=OptimizerFactory.get_default(),
                                help='Optimizer to be used for training')
    parser.add_argument('--data', default='./data/rnn-1/names', help='Location of data files')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
    parser.add_argument('--file', default=__file__, help='Used to save figure')
    return parser.parse_args()

def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i


def train(rnn, data_set, N=10, batch_size=64, report_every=5,
          criterion=nn.NLLLoss(), rng=np.random.default_rng(), optimizer=None):
    '''
    Learn on a batch of training data for a specified number of iterations and reporting thresholds

    Parameters:
        rnn              The network we are training
        data_set         Training data
        N                Number of epochs of training
        batch_size       Number of records per batch
        report_every     Controls frequency of reporting
        criterion        For judging match
        rng              Random number geerator
        optimizer        Used to improve fitness
    '''
    def create_minibatches():
        '''
        Organize data into minibatches
        we cannot use dataloaders because each of our names is a different length

        Returns:
           A permutation of the indices of the data items, origanized into an array of arrays.
           Each of the low level arrays contains the indices of data items comprising one batch
        '''
        permutation = list(range(len(data_set)))
        rng.shuffle(permutation)
        return np.array_split(permutation, len(permutation) // batch_size)

    current_loss = 0
    all_losses = []
    rnn.train()

    print(f'training on data set with n = {len(data_set)}')

    for epoch in range(1, N + 1):
        rnn.zero_grad()
        batches = create_minibatches()
        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch:
                (label_tensor, text_tensor, label, text) = data_set[i]
                output = rnn.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)

        all_losses.append(current_loss / len(batches))
        if epoch % report_every == 0:
            print(f'{epoch} ({epoch / N:.0%}): \t average batch loss = {all_losses[-1]}')
        current_loss = 0

    return all_losses


def evaluate(rnn, data_set, classes):
    '''
    Evaluate model against test data and compute confusion matrix

    Parameters:
        rnn         Model
        data_set    The data
        classes     List of classification labels
    '''
    def normalize(matrix):
        '''
        Normalize  matrix so each row sums to 1 (unless is is all zeros)
        '''
        for i in range(len(classes)):
            denom = matrix[i].sum()
            matrix[i] /= (denom if denom > 0 else 1)

        return matrix

    confusion = torch.zeros(len(classes), len(classes))

    rnn.eval()
    with torch.no_grad():
        for i in range(len(data_set)):
            (label_tensor, text_tensor, label, text) = data_set[i]
            output = rnn(text_tensor)
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1

    return normalize(confusion), classes


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
    device = get_device()
    torch_generator = torch.Generator(device=device).manual_seed(seed)
    character_set = CharacterSet()
    alldata = NamesDataset(args.data, character_set=character_set)
    print(f'loaded {len(alldata)} items of data')

    train_set, test_set = random_split(alldata, [.85, .15], generator=torch_generator)
    print(f'train examples = {len(train_set)}, validation examples = {len(test_set)}')

    rnn = CharRNN(len(character_set), args.hidden, len(alldata.labels_uniq))
    print(rnn)
    optimizer = OptimizerFactory.create(rnn, args)
    all_losses = train(rnn, train_set, N=args.N, optimizer=optimizer, rng=rng, batch_size=args.batch_size)

    confusion, classes = evaluate(rnn, test_set, classes=alldata.labels_uniq)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(list(range(1,len(all_losses)+1)),all_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{args.optimizer}: N={args.N}')

    ax2 = fig.add_subplot(2, 1, 2)
    cax = ax2.matshow(confusion.cpu().numpy())
    fig.colorbar(cax)
    ax2.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax2.set_yticks(np.arange(len(classes)), labels=classes)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.tight_layout(pad=3, h_pad=9, w_pad=3)
    fig.savefig(join(args.figs, Path(args.file).stem))

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
