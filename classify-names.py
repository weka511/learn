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
from utils import Logger, get_seed, user_has_requested_stop


class CharacterSet:
    def __init__(self):
        self.allowed_characters = ascii_letters + ' .,;\'' + '_'

    def __len__(self):
        return len(self.allowed_characters)

    def unicodeToAscii(self, s):
        '''
        Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
        '''
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn' and c in self.allowed_characters
        )

    def letterToIndex(self, letter):
        '''
        Find letter index from all_letters, e.g. 'a' = 0
        return our out-of-vocabulary character if we encounter a letter unknown to our model
        '''
        index = self.allowed_characters.find(letter)
        return index if index > -1 else self.allowed_characters.find('_')


class NamesDataset(Dataset):
    '''
    This class contains names from the dataset
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
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            for name in lines:
                self.data.append(name)
                self.data_tensors.append(lineToTensor(name, character_set=character_set))
                self.labels.append(label)

        #Cache the tensor representation of the labels
        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]
        return label_tensor, data_tensor, data_label, data_item


class CharRNN(nn.Module):
    '''
    Recurrent Neural Network
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
    parser.add_argument('--action', choices=['train', 'test'],
                        default='train', help='Chooses between training or testing')

    training_group = parser.add_argument_group('Parameters for --action train')
    training_group.add_argument('--batch_size', default=64, type=int, help='Number of images per batch')
    training_group.add_argument('--N', default=5, type=int, help='Number of epochs')
    training_group.add_argument('--n_hidden', default=128, type=int, help='Number of hidden nodes')
    training_group.add_argument('--params', default='./params', help='Location for storing plot files')
    training_group.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    training_group.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    training_group.add_argument('--optimizer', choices=OptimizerFactory.choices, default=OptimizerFactory.get_default(),
                                help='Optimizer to be used for training')
    training_group.add_argument('--restart', default=None, help='Restart from saved parameters')

    test_group = parser.add_argument_group('Parameters for --action test')
    test_group.add_argument('--file', default=__file__, help='Used to load weights')

    shared_group = parser.add_argument_group('General Parameters')
    shared_group.add_argument('--data', default='./data/rnn-1/names', help='Location of data files')
    shared_group.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    shared_group.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    shared_group.add_argument('--figs', default='./figs', help='Location for storing plot files')
    shared_group.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
    return parser.parse_args()


def lineToTensor(line, character_set=CharacterSet()):
    '''
    Turn a line into a <line_length x 1 x n_letters>,
    or an array of one-hot letter vectors
    '''
    tensor = torch.zeros(len(line), 1, len(character_set))
    for li, letter in enumerate(line):
        tensor[li][0][character_set.letterToIndex(letter)] = 1
    return tensor


def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i


def train(rnn, data, n_epoch=10, n_batch_size=64, report_every=1,
          criterion=nn.NLLLoss(), rng=np.random.default_rng(), optimizer=None):
    '''
    Learn on a batch of training data for a specified number of iterations and reporting thresholds

    Parameters:
        rnn
        data
        n_epoch
        n_batch_size
        report_every
        criterion=nn.NLLLoss()
        rng
        optimizer
    '''
    def create_minibatches():
        '''
        create some minibatches
        we cannot use dataloaders because each of our names is a different length

        Returns:
           A permutation of the indices of the data items, origanized into an array of arrays.
           Each of the low level arrays contains the indices of data items comprising one batch
        '''
        permutation = list(range(len(data)))
        rng.shuffle(permutation)
        return np.array_split(permutation, len(permutation) // n_batch_size)

    current_loss = 0
    all_losses = []
    rnn.train()

    print(f'training on data set with n = {len(data)}')

    for epoch in range(1, n_epoch + 1):
        rnn.zero_grad()
        batches = create_minibatches()
        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch:
                (label_tensor, text_tensor, label, text) = data[i]
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
            print(f'{epoch} ({epoch / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}')
        current_loss = 0

    return all_losses


def evaluate(rnn, data, classes):
    '''
    Evaluate model against test data and compute confusion matrix

    Parameters:
        rnn
        data
        classes
    '''
    confusion = torch.zeros(len(classes), len(classes))

    rnn.eval()
    with torch.no_grad():
        for i in range(len(data)):
            (label_tensor, text_tensor, label, text) = data[i]
            output = rnn(text_tensor)
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1

    for i in range(len(classes)):
        denom = confusion[i].sum()
        if denom > 0:
            confusion[i] = confusion[i] / denom

    return confusion, classes


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    print(f'Using device = {torch.get_default_device()}')
    character_set = CharacterSet()
    alldata = NamesDataset(args.data, character_set=character_set)
    print(f'loaded {len(alldata)} items of data')

    train_set, test_set = random_split(alldata, [.85, .15], generator=torch.Generator(device=device).manual_seed(int(seed)))
    print(f'train examples = {len(train_set)}, validation examples = {len(test_set)}')
    rnn = CharRNN(len(character_set), args.n_hidden, len(alldata.labels_uniq))
    print(rnn)
    optimizer = OptimizerFactory.create(rnn, args)
    all_losses = train(rnn, train_set, n_epoch=args.N, optimizer=optimizer, report_every=5, rng=rng, n_batch_size=args.batch_size)

    confusion, classes = evaluate(rnn, test_set, classes=alldata.labels_uniq)

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(all_losses)

    ax1 = fig.add_subplot(2, 1, 2)
    cax = ax1.matshow(confusion.cpu().numpy()) #numpy uses cpu here so we need to use a cpu version
    fig.colorbar(cax)

    ax1.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax1.set_yticks(np.arange(len(classes)), labels=classes)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.savefig(join(args.figs, Path(args.file).stem))
    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
