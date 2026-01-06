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
'''

from argparse import ArgumentParser
from glob import glob
from os.path import splitext, join, basename
from pathlib import Path
from string import ascii_letters
from time import time, localtime
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


class NamesDataset(Dataset):

    def __init__(self, data_dir):
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
                self.data_tensors.append(lineToTensor(name))
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
    test_group.add_argument('--file', default=__file__, help='Used to load weights')

    shared_group = parser.add_argument_group('General Parameters')
    shared_group.add_argument('--data', default='./data/rnn-1/names', help='Location of data files')
    shared_group.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    shared_group.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    shared_group.add_argument('--figs', default='./figs', help='Location for storing plot files')
    shared_group.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
    return parser.parse_args()


allowed_characters = ascii_letters + ' .,;\'' + '_'
n_letters = len(allowed_characters)


def unicodeToAscii(s):
    '''
    Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in allowed_characters
    )


def letterToIndex(letter):
    '''
    Find letter index from all_letters, e.g. 'a' = 0
    return our out-of-vocabulary character if we encounter a letter unknown to our model
    '''
    if letter not in allowed_characters:
        return allowed_characters.find('_')
    else:
        return allowed_characters.find(letter)


def lineToTensor(line):
    '''
    Turn a line into a <line_length x 1 x n_letters>,
    or an array of one-hot letter vectors
    '''
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i


def train(rnn, training_data, n_epoch=10, n_batch_size=64, report_every=1,
          learning_rate=0.2, criterion=nn.NLLLoss()):
    '''
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    '''
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    # start = time.time()
    print(f'training on data set with n = {len(training_data)}')

    for epoch in range(1, n_epoch + 1):
        rnn.zero_grad() # clear the gradients

        # create some minibatches
        # we cannot use dataloaders because each of our names is a different length
        batches = list(range(len(training_data)))
        # random.shuffle(batches)    FIXME
        batches = np.array_split(batches, len(batches) // n_batch_size)

        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch: #for each example in this batch
                (label_tensor, text_tensor, label, text) = training_data[i]
                output = rnn.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            # optimize parameters
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


def evaluate(rnn, testing_data, classes):
    confusion = torch.zeros(len(classes), len(classes))

    rnn.eval() #set to eval mode
    with torch.no_grad(): # do not record the gradients during eval phase
        for i in range(len(testing_data)):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            output = rnn(text_tensor)
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1

    # Normalize by dividing every row by its sum
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

    alldata = NamesDataset(args.data)
    print(f'loaded {len(alldata)} items of data')
    print(f'example = {alldata[0]}')
    train_set, test_set = random_split(alldata, [.85, .15], generator=torch.Generator(device=device).manual_seed(int(seed)))
    print(f'train examples = {len(train_set)}, validation examples = {len(test_set)}')
    n_hidden = 128
    rnn = CharRNN(n_letters, n_hidden, len(alldata.labels_uniq))
    print(rnn)
    all_losses = train(rnn, train_set, n_epoch=args.N, learning_rate=0.15, report_every=5)

    confusion, classes = evaluate(rnn, test_set, classes=alldata.labels_uniq)

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(all_losses)

    # Set up plot

    ax1 = fig.add_subplot(2, 1, 2)
    cax = ax1.matshow(confusion.cpu().numpy()) #numpy uses cpu here so we need to use a cpu version
    fig.colorbar(cax)

    # Set up axes
    ax1.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax1.set_yticks(np.arange(len(classes)), labels=classes)

    # Force label at every tick
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.savefig(join(args.figs, Path(args.file).stem))
    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
