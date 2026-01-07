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
    NLP From Scratch: Generating Names with a Character-Level RNN
    https://docs.pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
'''

from argparse import ArgumentParser
import glob
from os.path import splitext, join, basename
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.optim import SGD, Adam
from utils import Logger, get_seed, user_has_requested_stop
from classify_names import CharacterSet


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,n_categories):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

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


def findFiles(path):
    return glob.glob(path)


def readLines(filename, character_set=CharacterSet()):
    with open(filename, encoding='utf-8') as some_file:
        return [character_set.unicodeToAscii(line.strip()) for line in some_file]

class TrainingDataAdapter:
    def __init__(self,all_categories, category_lines,rng=np.random.default_rng(),character_set = CharacterSet()):
        self.all_categories = all_categories
        self.rng = rng
        self.category_lines = category_lines
        self.character_set = character_set

    def randomTrainingPair(self):
        category = self.rng.choice(self.all_categories)
        line = self.rng.choice(self.category_lines[category])
        return category, line

    def randomTrainingExample(self):
        category, line = self.randomTrainingPair()
        category_tensor = self.categoryTensor(category)
        input_line_tensor = inputTensor(line,self.character_set.n_letters,self.character_set.all_letters)
        target_line_tensor = targetTensor(line,self.character_set.all_letters,self.character_set.n_letters)
        return category_tensor, input_line_tensor, target_line_tensor

    def categoryTensor(self,category):
        '''
        '''
        li = all_categories.index(category)
        tensor = torch.zeros(1, len(self.all_categories))
        tensor[0][li] = 1
        return tensor

# One-hot matrix of first to last letters (not including EOS) for input


def inputTensor(line,n_letters,all_letters):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# ``LongTensor`` of second letter to end (EOS) for target


def targetTensor(line,all_letters,n_letters):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

# Make category, input, and target tensors from a random category, line pair


def read_all_data(data_path,character_set = CharacterSet()):
    category_lines = {}
    all_categories = []
    for filename in findFiles(join(data_path, '*.txt')):
        category = splitext(basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename, character_set=character_set)
        category_lines[category] = lines

    n_categories = len(all_categories)
    assert(n_categories > 0)

    print('# categories:', n_categories, all_categories)
    print(character_set.unicodeToAscii("O'Néàl"))
    return all_categories, category_lines


def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = torch.Tensor([0]) # you can also just simply use ``loss = 0``
    criterion = nn.NLLLoss()

    learning_rate = 0.0005
    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['train', 'test'],
                        default='train', help='Chooses between training or testing')

    parser.add_argument('--batch_size', default=128, type=int, help='Number of images per batch')
    parser.add_argument('--N', default=5, type=int, help='Number of epochs')
    parser.add_argument('--steps', default=5, type=int, help='Number of steps to an epoch')
    parser.add_argument('--params', default='./params', help='Location for storing parameter files')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--optimizer', choices=OptimizerFactory.choices, default=OptimizerFactory.get_default(),
                        help='Optimizer to be used for training')
    parser.add_argument('--restart', default=None, help='Restart from saved parameters')
    parser.add_argument('--file', default=__file__, help='Used to load weights')
    parser.add_argument('--data', default='./data/rnn-1/names', help='Location of data files')
    parser.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
    parser.add_argument('--hidden', default=128, type=int, help='Number of hidden nodes')
    parser.add_argument('--print_every', default=1000, type=int, help='Number of epochs')
    parser.add_argument('--plot_every', default=100, type=int, help='Number of epochs')
    return parser.parse_args()


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
    character_set = CharacterSet()

    all_losses = []
    total_loss = 0
    all_categories, category_lines = read_all_data(args.data,character_set=character_set)
    data = TrainingDataAdapter(all_categories, category_lines,rng=rng,character_set = character_set)
    rnn = RNN(character_set.n_letters, args.hidden, character_set.n_letters,len(data.all_categories))
    for iter in range(1, args.N + 1):
        output, loss = train(*data.randomTrainingExample())
        total_loss += loss

        if iter % args.print_every == 0:
            print (f'Iteration {iter}, {(iter / args.N) * 100}% , loss={loss}')

        if iter % args.plot_every == 0:
            all_losses.append(total_loss / args.plot_every)
            total_loss = 0

    rnn.save(join(args.params, Path(args.file).stem))

    ax = fig.add_subplot(1,1,1)
    ax.plot(all_losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    fig.savefig(join(args.figs, Path(args.file).stem))

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
