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
    NLP From Scratch: Translation with a Sequence to Sequence Network and Attention

    https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''

from __future__ import unicode_literals, print_function, division
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
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.optim import SGD, Adam
from utils import Logger, get_seed, user_has_requested_stop
from classify_names import CharacterSet


class Lang:
    '''
        We’ll need a unique index per word to use as the inputs and targets of the networks later.
        To keep track of all this we will use a helper class called Lang which has word -> index (word2index)
        and index-> word (index2word) dictionaries, as well as a count of each word word2count
        which will be used to replace rare words later.
    '''
    SOS_token = 0
    EOS_token = 1

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


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


class DataSet:

    def __init__(self,path='',max_length = 10,eng_prefixes = (
                                                                'i am ', 'i m ',
                                                                'he is', 'he s ',
                                                                'she is', 'she s ',
                                                                'you are', 'you re ',
                                                                'we are', 'we re ',
                                                                'they are', 'they re '
                                                            )):
        self.character_set = CharacterSet()
        self.path = path
        self.max_length = max_length
        self.eng_prefixes = eng_prefixes

    def filterPair(self,pair):
        return (len(pair[0].split(' ')) < self.max_length and
                len(pair[1].split(' ')) < self.max_length and
                pair[1].startswith(self.eng_prefixes))

    def filterPairs(self,pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    def readLangs(self,lang1, lang2, reverse=False):
        file_name = join(self.path, f'{lang1}-{lang2}.txt')
        print(f'Reading lines from {file_name}')
        lines = open(file_name, encoding='utf-8').read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[self.character_set.normalizeString(s) for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)

        return input_lang, output_lang, pairs

    def prepareData(self,lang1, lang2, reverse=False):
        input_lang, output_lang, pairs = self.readLangs(lang1, lang2, reverse)
        print(f'Read {len(pairs)} sentence pairs' )
        pairs = self.filterPairs(pairs)
        print('Trimmed to {len(pairs)} sentence pairs')
        print('Counting words...')
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print('Counted words:')
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['train', 'test'],
                        default='train', help='Chooses between training or testing')

    training_group = parser.add_argument_group('Parameters for --action train')

    training_group.add_argument('--batch_size', default=128, type=int, help='Number of images per batch')
    training_group.add_argument('--N', default=5, type=int, help='Number of epochs')
    training_group.add_argument('--steps', default=5, type=int, help='Number of steps to an epoch')
    training_group.add_argument('--params', default='./params', help='Location for storing parameter files')
    training_group.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    training_group.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    training_group.add_argument('--optimizer', choices=OptimizerFactory.choices, default=OptimizerFactory.get_default(),
                                help='Optimizer to be used for training')
    training_group.add_argument('--restart', default=None, help='Restart from saved parameters')

    test_group = parser.add_argument_group('Parameters for --action test')
    test_group.add_argument('--file', default=None, help='Used to load weights')

    shared_group = parser.add_argument_group('General Parameters')
    shared_group.add_argument('--data', default='./data/rnn-1', help='Location of data files')
    shared_group.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    shared_group.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    shared_group.add_argument('--figs', default='./figs', help='Location for storing plot files')
    shared_group.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
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
    dataset = DataSet(path=args.data)
    dataset.prepareData('eng', 'fra', reverse=True)
    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
