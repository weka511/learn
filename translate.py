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
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from utils import Logger, get_seed, user_has_requested_stop, get_device, get_moving_average
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

    MAX_LENGTH = 10

    def __init__(self, path='', eng_prefixes=(
        'i am ', 'i m ',
        'he is', 'he s ',
        'she is', 'she s ',
        'you are', 'you re ',
        'we are', 'we re ',
        'they are', 'they re '
    )):
        self.character_set = CharacterSet()
        self.path = path
        self.eng_prefixes = eng_prefixes

    def filterPair(self, pair):
        return (len(pair[0].split(' ')) < DataSet.MAX_LENGTH and
                len(pair[1].split(' ')) < DataSet.MAX_LENGTH and
                pair[1].startswith(self.eng_prefixes))

    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    def readLangs(self, lang1, lang2, reverse=False):
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

    def prepareData(self, lang1, lang2, reverse=False):
        input_lang, output_lang, pairs = self.readLangs(lang1, lang2, reverse)
        print(f'Read {len(pairs)} sentence pairs')
        pairs = self.filterPairs(pairs)
        print(f'Trimmed to {len(pairs)} sentence pairs')
        print('Counting words...')
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print('Counted words:')
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    r'''
                 input  ______________
                   \  /              |
                    \ /               |
                embedding             |
                    |                 |
                    |                 |
                embedded              |
                    |                 |
                    |                 |
                   GRU                |
                    /\                |
                   /  \               |
                  /    \              |
                 /      \             |
              output  hidden          |
                        |             |
                        |_____________|
    '''
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device='cpu'):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(Lang.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(DataSet.MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None: # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else: # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, device='cpu'):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(Lang.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(DataSet.MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['train', 'test'],
                        default='train', help='Chooses between training or testing')

    training_group = parser.add_argument_group('Parameters for --action train')

    training_group.add_argument('--batch_size', default=128, type=int, help='Number of sentnce pairs per batch')
    training_group.add_argument('--N', default=50, type=int, help='Number of epochs')
    training_group.add_argument('--params', default='./params', help='Location for storing parameter files')
    training_group.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    training_group.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    training_group.add_argument('--optimizer', choices=OptimizerFactory.choices, default=OptimizerFactory.get_default(),
                                help='Optimizer to be used for training')
    training_group.add_argument('--restart', default=None, help='Restart from saved parameters')
    training_group.add_argument('--decoder', choices=DecoderFactory.choices, default=DecoderFactory.get_default(),
                                help='Decoder to be used for training')
    training_group.add_argument('--hidden_size', default=128, type=int, help='Number of elements in hidden layer')
    training_group.add_argument('--output_size', default=5, type=int, help='Number of elements in output layer')
    training_group.add_argument('--dropout', default=0.1, type=float, help='Used by attention decoder for dropout')
    training_group.add_argument('--freq', default=5, type=int, help='Used to specify how frequently to print progress')

    test_group = parser.add_argument_group('Parameters for --action test')
    test_group.add_argument('--M', default=55, type=int, help='Number sentences to test')

    shared_group = parser.add_argument_group('General Parameters')
    shared_group.add_argument('--data', default='./data/rnn-1', help='Location of data files')
    shared_group.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    shared_group.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    shared_group.add_argument('--figs', default='./figs', help='Location for storing plot files')
    shared_group.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
    parser.add_argument('--file', default=__file__, help='Used to save figure')
    return parser.parse_args()

class DecoderFactory:
    choices = [
        'attention',
        'decoder'
    ]

    @staticmethod
    def get_default():
        return DecoderFactory.choices[0]

    @staticmethod
    def create(args,output_size=100,device='cpu'):
        match args.decoder:
            case 'attention':
                return AttnDecoderRNN( args.hidden_size, output_size, args.dropout, device=device)
            case 'decoder':
                return DecoderRNN(args.hidden_size, output_size, device=device)

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device='cpu'):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(Lang.EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def get_dataloader(batch_size, path='./', device='cpu'):
    dataset = DataSet(path=path)
    input_lang, output_lang, pairs = dataset.prepareData('eng', 'fra', reverse=True)

    n = len(pairs)
    input_ids = np.zeros((n, DataSet.MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, DataSet.MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(Lang.EOS_token)
        tgt_ids.append(Lang.EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader, pairs


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion):
    '''
    Train for one epoch

    Parameters:
        dataloader
        encoder
        decoder
        encoder_optimizer
        decoder_optimizer
        criterion
    '''
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(dataloader, encoder, decoder, n_epochs,
          freq=100,criterion = nn.NLLLoss(),save_file='save.pth'):
    '''
    Train network over the range of epochs

    Parameters:
        dataloader
        encoder
        decoder
        n_epochs,
        freq
        criterion
    '''
    losses = []
    print_loss_total = 0

    encoder_optimizer = OptimizerFactory.create(encoder,args)
    decoder_optimizer = OptimizerFactory.create(decoder,args)

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if epoch % freq == 0:
            print_loss_avg = print_loss_total / freq
            print_loss_total = 0
            print(f'{epoch}, {int((epoch / n_epochs) * 100)}%, {print_loss_avg}')

        losses.append(loss)

        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'losses': losses,
        }, save_file)

    return losses


def evaluate(encoder, decoder, sentence, input_lang, output_lang, device='cpu'):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device=device)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == Lang.EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluateRandomly(encoder, decoder, pairs, n=10, rng=np.random.default_rng(), device='cpu'):
    for i in range(n):
        pair = rng.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang, device=device)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def get_file_name(args):
    return f'{Path(args.file).stem}-{args.decoder}-{args.N}'

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

    input_lang, output_lang, train_dataloader, pairs = get_dataloader(batch_size=args.batch_size, path=args.data, device=device)

    encoder = EncoderRNN(input_lang.n_words, args.hidden_size).to(device)
    decoder = DecoderFactory.create(args,output_size=output_lang.n_words,device=device)

    match args.action:
        case 'train':
            losses = train(train_dataloader, encoder, decoder, args.N, freq=args.freq,
                           save_file=join(args.params, get_file_name(args)+'.pth'))

            encoder.eval()
            decoder.eval()
            evaluateRandomly(encoder, decoder, pairs, rng=rng, device=device)

            ax1 = fig.add_subplot(1, 1, 1)
            epochs = range(1,len(losses)+1)
            epochs_moving_average,losses_moving_average = get_moving_average(epochs,losses)
            ax1.plot(epochs_moving_average, losses_moving_average)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'{Path(args.file).stem}: {args.decoder},N={args.N}')

            fig.savefig(join(args.figs, get_file_name(args)))

        case 'test':
            loaded = torch.load(join(args.params, get_file_name(args)+'.pth'))
            encoder.load_state_dict(loaded['encoder_state_dict'])
            decoder.load_state_dict(loaded['decoder_state_dict'])
            matches = 0
            mismatches = 0
            with open(get_file_name(args)+'.txt','w') as mismatch_file:
                for pair in pairs:
                    if matches+mismatches > args.M: break
                    pair = rng.choice(pairs)
                    output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang, device=device)
                    output_sentence = ' '.join(output_words[:-1])
                    if pair[1] == output_sentence:
                        matches += 1
                    else:
                        mismatches += 1
                        mismatch_file.write(f'> {pair[0]}\n')
                        mismatch_file.write(f'= {pair[1]}\n')
                        mismatch_file.write(f'< {output_sentence}\n\n')

            print (f'{mismatches} mismatches out of {matches+mismatches} pairs, accuracy = {int(100*matches/(matches+mismatches))}%')

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
