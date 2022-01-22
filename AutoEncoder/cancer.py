# Copyright (C) 2022 Greenweaves Software Limited

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>

'''
    Read the cancer mutations dataset
    Partition training dataset into training and validation.
'''

from argparse          import ArgumentParser
from AutoEncoder       import AutoEncoder
from matplotlib.pyplot import figure, hist, legend, savefig, show, title, xticks
from multiprocessing   import cpu_count
from os.path           import join
from pandas            import read_csv
from prepare           import save_plot_dataset
from torch             import load, tensor
from torch.utils.data  import Dataset, DataLoader, random_split
from tune              import Trainer,Plotter,plot

class CancerDataset(Dataset):
    def __init__(self,
                 path      = r'D:\data\cancer_mutations',
                 file_name = 'cancer_mutations', ext='txt'):
        self.df     = read_csv(join(path,f'{file_name}.{ext}'), sep='\t')
        self.labels = self.df['cancer_type']
        self.df.drop(['cancer_type'],
                     axis    = 1,
                     inplace = True)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (tensor(self.df.loc[[idx]].values),self.labels[idx])

def parse_args():
    '''Extract command line arguments'''
    parser = ArgumentParser(__doc__)
    parser.add_argument('action',
                        choices = ['split', 'tune'])
    parser.add_argument('--validation',
                        default = 0.2,
                        type    = float)
    parser.add_argument('--root',
                        default = r'd:\data\cancer_mutations',
                        help    = 'Path for storing downloaded data')
    parser.add_argument('--cancer',
                        default = r'cancer_mutations',
                        help    = 'Path for storing downloaded data')
    parser.add_argument('--data',
                        default = r'./data',
                        help    = 'Path for storing intermediate data, such as training and validation and saved weights')
    parser.add_argument('--figs',
                        default = r'./figs',
                        help    = 'Path for storing plots')
    parser.add_argument('--seed',
                        type    = int,
                        default = None,
                        help    = 'Used to initialize random number generator')
    parser.add_argument('--show',
                        default = False,
                        action  = 'store_true',
                        help    = 'Controls whether histograms shown')
    parser.add_argument('--encoder',
                        nargs   = '+',
                        type    = int,
                        default = [778, 400, 200, 100, 50, 25],
                        help    = 'Sizes of each layer in encoder')
    parser.add_argument('--dimension',
                        type    = int,
                        default = 6,
                        help    = 'Dimension of encoded vectors')
    parser.add_argument('--decoder',
                        nargs   = '*',
                        type    = int,
                        default = [],
                        help    = 'Sizes of each layer in decoder (omit if decoder is a mirror image of encoder)')
    parser.add_argument('--nonlinearity',
                        nargs   = '+',
                        default = ['relu'],
                        help    = 'Non linearities between layers (default relu)')
    parser.add_argument('--batch',
                        default = 128,
                        type    = int,
                        help    = 'Training batch size')
    parser.add_argument('--lr',
                        default = 0.001,
                        type    = float,
                        help    = 'Learning rate')
    parser.add_argument('--weight_decay',
                        default = 0.01,
                        type    = float,
                        help    = 'Weight decay')
    parser.add_argument('--N',
                        default = 100,
                        type    = int,
                        help    = 'Maximum number of epochs')

    return parser.parse_args()

def split_dataset(args):
    dataset           = CancerDataset(path      = args.root,
                                      file_name = args.cancer)
    len_validation    = int(args.validation * len(dataset))
    train,validation  = random_split(dataset,[len(dataset)-len_validation,len_validation],
                                     generator = None if args.seed==None else Generator().manual_seed(args.seed))
    figure(figsize=(10,10))
    save_plot_dataset(train,'cancer-train.pt',
                      density = False,
                      path    = args.data)
    save_plot_dataset(validation,'cancer-validation.pt',
                      path    = args.data,
                      density = False,
                      colour  = 'xkcd:blue')

    xticks(range(-1,10))
    legend()
    title ('Frequencies of Classes')
    savefig(join(args.figs,'cancer-freqs'))
    if args.show:
        show()

def tune_autoencoder(args):
    enl,dnl = AutoEncoder.get_non_linearity(args.nonlinearity)
    trainer = Trainer(AutoEncoder(encoder_sizes         = args.encoder,
                                  encoding_dimension    = args.dimension,
                                  encoder_non_linearity = enl,
                                  decoder_non_linearity = dnl,
                                  decoder_sizes         = args.decoder),
                      DataLoader(load(join(args.data,
                                           'cancer-train.pt')),
                                 batch_size  = args.batch,
                                 shuffle     = True,
                                 num_workers = cpu_count()),
                      DataLoader(load(join(args.data,
                                           'cancer-validation.pt')),
                                 batch_size  = 32,
                                 shuffle     = False,
                                 num_workers = cpu_count()),
                      lr           = args.lr,
                      weight_decay = args.weight_decay,
                      path         = args.data)
    loss = trainer.train(N_EPOCHS  = args.N,
                         args_dict = {
                             'nonlinearity' : args.nonlinearity,
                             'encoder'      : args.encoder,
                             'decoder'      : args.decoder,
                             'dimension'    : args.dimension,
                         })
    with Plotter('Cancer', args, loss):
        plot(trainer.Losses, 'bo',
             label = 'Training Losses')
        plot(trainer.ValidationLosses, 'r+',
             label = 'Validation Losses')
        legend()
    if args.show:
        show()

if __name__=='__main__':
    args    = parse_args()
    if args.action=='split':
        split_dataset(args)
    if args.action=='tune':
        tune_autoencoder(args)
