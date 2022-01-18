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
    Ensure that training and validation datasets have been downloaded.
    Partition training dataset into training and validation.
'''

from argparse               import ArgumentParser
from matplotlib.pyplot      import figure, hist, legend, savefig, show, title, xticks
from os.path                import join
from torch                  import Generator, load, save
from torch.utils.data       import random_split
from torchvision.datasets   import MNIST
from torchvision.transforms import Compose, ToTensor



def save_plot_dataset(dataset,name,
                      epsilon = 0.005,
                      colour  = 'xkcd:red',
                      path    = ''):
    '''
       Save dataset and plot frequencies of all classes

       Parameters:
            dataset   Dataset to be saved
            name      Base file name for save
       Keyword Parameters:
            epsilon   Used to ensure that bins don't include the lower limit
            colour    For plotting histogram
            path      Pathe for savibg dataset
    '''
    save(dataset, join(path,name))
    subset = load(join(path,name))
    hist([y for _,y in subset],
         bins    = [x+epsilon for x in range(-1,10)],
         alpha   = 0.5,
         density = True,
         label = f'{join(path,name)} {len(subset)} records',
         color = colour)

def parse_args():
    '''Extract command line arguments'''
    parser = ArgumentParser(__doc__)
    parser.add_argument('--root',
                        default = r'd:\data',
                        help    = 'Path for storing downloaded data')
    parser.add_argument('--data',
                        default = r'./data',
                        help    = 'Path for storing intermediate data, such as training and validation and saved weights')
    parser.add_argument('--figs',
                        default = r'./figs',
                        help    = 'Path for storing plots')
    parser.add_argument('--validation',
                        type    = float,
                        default = 0.1,
                        help    = 'Size of validation dataset as fraction of test')
    parser.add_argument('--seed',
                        type    = int,
                        default = None,
                        help    = 'Used to initialize random number generator')
    parser.add_argument('--show',
                        default = False,
                        action  = 'store_true',
                        help    = 'Controls whether histograms shown')
    return parser.parse_args()

if __name__=='__main__':
    args              = parse_args()

    transform         = Compose([ToTensor()])

    train_dataset     = MNIST(root      = args.root,
                              train     = True,
                              transform = transform,
                              download  = True)
    test_dataset      = MNIST(root      = args.root,
                              train     = False,
                              transform = transform,
                              download  = True)

    len_validation    = int(args.validation * len(train_dataset))
    train,validation  = random_split(train_dataset,[len(train_dataset)-len_validation,len_validation],
                                     generator = None if args.seed==None else Generator().manual_seed(args.seed))

    figure(figsize=(10,10))
    save_plot_dataset(train,'train.pt',
                      path = args.data)
    save_plot_dataset(validation,'validation.pt',
                      path   = args.data,
                      colour = 'xkcd:blue')

    xticks(range(-1,10))
    legend()
    title ('Frequencies of Classes')
    savefig(join(args.figs,'freqs'))
    if args.show:
        show()
