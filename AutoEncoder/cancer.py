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
from matplotlib.pyplot import figure, hist, legend, savefig, show, title, xticks
from os.path           import join
from pandas            import read_csv
from prepare           import save_plot_dataset
from torch             import tensor
from torch.utils.data  import Dataset, DataLoader, random_split

class CancerDataset(Dataset):
    def __init__(self,
                 path      = r'D:\data\cancer_mutations',
                 file_name = 'cancer_mutations', ext='txt'):
        self.df = read_csv(join(path,f'{file_name}.{ext}'), sep='\t')
        self.labels = self.df['cancer_type']
        self.df.drop(['cancer_type'],axis=1,inplace=True)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (tensor(self.df.loc[[idx]].values),self.labels[idx])

def parse_args():
    '''Extract command line arguments'''
    parser = ArgumentParser(__doc__)
    parser.add_argument('action',
                        choices = ['split'])
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
    return parser.parse_args()

if __name__=='__main__':
    args    = parse_args()
    dataset = CancerDataset(path      = args.root,
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

