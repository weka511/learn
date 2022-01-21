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

from argparse         import ArgumentParser
from os.path          import join
from pandas           import read_csv
from torch            import tensor
from torch.utils.data import Dataset, DataLoader

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
        return (tensor(self.df.loc[[idx]].values),self.labels[i])

def parse_args():
    '''Extract command line arguments'''
    parser = ArgumentParser(__doc__)
    parser.add_argument('--root',
                        default = r'd:\data\cancer_mutations',
                        help    = 'Path for storing downloaded data')
    parser.add_argument('--data',
                        default = r'cancer_mutations',
                        help    = 'Path for storing downloaded data')
    return parser.parse_args()

if __name__=='__main__':
    args    = parse_args()
    dataset = CancerDataset(path      = args.root,
                            file_name = args.data)
    print (len(dataset))
    for i in range(3):
        print (dataset[i])
