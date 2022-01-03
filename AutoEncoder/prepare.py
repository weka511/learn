# Copyright (C) 2021 Greenweaves Software Limited

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

from argparse               import ArgumentParser
from matplotlib.pyplot      import hist, legend, show, xticks
from os.path                import join
from torch                  import Generator, load, save
from torch.utils.data       import random_split
from torchvision.datasets   import MNIST
from torchvision.transforms import Compose, ToTensor

if __name__=='__main__':
    parser = ArgumentParser('Download dataset and partition training dataset into training and validation')
    parser.add_argument('--root',                   default = r'd:\data')
    parser.add_argument('--validation', type=float, default=0.1,  help = 'Size of validation dataset as fraction of test')
    parser.add_argument('--seed',       type=int,   default=None, help = 'Used to initialize random number generator')
    args           = parser.parse_args()

    transform      = Compose([ToTensor()])

    train_dataset  = MNIST(root      = args.root,
                           train     = True,
                           transform = transform,
                           download  = True)
    test_dataset   = MNIST(root      = args.root,
                           train     = False,
                           transform = transform,
                           download  = True)

    len_validation = int(args.validation * len(train_dataset))
    train_dataset,validation_dataset  = random_split(train_dataset,[len(train_dataset)-len_validation,len_validation],
                                                     generator = None if args.seed==None else Generator().manual_seed(args.seed))

    save(train_dataset,'train.pt')
    save(validation_dataset,'validation.pt')
    f1 = load('train.pt')
    f2 = load('validation.pt')
    hist([y for _,y in f1],
         bins    = 10,
         alpha   = 0.5,
         density = True,
         label = f'train {len(f1)} records')
    hist([y for _,y in f2],
         bins    = 10,
         alpha   = 0.5,
         density = True,
         label = f'validate {len(f2)} records')
    xticks(range(-1,10))
    legend()
    show()
