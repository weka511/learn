#!/usr/bin/env python

# Copyright (C) 2026 Simon Crase  simon@greenweaves.nz

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>.

'''
    Generate test data from a Bradley-Terry model, then try to fit to the data.
    Compare fitted parameters to original. How large does dataset need to be?
'''


from argparse import ArgumentParser
from pathlib import Path
from matplotlib.pyplot import figure,show
import numpy as np
import pandas as pd
from bt import bt

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--N', type=int, default=50,help='Number of Iterations')
    parser.add_argument('--burn', type=int, default=0,help='Burn in: skip this many iterations when we display')
    parser.add_argument('out',  help='Name of plot file')
    parser.add_argument('--data',default = './data/sumo')
    parser.add_argument('--year',type=int,default=2019)
    return parser.parse_args()

class Results:
    def __init__(self,year,data):
        self.year = year
        file_name = str(year)
        path_name = (Path(data) / file_name).with_suffix('.csv')
        self.basho = pd.read_csv(path_name)
        shikona = self.basho['rikishi1_shikona']
        z=0
        

if __name__ == '__main__':
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    result = Results(args.year,args.data)
    