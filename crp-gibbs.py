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

'''Attempt to cluster data using simple Gibbs sampling'''

from argparse import ArgumentParser
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc
import numpy as np
from shared.utils import Logger

def parse_args():
    parser = ArgumentParser(description=__doc__)

    parser.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
    parser.add_argument('--data', default='./data', help='Location of data files')
    return parser.parse_args()

def main():
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
 
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    
    fig = figure(figsize=(8, 8))

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
        
if __name__=='__main__':
    main()
