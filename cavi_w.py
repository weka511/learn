#!/usr/bin/env python

# Copyright (C) 2026 Greenweaves Software Limited

# Simon A. Crase -- simon@greenweaves.nz

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://github.com/weka511/learn/blob/master/LICENSE or
# <http://www.gnu.org/licenses/>.

'''
    Driver for cavi_np.py, which used a Pool to apportion the load over multiple processes.
'''

from argparse import ArgumentParser
from glob import glob
from os import remove
from os.path import basename
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, rcParams, show
from multiprocessing import cpu_count, Process, Pool
from tempfile import gettempdir
import numpy as np
from shared.utils import generate_xkcd_colours, Splitter
from gmm import GaussionMixtureModel, get_name, create_colours
import cavi_nd as cavi

def parse_args():
    '''
    Parse command line argumengts
    '''
    parser = ArgumentParser(__doc__)
    parser.add_argument('name', help='Name of data file')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')
    parser.add_argument('--K', type=int, default=3, help='Number of Gaussians')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot displayed')
    parser.add_argument('--N', type=int, default=100, help='Maximum number of iterations per run')
    parser.add_argument('--M', type=int, default=25, help='Number of runs')
    parser.add_argument('--BURN_IN', type=int, default=3, help='Minimum number of iterations')
    parser.add_argument('--atol', type=float, default=1e-6, help='Tolerance for improving ELBO')
    parser.add_argument('--sigma', type=float, default=1, help='Standard deviation')
    parser.add_argument('--figs', default='./figs', help='Folder to store plots')
    parser.add_argument('--path', default='./data', help='Path to folder where data are stored')
    parser.add_argument('--test', type=float, default=0.1, help='Size of held out dataset')
    args = parser.parse_args()
    return args

def foo(n):
    return n**2

def main():
    args = parse_args()
    inputs = list(range(100))
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(foo, inputs)
    print( results)    

if __name__ == '__main__':
    main()
    