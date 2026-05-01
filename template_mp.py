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
    Template for using multiprocessing library
'''

from argparse import ArgumentParser
from glob import glob
from os import remove,system
from pathlib import Path
from time import time
from multiprocessing import cpu_count, Process, Pool#,Lock
from tempfile import gettempdir
import numpy as np
from shared.utils import Splitter
from gmm import GaussionMixtureModel, get_name
import cavi_nd as cavi

class Model:
    def load(self,path_name):
        '''
        Load data and its sufficient statistics

        Parameters:
            path_name   Full path name of file
        '''
        with open(path_name, 'rb') as f:
            npzfile = np.load(f)
            x = npzfile['data']
            return x
    
class Explorer:
    '''
    This class provided a function, explore, which explores the solution space
    '''
    def __init__(self,x_train, x_test, N, 
                 path=gettempdir(),prefix='cavi_w',data_suffix='-data'):
        self.x_train = x_train
        self.x_test = x_test
        self.N = N
        self.path = path
        self.prefix = prefix
        self.data_suffix = data_suffix
        
    def ensure_no_temporary_files_left(self):
        '''
        Make sure there are no temporary files left over
        '''
        removed_files = 0
        for f in glob(str(Path(self.path) / self.prefix) + '*'):
            remove(f)
            removed_files += 1
        if removed_files > 0:
            print(f'Removed {removed_files} temporary files from {self.path}')    
        
    def save_test_data(self):
        '''
        Used to save test data
        '''
        np.savez(get_solution_path(self.path,self.prefix,self.data_suffix),
                 x_test=self.x_test)
        
    def create_solution(self,rng = np.random.default_rng(),id=None):
        '''
        Create a Solution and initialize using training data
        '''
        raise Exception(TBP)
    
    def explore(self,solution):
        raise Exception(TBP)
        
def parse_args():
    '''
    Parse command line argumengts
    '''
    parser = ArgumentParser(__doc__)
    parser.add_argument('name', help='Name of data file')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')
    parser.add_argument('--N', type=int, default=100, help='Maximum number of iterations per run')
    parser.add_argument('--M', type=int, default=25, help='Number of runs')
    parser.add_argument('--path', default='./data', help='Path to folder where data are stored')
    parser.add_argument('--test', type=float, default=0.1, help='Size of held out dataset')
    parser.add_argument('--prefix', default=Path(__file__).stem,help='Prefix for saving solutions')
    parser.add_argument('--debug',default=False, action='store_true',help='No multiprogramming')
    return parser.parse_args()

def get_solution_path(path, prefix, identifier):
    '''
    Used to write solution to a from file or read it back again
    
    Parameters:
        path          Location of files
        prefix        First part of file name
        identifier    Unique identifier for each file
        
    Returns:
        Path name for load or save
    '''
    try:
        return (Path(path) / f'{prefix}{identifier:04}').with_suffix('.npz')
    except Exception as e:
        return (Path(path) / f'{prefix}{identifier}').with_suffix('.npz')
 
def main():
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
      
    path_name = Path(args.path) / args.name
    model = Model()
    model = model.load(path_name.with_suffix('.npz'))
    splitter = Splitter(rng=rng, test_size=args.test)
    x_train, x_test = splitter.split(model)
    
    explorer = Explorer(x_train, x_test, args.N, prefix=args.prefix)
    explorer.ensure_no_temporary_files_left()
    explorer.save_test_data()
    intial_values = [explorer.create_solution(rng=rng,id=i) for i in range(args.M)]
    if debug:
        Solutions = list(map(explorer.explore, initial_values))
    else:
        with Pool(processes=cpu_count()-1) as pool:
            Solutions = pool.map(explorer.explore, initial_values)
    
    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')
    system(f'python cavi_d.py {args.name}  --K {args.K} --show --prefix {args.prefix}')
    
if __name__ == '__main__':
    main()    