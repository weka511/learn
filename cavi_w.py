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
from multiprocessing import cpu_count, Process, Pool#,Lock
from tempfile import gettempdir
import numpy as np
from shared.utils import Splitter
from gmm import GaussionMixtureModel, get_name
import cavi_nd as cavi

def parse_args():
    '''
    Parse command line argumengts
    '''
    parser = ArgumentParser(__doc__)
    parser.add_argument('name', help='Name of data file')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')
    parser.add_argument('--K', type=int, default=3, help='Number of Gaussians')
    parser.add_argument('--N', type=int, default=100, help='Maximum number of iterations per run')
    parser.add_argument('--M', type=int, default=25, help='Number of runs')
    parser.add_argument('--BURN_IN', type=int, default=3, help='Minimum number of iterations')
    parser.add_argument('--atol', type=float, default=1e-6, help='Tolerance for improving ELBO')
    parser.add_argument('--sigma', type=float, default=1, help='Standard deviation')
    parser.add_argument('--path', default='./data', help='Path to folder where data are stored')
    parser.add_argument('--test', type=float, default=0.1, help='Size of held out dataset')
    args = parser.parse_args()
    return args

def get_solution_path(path, prefix, run_number):
    '''
    Read solution back from file
    
    Parameters:
        path          Location of files
        prefix        First part of file name
        run_number    Unique identifier for each run
    '''
    try:
        return (Path(path) / f'{prefix}{run_number:04}').with_suffix('.npz')
    except Exception as e:
        return (Path(path) / f'{prefix}{run_number}').with_suffix('.npz')

class Explorer:
    def __init__(self,x_train, x_test, K, N, BURN_IN, atol,path=gettempdir(),prefix='cavi_w'):
        self.x_train = x_train
        self.x_test = x_test
        self.K = K
        self.N = N
        self.BURN_IN = BURN_IN
        self.atol = atol
        self.path = path
        self.prefix = prefix
        
    def save_test_data(self,path):
        np.savez(path,x_test=self.x_test)
   
    def create_solution(self,rng = np.random.default_rng(),id=None):
        Product = cavi.Solution(id=id)
        m, s, c = cavi.initialize(self.x_train, self.K, rng=rng)
        Product.set_params(m,s,c,[])
        Product.accumulateELBO(cavi.get_ELBO(m, s, c, self.x_train))
        return Product
        
    def explore(self,solution):
        m = solution.m
        s = solution.s
    
        for j in range(self.N):
            c = cavi.get_updated_assignments(m, s, self.x_train)
            m, s = cavi.get_updated_statistics(m, s, c, self.x_train)
            c_test = cavi.get_updated_assignments(m, s, self.x_test)
            solution.accumulateELBO(cavi.get_ELBO(m, s, c_test, self.x_test))
            if j > self.BURN_IN and solution.ELBO[-1] - solution.ELBO[-2] < self.atol:
                break
    
        solution.set_params(m, s, c, c_test) 
        print (cavi.get_ELBO(m, s, c_test, self.x_test))
        solution.save(get_solution_path(self.path,self.prefix,solution.id))
        self.save_test_data(get_solution_path(self.path,self.prefix,'-data'))
        return solution
    
    def get_solution_path(self):
        '''
        Read solution back from file
        
        Parameters:
            path          Location of files
            prefix        First part of file name
            run_number    Unique identifier for each run
        '''
        self.run_number += 1
        return (Path(self.path) / f'{prefix}{self.run_number:04}').with_suffix('.npz') 

def main():
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model = GaussionMixtureModel()
    path_name = Path(args.path) / args.name

    splitter = Splitter(rng=rng, test_size=args.test)
    x_train, x_test = splitter.split(model.load(path_name.with_suffix('.npz')))
    
    explorer = Explorer(x_train, x_test, args.K, args.N, args.BURN_IN, args.atol)
    with Pool(processes=cpu_count()-1) as pool:
        Solutions = pool.map(explorer.explore, 
                             [explorer.create_solution(rng=rng,id=i) for i in range(args.M)])
    
    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s') 

if __name__ == '__main__':
    main()
    