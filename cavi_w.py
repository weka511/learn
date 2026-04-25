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
from matplotlib.pyplot import figure, rcParams, show,subplot
from multiprocessing import cpu_count, Process, Pool
from tempfile import gettempdir
import numpy as np
from shared.utils import generate_xkcd_colours, Splitter
from gmm import GaussionMixtureModel, get_name, create_colours
import cavi_nd as cavi
import matplotlib
matplotlib.use('Agg')
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

def create_solution(x_train,x_test,K,rng = np.random.default_rng()):
    Product = cavi.Solution()
    m, s, c = cavi.initialize(x_train, K, rng=rng)
    Product.set_params(m,s,c,[])
    Product.accumulateELBO(cavi.get_ELBO(m, s, c, x_train))
    return Product



class Explorer:
    def __init__(self,x_train, x_test, K, N, BURN_IN, atol):
        self.x_train = x_train
        self.x_test = x_test
        self.K = K
        self.N = N
        self.BURN_IN = BURN_IN
        self.atol = atol
    
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
        return solution

def main():
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model = GaussionMixtureModel()
    path_name = Path(args.path) / args.name

    splitter = Splitter(rng=rng, test_size=args.test)
    x_train, x_test = splitter.split(model.load(path_name.with_suffix('.npz')))
    
    explorer = Explorer(x_train, x_test, args.K, args.N, args.BURN_IN, args.atol)
    Solutions0 = [create_solution(x_train,x_test,args.K,rng=rng) for _ in range(args.M)]
    with Pool(processes=cpu_count()-1) as pool:
        Solutions = pool.map(explorer.explore, Solutions0)
    ELBOS = [solution.ELBO[-1] for solution in Solutions]
    index_best = np.argmax(ELBOS)
    print (len(ELBOS),ELBOS[index_best])
    print (ELBOS)
    #fig = figure(figsize=(12, 12))
    #fig.suptitle(f'{args.name}')
    #cavi.plotELBOs(index_best, Solutions, ax=fig.add_subplot(2, 1, 1))   
    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()    

if __name__ == '__main__':
    main()
    