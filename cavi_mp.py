#!/usr/bin/env python

# Copyright (C) 2022-2026 Greenweaves Software Limited

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
    The Coordinate Ascent Mean-Field Variational Inference (CAVI) example from Section 3 of Blei et al
	with data in 1, 2 or 3 dimensions.
'''

from argparse import ArgumentParser
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, rcParams, show
from multiprocessing import cpu_count, Process
import numpy as np
from shared.utils import generate_xkcd_colours,Splitter
from gmm import GaussionMixtureModel, get_name, create_colours
from cavi_nd import Solution, initialize, get_updated_assignments, get_updated_statistics, get_ELBO

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('name',help='Name of data file')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')
    parser.add_argument('--K', type=int, default=3, help='Number of Gaussians')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot displayed')
    parser.add_argument('--N', type=int, default=100, help='Number of iterations per run')
    parser.add_argument('--M', type=int, default=25, help='Number of runs')
    parser.add_argument('--BURN_IN', type=int, default=3, help='Minimum number of iterations')
    parser.add_argument('--atol', type=float, default=1e-6, help='Tolerance for improving ELBO')
    parser.add_argument('--sigma', type=float, default=1, help='Standard deviation')
    parser.add_argument('--figs', default='./figs', help='Folder to store plots')
    parser.add_argument('--path', default='./data', help='Path to folder where data are stored')
    parser.add_argument('--test', type=float, default=0.1,help='Size of held out dataset')
    parser.add_argument('--processes', type=int, default=cpu_count(), help='Number of processors')
    return parser.parse_args()
  
def cavi_run(run_number:int,x_train,x_test,K,N,rng,solution,path,file_name):
    print (f'Run {run_number}')

    m, s, c = initialize(x_train, K, rng=rng)
    solution.accumulateELBO(get_ELBO(m, s, c, x_train)) 
    for j in range(N):
        print (f'Run {run_number},j={j}') 
        c = get_updated_assignments(m, s, x_train)
        m, s = get_updated_statistics(m, s, c, x_train)
        c_test = get_updated_assignments(m, s, x_test)
        solution.accumulateELBO(get_ELBO(m, s, c_test, x_test))

    solution.set_params(m, s, c,c_test)    
    with open((Path(path) / f'{file_name}{j:.04d}').with_suffix('.txt')) as f:
        f.write(f'{solution}\n'))

    
def run_processes(args,x_train,x_test,rng=np.random.default_rng()):
    print (f'There are {cpu_count()} cores')
    run_number: int = 0
    solutions = []
    while run_number < args.M:
        processes = []
        for i in range(args.processes):
            if run_number < args.M:
                solution = Solution(id=run_number)
                solutions.append(solution)
                process = Process(target=cavi_run,
                                  args=(run_number,x_train,x_test,args.K,args.N,rng,solution,args.path,'foo'))
                processes.append(process)
                run_number += 1
                process.start()
            
        for i in range(len(processes)):  
            processes[i].join()

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ELBO_colours = generate_xkcd_colours()
    model = GaussionMixtureModel()
    path_name = Path(args.path) / args.name
    x = model.load(path_name.with_suffix('.npz'))
    splitter = Splitter(rng=rng,test_size=args.test)
    x_train,x_test = splitter.split(x)    
    run_processes(args,x_train,x_test,rng=rng)
    
if __name__ == '__main__':
    main()
