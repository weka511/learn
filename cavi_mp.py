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
from os.path import basename
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, rcParams, show
from multiprocessing import cpu_count, Process
from tempfile import gettempdir
import numpy as np
from shared.utils import generate_xkcd_colours,Splitter
from gmm import GaussionMixtureModel, get_name, create_colours
from cavi_nd import Solution, initialize, get_updated_assignments, get_updated_statistics, get_ELBO,create_data_colours

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
  
def cavi_run(run_number:int,x_train,x_test,K,N,rng,solution,path,file_name,BURN_IN,atol):
    print (f'Run {run_number}')
    dummy = rng.integers(0, 100, size=K*run_number+7)
    m, s, c = initialize(x_train, K, rng=rng)
    solution.accumulateELBO(get_ELBO(m, s, c, x_train)) 
    for j in range(N):
        c = get_updated_assignments(m, s, x_train)
        m, s = get_updated_statistics(m, s, c, x_train)
        c_test = get_updated_assignments(m, s, x_test)
        solution.accumulateELBO(get_ELBO(m, s, c_test, x_test))
        if j > BURN_IN and solution.ELBO[-1] - solution.ELBO[-2] < atol:
            break        

    solution.set_params(m, s, c,c_test)
    solution.save(get_solution_path(path,file_name,run_number))
    
def get_solution_path(path,file_name,run_number):
    return (Path(path) / f'{file_name}{run_number:04d}').with_suffix('.npz')

def run_processes(args,x_train,x_test,rng=np.random.default_rng(),prefix='foo'):
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
                                  args=(run_number,x_train,x_test,args.K,
                                        args.N,rng,solution,gettempdir(),prefix,args.BURN_IN,args.atol))
                processes.append(process)
                run_number += 1
                process.start()
            
        for i in range(len(processes)):  
            processes[i].join()
            
def create_solutions(temp_path,prefix,M):
    Solutions = [Solution.create(get_solution_path(temp_path,prefix,i)) for i in range(M)]
    ELBOS = [solution.ELBO[-1] for solution in Solutions]
    return np.argmax(ELBOS),Solutions

def plotELBOs(index_best,Solutions,ax=None):
    ELBO_colours = generate_xkcd_colours()
    for i in range(len(Solutions)):
        label = None
        linestyle = 'dotted'
        if i == index_best:
            label = f'best {Solutions[i].ELBO[-1]:.6}'
            linestyle = 'solid'
        ax.plot(range(len(Solutions[i].ELBO)),Solutions[i].ELBO, c=next(ELBO_colours), label=label, linestyle=linestyle)
    ax.legend()
    ax.set_title(f'ELBO for {len(Solutions)} runs, plotted with held-out data')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('ELBO')
    
def main():
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    temp_path = gettempdir()
    prefix = basename(__file__).split('.')[0]
    
    model = GaussionMixtureModel()
    path_name = Path(args.path) / args.name
    
    splitter = Splitter(rng=rng,test_size=args.test)
    x_train,x_test = splitter.split(model.load(path_name.with_suffix('.npz')))    
    run_processes(args,x_train,x_test,rng=rng,prefix=prefix)
       
    index_best,Solutions = create_solutions(temp_path,prefix,args.M)
    
    fig = figure(figsize=(12, 12))
    fig.suptitle(f'{args.name}')
    plotELBOs(index_best,Solutions,ax=fig.add_subplot(2, 1, 1))
    
    _,d = x_train.shape
    match d:
        case 1:
            ax2 = fig.add_subplot(2, 1, 2)
            n, _, _ = ax2.hist(x_test, bins='sturges', color='xkcd:blue', label='x',density=True)
            ax2.vlines(np.ravel(Solutions[index_best].m), 0, max(n), 
                       colors='xkcd:red', linestyles='dashed', label='Means (fitted)')
            ax2.set_xlabel('X')
            ax2.set_ylabel('p')
            ax2.legend()

        case 2:
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.scatter(x_test[:, 0], x_test[:, 1],
                        c=create_data_colours(x_test, Solutions[index_best].c_test, create_colours(args.K)), s=1)
            for k in range(args.K):
                ax2.scatter(Solutions[index_best].m[k, 0], Solutions[index_best].m[k, 1],
                            c='xkcd:black', marker='+', s=25)
            ax2.set_title(f'Solution with best ELBO: {Solutions[index_best].ELBO[-1]:.6} after {args.M} runs')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')

        case 3:
            ax2 = fig.add_subplot(2, 1, 2,projection='3d')
            ax2.scatter(x_test[:,0],x_test[:,1],x_test[:,2],
                        c=create_data_colours(x_test, Solutions[index_best].c_test, create_colours(args.K)), s=1)
            ax2.set_title(f'Solution with best ELBO: {Solutions[index_best].ELBO[-1]:.6} after {args.M} runs')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')    
        
    fig.tight_layout(pad=3,h_pad=4)
    figs_path_name = Path(args.figs) / args.name
    fig.savefig(figs_path_name.with_suffix('.png'))    
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s') 
    if args.show:
        show()    
    
    
if __name__ == '__main__':
    main()
