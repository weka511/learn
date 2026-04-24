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
    Driver for cavi_np.py, the Coordinate Ascent Mean-Field Variational Inference (CAVI) example from Section 3 of Blei et al
    with data in 1, 2 or 3 dimensions. The driver apportions the load over multiple processes.
'''

from argparse import ArgumentParser
from glob import glob
from os import remove
from os.path import basename
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, rcParams, show
from multiprocessing import cpu_count, Process
from tempfile import gettempdir
import numpy as np
from shared.utils import generate_xkcd_colours, Splitter
from gmm import GaussionMixtureModel, get_name, create_colours
import cavi_nd as cavi

def ncores(s):
    '''
    Used to validate --processes
    '''
    if s == 'all':
        return None
    else:
        return int(s)
    
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
    parser.add_argument('--processes', type=ncores, default=cpu_count(), help='Number of processors')
    args = parser.parse_args()
    if args.processes == None:
        args.processes = args.M
    return args


def cavi_run(run_number: int, x_train: np.array, x_test: np.array, K: int, N: int,
             rng: np.random._generator.Generator, solution: cavi.Solution,
             path: str, file_name: str, BURN_IN: int, atol: float):
    '''
    Used by each process to create a Solution
    
    Parameters:
        run_number  Unique vaue for each run--used to calculate file name
        x_train     Training data *used for fitting)
        x_test      Test data (used for calculating ELBO)
        K           Number of Gaussians
        N           Maximum number of iterations per run
        rng         Random number generator
        solution    Used to store results
        path        Used for stopring data
        file_name   Used for stopring data
        BURN_IN     Minimum number of iterations
        atol        Tolerance for improving ELBO
    '''
    print(f'Run {run_number}')
    # When processes start, each one gets a copy of the random number generator,
    # so they will all generate the same sequnce unless something is done. We
    # therefore run each generator a random number of times to break syncronization
    dummy = rng.integers(0, 100, size=K * run_number + 7 + rng.integers(0, 99))

    m, s, c = cavi.initialize(x_train, K, rng=rng)
    solution.accumulateELBO(cavi.get_ELBO(m, s, c, x_train))
    for j in range(N):
        c = cavi.get_updated_assignments(m, s, x_train)
        m, s = cavi.get_updated_statistics(m, s, c, x_train)
        c_test = cavi.get_updated_assignments(m, s, x_test)
        solution.accumulateELBO(cavi.get_ELBO(m, s, c_test, x_test))
        if j > BURN_IN and solution.ELBO[-1] - solution.ELBO[-2] < atol:
            break

    solution.set_params(m, s, c, c_test)
    solution.save(get_solution_path(path, file_name, run_number))


def get_solution_path(path: str, prefix: str, run_number: int):
    '''
    Read solution back from file
    
    Parameters:
        path          Location of files
        prefix        First part of file name
        run_number    Unique identifier for each run
    '''
    return (Path(path) / f'{prefix}{run_number:04d}').with_suffix('.npz')


def run_processes(args, x_train, x_test, rng=np.random.default_rng(), prefix='foo'):
    '''
    Start and run specified number of processes until all runs completed
    
    Parameters:
        args
        x_train
        x_test
        rng
        prefix
    '''
    print(f'There are {cpu_count()} cores')
    run_number: int = 0
    while run_number < args.M:
        processes: list[Process] = []
        for i in range(args.processes):
            if run_number < args.M:
                processes.append(Process(
                    target=cavi_run,
                    args=(run_number, x_train, x_test, args.K,
                          args.N, rng, cavi.Solution(id=run_number),
                          gettempdir(), prefix, args.BURN_IN, args.atol)))
                run_number += 1
                processes[-1].start()

        for i in range(len(processes)):
            processes[i].join()


def create_solutions(temp_path: str, prefix: str, M: int):
    '''
    Read list of solutions from stored values
    '''
    Solutions = [cavi.Solution.create(get_solution_path(temp_path, prefix, i)) for i in range(M)]
    ELBOS = [solution.ELBO[-1] for solution in Solutions]
    return np.argmax(ELBOS), Solutions


def ensure_no_temporary_files_left(temp_path, prefix):
    '''
    Make sure there are no temporary files left overS
    '''
    removed_files = 0
    for f in glob(str(Path(temp_path) / prefix) + '*'):
        remove(f)
        removed_files += 1
    if removed_files > 0:
        print(f'Removed {removed_files} temporary files from {temp_path}')


def main():
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    temp_path = gettempdir()
    prefix = basename(__file__).split('.')[0]
    ensure_no_temporary_files_left(temp_path, prefix)
    model = GaussionMixtureModel()
    path_name = Path(args.path) / args.name

    splitter = Splitter(rng=rng, test_size=args.test)
    x_train, x_test = splitter.split(model.load(path_name.with_suffix('.npz')))
    run_processes(args, x_train, x_test, rng=rng, prefix=prefix)

    index_best, Solutions = create_solutions(temp_path, prefix, args.M)

    fig = figure(figsize=(12, 12))
    fig.suptitle(f'{args.name}')
    cavi.plotELBOs(index_best, Solutions, ax=fig.add_subplot(2, 1, 1))

    _, d = x_test.shape
    cavi.plot_clusters(d, x_test, index_best, Solutions, args.K, args.M,
                       ax=fig.add_subplot(2, 1, 2, projection='3d') if d == 3 else fig.add_subplot(2, 1, 2))

    fig.tight_layout(pad=3, h_pad=4)
    figs_path_name = Path(args.figs) / args.name
    fig.savefig(figs_path_name.with_suffix('.png'))

    ensure_no_temporary_files_left(temp_path, prefix)

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()


if __name__ == '__main__':
    main()
