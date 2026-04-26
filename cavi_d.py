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

def get_solution_path(path: str, prefix: str, run_number: int):
    '''
    Read solution back from file
    
    Parameters:
        path          Location of files
        prefix        First part of file name
        run_number    Unique identifier for each run
    '''
    return (Path(path) / f'{prefix}{run_number:04d}').with_suffix('.npz')

def create_solutions(temp_path: str, prefix: str, M: int):
    '''
    Read list of solutions from stored values
    '''
    Solutions = []
    i = 0
    while True:
        try:
            Solutions.append(cavi.Solution.create(get_solution_path(temp_path, prefix, i)))
            i += 1
        except FileNotFoundError:
            break
    ELBOS = [solution.ELBO[-1] for solution in Solutions]
    return np.argmax(ELBOS), Solutions

def main():
    start = time()
    args = parse_args()

    temp_path = gettempdir()
    prefix = 'cavi_w' 
    
    index_best, Solutions = create_solutions(temp_path, prefix, args.M)

    fig = figure(figsize=(12, 12))
    #fig.suptitle(f'{args.name}')
    cavi.plotELBOs(index_best, Solutions, ax=fig.add_subplot(2, 1, 1),SKIP=1)
    
    
    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()    

if __name__ == '__main__':
    main()