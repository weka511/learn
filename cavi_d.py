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
    Plot results from cavi_w.py
'''

from argparse import ArgumentParser
from os.path import basename
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, rcParams, show
from tempfile import gettempdir
import numpy as np
from shared.utils import generate_xkcd_colours
from gmm import GaussionMixtureModel, get_name, create_colours
import cavi_nd as cavi

def parse_args():
    '''
    Parse command line argumengts
    '''
    parser = ArgumentParser(__doc__)
    parser.add_argument('name', help='Name of data file')
    parser.add_argument('--K', type=int, default=3, help='Number of Gaussians')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot displayed')
    parser.add_argument('--skip', type=int, default=1, help='Skip this maany iterations')
    parser.add_argument('--figs', default='./figs', help='Folder to store plots')
    parser.add_argument('--prefix', default=Path(__file__).stem,help='Prefix for saving solutions')
    return parser.parse_args()
 
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

def create_solutions(temp_path: str, prefix: str):
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

    index_best, Solutions = create_solutions(gettempdir(), args.prefix)
    z = np.load(get_solution_path(gettempdir(), args.prefix, '-data'))
    x_test = z['x_test']
    
    fig = figure(figsize=(12, 12))
    fig.suptitle(f'{args.name}')
    cavi.plotELBOs(index_best, Solutions, ax=fig.add_subplot(2, 1, 1),SKIP=args.skip)
    _, d = x_test.shape
    cavi.plot_clusters(d, x_test, index_best, Solutions, args.K, len(Solutions),
                       ax=fig.add_subplot(2, 1, 2, projection='3d') if d == 3 else fig.add_subplot(2, 1, 2))

    fig.tight_layout(pad=3, h_pad=4)
    figs_path_name = Path(args.figs) / args.name
    fig.savefig(figs_path_name.with_suffix('.png'))
    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()    

if __name__ == '__main__':
    main()