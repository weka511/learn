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

'''
Sampling the Dirichlet distribution
'''

from argparse import ArgumentParser
from os.path import splitext,join
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc
import numpy as np
from shared.utils import Logger,generate_xkcd_colours

def parse_args():
    parser = ArgumentParser(description=__doc__)

    parser.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
    parser.add_argument('--data', default='./data', help='Location of data files')
    parser.add_argument('--N', type=int, default=1000, help='Number of samples')
    parser.add_argument('--alpha', default=[1000.0,1.0,0.001],nargs='+',help='Parameters for Dirichlet Distribution')

    return parser.parse_args()

def plot(samples,m=2,n=2,alpha=[0.1,0.1,0.1],figs='./figs'):
    fig = figure(figsize=(8, 8))
    k = 0
    colours = generate_xkcd_colours()
    lines = None
    labels = None
    for i in range(m):
        for j in range(n):
            if k >= len(alpha): break
            ax = fig.add_subplot(m,n,k+1)
            ax.hist(samples[:,k],color=next(colours),label=r'$\alpha$'+f'[{alpha[k]}]',bins=10)
            if k == 0:
                lines, labels = ax.get_legend_handles_labels()
            else:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines = lines + lines1
                labels = labels + labels1
            k += 1
            
    ax.legend(lines,labels)
    fig.tight_layout(pad=3,h_pad=4)
    figs_path_name = (Path(figs) / Path(__file__).stem).with_suffix('.png')
    fig.savefig(figs_path_name.with_suffix('.png'))    

def get_rows_and_columns(M):
    m = int(np.sqrt(M))
    n = M // m
    while m*n < M:
        n += 1
    return m,n

def main():
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
 
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    m,n = get_rows_and_columns(len(args.alpha))   
    plot(samples = rng.dirichlet(args.alpha,size=args.N),m=m,n=n,alpha=args.alpha,figs=args.figs)
 
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
        
if __name__=='__main__':
    main()
