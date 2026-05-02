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

'''Generate data using Chinese Restaurant Process'''

from argparse import ArgumentParser
from os.path import splitext,join
from pathlib import Path
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc
import numpy as np
from shared.utils import Logger,generate_xkcd_colours

class Table:
    mu0 = 0
    sigma0 = 1
    colours = generate_xkcd_colours()
    
    def __init__(self,rng=np.random.default_rng()):
        self.rng = rng
        self.mu = rng.normal(loc=Table.mu0,scale=Table.sigma0)
        self.sigma = 0.25*Table.sigma0   #FIXME
        self.indices = []
        self.colour = next(Table.colours)
        
    def __len__(self):
        return len(self.indices)
        
    def get_sample(self):
        return self.rng.normal(loc=self.mu,scale=Table.sigma0)
    
    def append(self,index):
        self.indices.append(index)
    
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out','-o',required=True,help='Name of output file')
    parser.add_argument('--logfiles', default='./logfiles', help='Location of log files')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--seed', default=None, type=int, help='Used to initialize random number generator')
    parser.add_argument('--data', default='./data', help='Location of data files')
    parser.add_argument('--N', type=int, default=1000, help='Number of samples')
    parser.add_argument('--dimensionality', '-d',type=int, default=3, help='Dimensionality')
    parser.add_argument('--alpha', default=0.1,type=float,help='Parameter for CRP')
    parser.add_argument('--mu', default=[0,0,0],type=float,nargs='+',help='Parameter for CRP')
    parser.add_argument('--sigma', default=1,type=float,help='Parameter for CRP')
    
    return parser.parse_args()

def create_weights(tables,alpha):
    p = np.empty((len(tables) + 1))
    for j in range(len(tables)):
        p[j] = len(tables[j]) - 1 + alpha 
    p[-1] = alpha
    return p / p.sum()
 

def main():
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
 
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    
    z = np.zeros((args.N,args.dimensionality))
    Table.mu0 = args.mu
    Table.sigma0 = args.sigma
 
    tables = []
    for i in range(args.N):        
        index = rng.choice(len(tables)+1,
                           p=create_weights(tables,args.alpha))
        if index == len(tables):
            tables.append(Table(rng))
        z[i] = tables[index].get_sample()
        tables[index].append(i)
    
    output_file = (Path(args.data) / args.out).with_suffix('.npz')
    np.savez(output_file,z=z)
    print (f'Saved {args.N} points in {len(tables)} clusters to {output_file}')
    
    fig = figure(figsize=(8, 8))
    ax1 = fig.add_subplot(2,1,1)
    ax1.bar(range(len(tables)),[len(table) for table in tables],
            label=[f'{i}' for i in range(len(tables))],
            color=[table.colour for table in tables])
    ax1.set_xlabel('Tables')
    ax1.set_ylabel('Number')
    ax1.set_title(r'$\alpha=$'+f'{args.alpha}')
    ax1.legend(title='Clusters',ncols=int(np.sqrt(len(tables))))
    match args.dimensionality:
        case 1:
            pass
        case 2:
            ax2 = fig.add_subplot(2,1,2)
            for i,table in enumerate(tables):
                indices = table.indices
                ax2.scatter(z[indices,0],z[indices,1],
                            c=table.colour,label=f'{i}')
 
        case 3:
            pass
        
    fig.tight_layout(pad=3,h_pad=4)
    figs_path_name = (Path(args.figs) / Path(__file__).stem).with_suffix('.png')
    fig.savefig(figs_path_name.with_suffix('.png'))    
    
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()
        
if __name__=='__main__':
    main()
