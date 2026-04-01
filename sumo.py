#!/usr/bin/env python

# Copyright (C) 2026 Simon Crase  simon@greenweaves.nz

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>.

'''
    Generate test data from a Bradley-Terry model, then try to fit to the data.
    Compare fitted parameters to original. How large does dataset need to be?
'''


from argparse import ArgumentParser
from pathlib import Path
from matplotlib.pyplot import figure,show
import numpy as np
from bt import bt

class Rikishi:
    next_seq = 0
    
    def __init__(self,rikishi_id,rank,shikona):
        self.seq = Rikishi.next_seq
        self.rikishi_id = rikishi_id
        self.rank = rank
        self.shikona = shikona
        Rikishi.next_seq += 1
        
    def __str__(self):
        return f'{self.seq} {self.rikishi_id} {self.rank} {self.shikona}'
    
class Results:
    INDEX = 0
    BASHO = 1
    DAY = 2
    RIKISHI_1_ID = 3
    RIKISHI_1_RANK = 4
    RIKISHI_1_SHIKONA = 5
    RIKISHI_1_RESULT = 6
    RIKISHI_1_WIN = 7
    KIMARITE = 8
    RIKISHI_2_ID = 9
    RIKISHI_2_RANK = 10
    RIKISHI_2_SHIKONA = 11
    RIKISHI_2_RESULT = 12
    RIKISHI_2_WIN = 13    
    
    def __init__(self):
        self.rikishi = {}
        self.rikishi_by_seq = {}
        
    def get_rikishi(self,rikishi_id,rank,shikona):
        if rikishi_id not in self.rikishi:
            rikishi =  Rikishi(rikishi_id,rank,shikona)
            self.rikishi[rikishi_id] = rikishi
            self.rikishi_by_seq[rikishi.seq] =  rikishi
        return self.rikishi[rikishi_id]     
        
    def build(self,path):
        Product = []
        with (open(path)) as file:
            for i,line in enumerate(file):
                if i == 0: continue
                fields = line.strip().split(',')
                basho = int(fields[Results.BASHO].split('.')[1])
                if args.basho != None and basho != args.basho: continue
                rikishi_1 = self.get_rikishi(fields[Results.RIKISHI_1_ID],
                                             fields[Results.RIKISHI_1_RANK],
                                             fields[Results.RIKISHI_1_SHIKONA])
                rikishi_2 = self.get_rikishi(fields[Results.RIKISHI_2_ID],
                                             fields[Results.RIKISHI_2_RANK],
                                             fields[Results.RIKISHI_2_SHIKONA])
                if int(fields[Results.RIKISHI_1_WIN]) == 1 and int(fields[Results.RIKISHI_2_WIN]) == 0:
                    Product.append([rikishi_1.seq,rikishi_2.seq,rikishi_1.seq])
                elif int(fields[Results.RIKISHI_2_WIN]) == 1 and int(fields[Results.RIKISHI_1_WIN]) == 0:
                    Product.append([rikishi_1.seq,rikishi_2.seq,rikishi_2.seq])
                else:
                    raise RuntimeError(str(i))
               
        return np.array(Product)

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--N', type=int, default=200,help='Number of Iterations')
    parser.add_argument('--burn', type=int, default=0,help='Burn in: skip this many iterations when we display')
    parser.add_argument('--data',default = './data/sumo',help='Path to data files')
    parser.add_argument('--year',type=int,default=2019,help='Year to be processed')
    parser.add_argument('--epsilon',type=float,default=1.0e-6,help="Correction for Cromwell's rule")
    parser.add_argument('--cutoff',type=int, default=10,help='Controls how many labels will be displayed in plot')
    parser.add_argument('--basho',type=int, default = None)
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    results = Results()
    Contests = results.build((Path(args.data) / str(args.year)).with_suffix('.csv'))
    scores = bt(Contests,Rikishi.next_seq,args.N,epsilon=args.epsilon,log=True)
    means = np.mean(scores,axis=1)
    indices = np.argsort(scores[-1,:])[::-1][0:args.cutoff]
    fig = figure(figsize=(12,12))
    ax1 = fig.add_subplot(1,1,1)
    for k in range(Rikishi.next_seq):
        ax1.plot(scores[args.burn:,k] - means[args.burn:],
                 linestyle='solid' if k in indices else 'dotted',
                 label=results.rikishi_by_seq[k] if k in indices else None)
    ax1.legend(loc='center left')
    basho_text = 'all Bashos' if args.basho == None else f'Basho {args.basho}'
    ax1.set_title(f'Evolution of Bradley-Terry Parameters for {basho_text} in {args.year}' )
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('P')
    ax1.set_xbound(args.burn,args.N)  
    file_stem = str(args.year) if args.basho == None else f'{args.year}-{args.basho}'
    file_name = (Path(args.figs) / file_stem).with_suffix('.png')
    fig.savefig(file_name)
    print (f'Saved in {file_name}')
    
    if args.show:
        show()    