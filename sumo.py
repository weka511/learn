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

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--N', type=int, default=50,help='Number of Iterations')
    parser.add_argument('--burn', type=int, default=0,help='Burn in: skip this many iterations when we display')
    parser.add_argument('out',  help='Name of plot file')
    parser.add_argument('--data',default = './data/sumo')
    parser.add_argument('--year',type=int,default=2019)
    parser.add_argument('--epsilon',type=float,default=1.0e-6)
    return parser.parse_args()

class Rikishi:
    next_seq = 0
    def __init__(self,rikishi_id,rank,shikona):
        self.seq = Rikishi.next_seq
        self.rikishi_id = rikishi_id
        self.rank = rank
        self.shikona = shikona
        Rikishi.next_seq += 1
    
class Results:
    def __init__(self):
        self.rishiki = {}
        
    def get_rishiki(self,rikishi_id,rank,shikona):
        if rikishi_id not in self.rishiki:
            self.rishiki[rikishi_id] = Rikishi(rikishi_id,rank,shikona)
        return self.rishiki[rikishi_id]     
        
# 0      1      2  3           4             5                6               7             8      9
# index,basho,day,rikishi1_id,rikishi1_rank,rikishi1_shikona,rikishi1_result,rikishi1_win,kimarite,rikishi2_id
#      10            11              12            13
# ,rikishi2_rank,rikishi2_shikona,rikishi2_result,rikishi2_win
    def build(self,path):
        Product = []
        with (open(path)) as file:
            for i,line in enumerate(file):
                if i == 0: continue
                fields = line.strip().split(',')
                basho = fields[0]
                rikishi_1 = self.get_rishiki(fields[3],fields[4],fields[5])
                rikishi_2 = self.get_rishiki(fields[9],fields[10],fields[11])
                if int(fields[7]) == 1 and int(fields[13]) == 0:
                    Product.append([rikishi_1.seq,rikishi_2.seq,rikishi_1.seq])
                elif int(fields[13]) == 1 and int(fields[7]) == 0:
                    Product.append([rikishi_1.seq,rikishi_2.seq,rikishi_2.seq])
                else:
                    raise RuntimeError(str(i))
        return np.array(Product)

if __name__ == '__main__':
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    results = Results()
    Contests = results.build((Path(args.data) / str(args.year)).with_suffix('.csv'))
    scores = bt(Contests,Rikishi.next_seq,args.N,epsilon=0.001)
    
    fig = figure(figsize=(12,12))
    ax1 = fig.add_subplot(1,1,1)
    for k in range(Rikishi.next_seq):
        ax1.plot(scores[args.burn:,k])
    ax1.set_title('Evolution of Bradley-Terry Parameters')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('P')
    ax1.set_xbound(args.burn,args.N)  
    
    file_name = (Path(args.figs) / args.out).with_suffix('.png')
    fig.savefig(file_name)
    print (f'Saved in {file_name}')
    
    if args.show:
        show()    