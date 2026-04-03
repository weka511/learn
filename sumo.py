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
    Fit a Bradley-Terry model to data from Sumo bashos. 
    The data has been downloaxed from
    https://www.kaggle.com/datasets/thedevastator/sumo-wrestling-matches-results-1985-2019
'''
from argparse import ArgumentParser
from pathlib import Path
from matplotlib.pyplot import figure,show
import numpy as np
from bt import bt

class Rikishi:
    '''
    This class represents one Rikishi: name, rank at time of tournament, etc,
    plus the number of wins and number of losses.
    '''
    next_seq = 0
    
    def __init__(self,rikishi_id,rank,shikona):
        self.seq = Rikishi.next_seq
        self.rikishi_id = rikishi_id
        self.rank = rank
        self.shikona = shikona
        Rikishi.next_seq += 1
        self.win = 0
        self.loss = 0
    
    def get_rank(self):
        '''
        Format rank for display"
        '''
        match self.rank[0]:
            case 'Y':
                return f'Yokozuna {self.rank[-1].upper()}'
            case 'O':
                return f'Ozeki {self.rank[-1].upper()}'
            case 'S':
                return f'Sekiwaki {self.rank[-1].upper()}'
            case 'K':
                return f'Komosubi {self.rank[-1].upper()}'
            case 'M':
                try:
                    return f'Maegashira {int(self.rank[1:-1])} {self.rank[-1].upper()}'
                except ValueError:
                    pass
            case 'J':
                try:
                    return f'Juryo {int(self.rank[1:-1])} {self.rank[-1].upper()}'
                except ValueError:
                    pass            
                
        return self.rank
    
    def __str__(self):
        return f'{self.shikona} {self.get_rank()} ({self.win}-{self.loss})'
    
class Results:
    '''
    This class holds the reults of a basho or set of bashos
    '''
    # The index of each field in a data record
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
        '''
        Ensure rikishi is stored uniquely in dataset
        
        Parameters:
            rikishi_id
            rank
            shikona
            
        Returns:
            Unique occurence of rikishi
        '''
        if rikishi_id not in self.rikishi:
            rikishi =  Rikishi(rikishi_id,rank,shikona)
            self.rikishi[rikishi_id] = rikishi
            self.rikishi_by_seq[rikishi.seq] =  rikishi
        return self.rikishi[rikishi_id]     
        
    def build(self,path):
        '''
        Construct results from file
        
        Parameters:
            path      Identified location of file
        '''
        Product = []
        id_1 = None
        id_2 = None
        with (open(path)) as file:
            for i,line in enumerate(file):
                if i == 0: continue
                fields = line.strip().split(',')
                basho = int(fields[Results.BASHO].split('.')[1])
                if args.basho != None and basho != args.basho: continue
                if id_2 == fields[Results.RIKISHI_1_ID] and id_1 == fields[Results.RIKISHI_2_ID]: continue
                id_1 = fields[Results.RIKISHI_1_ID]
                id_2 = fields[Results.RIKISHI_2_ID]
                rikishi_1 = self.get_rikishi(id_1,
                                             fields[Results.RIKISHI_1_RANK],
                                             fields[Results.RIKISHI_1_SHIKONA])
                rikishi_2 = self.get_rikishi(id_2,
                                             fields[Results.RIKISHI_2_RANK],
                                             fields[Results.RIKISHI_2_SHIKONA])
                if int(fields[Results.RIKISHI_1_WIN]) == 1 and int(fields[Results.RIKISHI_2_WIN]) == 0:
                    Product.append([rikishi_1.seq,rikishi_2.seq,rikishi_1.seq])
                    rikishi_1.win += 1
                    rikishi_2.loss += 1
                elif int(fields[Results.RIKISHI_2_WIN]) == 1 and int(fields[Results.RIKISHI_1_WIN]) == 0:
                    Product.append([rikishi_1.seq,rikishi_2.seq,rikishi_2.seq])
                    rikishi_2.win += 1
                    rikishi_1.loss += 1                    
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
    parser.add_argument('--basho',type=int, default = None)
    parser.add_argument('--ranks',default='MKSOY')
    return parser.parse_args()
    
if __name__ == '__main__':
    LEGEND_BREAK = 24          # Number of rikishi per legend
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    results = Results()
    Contests = results.build((Path(args.data) / str(args.year)).with_suffix('.csv'))
    scores = bt(Contests,Rikishi.next_seq,args.N,epsilon=args.epsilon,log=True)
    means = np.mean(scores,axis=1)
    indices = np.argsort(scores[-1,:])[::-1]
    fig = figure(figsize=(12,12))
    ax1 = fig.add_subplot(1,1,1)
    line_plots_for_legend = []
    for k in range(Rikishi.next_seq):
        if results.rikishi_by_seq[k].rank[0] not in args.ranks: # Is this rikishi within 
            continue                                            # the ranks we want to plot?
        line_plot, = ax1.plot(scores[args.burn:,k] - means[args.burn:],
                 linestyle='solid' if k in indices else 'dotted',
                 label=results.rikishi_by_seq[k])
        line_plots_for_legend.append(line_plot)
 
    legend1 = ax1.legend(handles=line_plots_for_legend[:LEGEND_BREAK],loc='lower left')
    ax1.add_artist(legend1)
    ax1.legend(handles=line_plots_for_legend[LEGEND_BREAK:],loc='lower right')
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