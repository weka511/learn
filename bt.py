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
from scipy.stats import linregress

def generate_parameters(m,rng = np.random.default_rng()):
    '''
    Generate a set of Bradley Terry parameters
    
    Parameters:
        m       Number of parameters
        rng
        
    Returns:
        Bradlet Terry Parameters
    '''
    return rng.uniform(size=m)

def create_contests(n,P,rng = np.random.default_rng()):
    '''
    Create test data
    
    Parameters:
         n    Number of rounds
         P    The parameters from the Bradley-Terry model
         rng 
         
        Returns:
            An array of contests between two players,
            accompanied by the outcome of each contest   
    '''
    m = len(P)
    Pairwise = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            if i != j:
                Pairwise[i,j] = P[i]/(P[i] + P[j])
    
    Product = np.zeros((n,(m//2),3),dtype=int)
    for k in range(n):
        Product[k,:,0:2] = rng.permutation(m).reshape(m//2,2)
    Product = Product.reshape(-1, Product.shape[-1])
    for k in range(n *(m//2)):
        i = Product[k,0] 
        j = Product[k,1]         
        assert i != j 
        Product[k,2] = i if rng.uniform() < Pairwise[i,j] else j
 
    return Product
    
def bt(Contests,m,N):
    '''
    Calculate Bradley-Terry parameters from a set of contests.
    
    Parameters:
        Contests   Represents the contests between players: the first two columns
                   are the two players, and the third contains the winners
        m          Number of Players
        N          Number of Iterations
        
    Returns:
       Fitted set of parameters
    '''
    def createW():
        W = np.zeros((m,m),dtype=int)
        n,_ = Contests.shape
        for k in range(n):
            i = Contests[k,0]
            j = Contests[k,1]
            if Contests[k,2] == i:     # i wins
                W[i,j] += 1
            elif Contests[k,2] == j:   # j wins
                W[j,i] += 1
            else:
                raise RuntimeError(f'k={k}, i={i}, j={j}, Outcome mismatch: {Contests[k,2]}')
            
        return W,np.sum(W,axis=1)
    
    W,w = createW()
    p = np.zeros((N+1,m))
    p[0,:] = rng.uniform(size=m)
    
    for k in range(N):
        for i in range(m):                  
            p[k+1,i] = w[i]/sum((W[i,j] + W[j,i])/(p[k,i] + p[k,j]) for j in range(m) if i != j)
    return p
                  

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be displayed')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--m', type=int, default=42,help='Number of Players')
    parser.add_argument('--n', type=int, default=100,help='Number of rounds')
    parser.add_argument('--N', type=int, default=50,help='Number of Iterations')
    parser.add_argument('--burn', type=int, default=0,help='Burn in: skip this many iterations when we display')
    parser.add_argument('out',  help='Name of plot file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    P = generate_parameters(args.m,rng=rng)
    p = bt(create_contests(args.n,P,rng=rng),args.m,args.N)
    slope, intercept, r, pvalue, se = linregress(P,p[-1,:])
    P_sorted = np.sort(P)
    
    fig = figure(figsize=(12,12))
    ax1 = fig.add_subplot(1,2,1)
    for k in range(args.m):
        ax1.plot(p[args.burn:,k])
    ax1.set_title('Evolution of Bradley-Terry Parameters')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('P')
    ax1.set_xbound(args.burn,args.N)

    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(P,p[-1,:],c='xkcd:blue',label='Calculated')
    ax2.plot(P_sorted,slope*P_sorted+intercept,c='xkcd:red',label=f'Slope={slope:.3f}, intercept={intercept:.3e}')
    ax2.set_title(f'r={r:.3f}, pvalue={pvalue:.3e}, se={se:.3e}')
    ax2.set_xlabel('Ground Truth')
    ax2.set_ylabel('Calculated')
    ax2.legend()
    
    fig.suptitle(f'Number of Players={args.m:,}, Number of Rounds={args.n:,}, Number of Iterations={args.N:,}')
    fig.tight_layout(pad=2,h_pad=2,w_pad=2)
    file_name = (Path(args.figs) / args.out).with_suffix('.png')
    fig.savefig(file_name)
    print (f'Saved in {file_name}')
    
    if args.show:
        show()