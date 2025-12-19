#!/usr/bin/env python

# Copyright (C) 2022-2025 Greenweaves Software Limited

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
from os.path import basename, join
from matplotlib.pyplot import figure, rcParams, show
import numpy as np
from xkcd import generate_xkcd_colours
from gmm import GaussionMixtureModel, get_name, create_colours,generate_xkcd_colours

class Solution:
    def __init__(self):
        self.ELBO = []

    def set_params(self,m,s,c):
        self.m = m.copy()
        self.s = s.copy()
        self.c = c.copy()

    def append_ELBO(self,ELBO):
        self.ELBO.append(ELBO)

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--name')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')
    parser.add_argument('--K', type=int, default=3, help='Number of Gaussians')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot displayed')
    parser.add_argument('--N', type=int, default=100, help='Number of iterations per run')
    parser.add_argument('--M', type=int, default=25, help='Number of runs')
    parser.add_argument('--BURN_IN',type=int, default=3,help='Minimum number of iterations')
    parser.add_argument('--atol', type=float, default=1e-6, help='Tolerance for improving ELBO')
    parser.add_argument('--sigma', type=float, default=1, help='Standard deviation')
    parser.add_argument('--figs', default='./figs', help='Folder to store plots')
    parser.add_argument('--path', default='./data', help='Path to folder where data are stored')
    return parser.parse_args()

def initialize(x,K, rng = np.random.default_rng()):
    n,d = x.shape
    s = np.ones((K,d))
    indices = rng.integers(0,n,size=K)
    mu0 = np.empty((K,d))
    for k in range(K):
        mu0[k,:] = x[indices[k],:]

    c = np.zeros((n,K))
    for i in range(n):
        c[i, rng.integers(0,2)] = 1

    return mu0,s,c

def get_ELBO(m,s,c,x):
    '''
    Calculate ELBO using equation (18) of my notes
    '''
    def create_offsets():
        n,d = x.shape
        _,K = c.shape
        offsets = np.zeros((n,K))
        for i in range(n):
            for k in range(K):
                for j in range(d):
                    offsets[i,k] += s[k,j]**2 + (x[i,j] - m[k,j])**2
        return offsets
    return -0.5 * ( np.sum(m**2 + s**2) + np.sum(c * create_offsets()))

def get_updated_assignments(m,s,x):
    n,d = x.shape
    K,_ = m.shape
    log_c = np.zeros((n,K))
    for i in range(n):
        for k in range(K):
            for j in range(d):
                log_c[i,k] += m[k,j]*x[i,j] - 0.5 *(s[k,j]**2 + m[k,j]**2)
    c = np.exp(log_c)
    Z = np.sum(c,axis=1)
    for i in range(n):
        c[i,:] /= Z[i]
    return c

def get_updated_statistics(m0,s0,c,x,sigma=1):
    n,d = x.shape
    K,_ = m0.shape
    denominator = 1/sigma**2 * np.ones((K))
    for i in range(n):
        for k in range(K):
            denominator[k] += c[i,k]
    m = np.zeros_like(m0)
    for i in range(n):
        for k in range(K):
            for j in range(d):
                m[k,j] += c[i,k] * x[i,j]
    for j in range(d):
        m[:,j] /= denominator
    s = np.empty_like(s0)
    for j in range(d):
        s[:,j] = np.sqrt(1/denominator)
    return m,s

def display(l,x,c,fig,rows=10,cols=10):
    ax = fig.add_subplot(rows,cols,l)
    n,d = x.shape
    x_colours = np.empty((n),dtype=np.dtypes.StringDType())
    for i in range(n):
        index = np.argmax(c[i,:])
        x_colours[i] = colours[index]
    ax.scatter(x[:,0],x[:,1],c=x_colours,s=1)
    return rows*cols

if __name__ == '__main__':
    rcParams.update({
        "text.usetex": True
    })

    args = parse_args()
    rng = np.random.default_rng(args.seed)

    fig = figure(figsize=(12, 12))

    cluster_colours = create_colours(args.K)
    ELBO_colours = generate_xkcd_colours()
    model = GaussionMixtureModel(name=get_name(args))
    x = model.load(path=args.path)
    Solutions = []
    index_best = -1

    for i in range(args.M):
        print (i)
        m,s,c = initialize(x,args.K,rng=rng)
        Solutions.append(Solution())
        Solutions[-1].append_ELBO( get_ELBO(m,s,c,x))

        for j in range(args.N):
            c = get_updated_assignments(m,s,x)
            m,s = get_updated_statistics(m,s,c,x)
            Solutions[-1].append_ELBO(get_ELBO(m,s,c,x))
            if len(Solutions) > args.BURN_IN and Solutions[-1].ELBO[-1] - Solutions[-1].ELBO[-2] < args.atol: break

        Solutions[-1].set_params(m,s,c)
        if index_best == -1 or Solutions[-1].ELBO[-1] < Solutions[index_best].ELBO[-1]:
            index_best = i

    ax1 = fig.add_subplot(2,1,1)
    for i in range(args.M):
        ax1.plot(Solutions[i].ELBO,c=next(ELBO_colours),label =f'best {Solutions[i].ELBO[-1]:.6}' if i==index_best else None)
    ax1.legend()

    ax2 = fig.add_subplot(2,1,2)
    n,d = x.shape
    x_colours = np.empty((n),dtype=np.dtypes.StringDType())
    for i in range(n):
        index = np.argmax(c[i,:])
        x_colours[i] = cluster_colours[index]
    ax2.scatter(x[:,0],x[:,1],c=x_colours,s=1)
    for k in range(args.K):
        ax2.scatter(Solutions[index_best].m[k,0],Solutions[index_best].m[k,1],c=cluster_colours[k])

    if args.show:
        show()

