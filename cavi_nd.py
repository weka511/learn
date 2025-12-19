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

	N dimensions
'''

from argparse import ArgumentParser
from os.path import basename, join
from matplotlib.pyplot import figure, rcParams, show
import numpy as np
from xkcd import generate_xkcd_colours
from gmm import GaussionMixtureModel, get_name, create_colours

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--name')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')
    parser.add_argument('--K', type=int, default=3, help='Number of Gaussians')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot displayed')
    parser.add_argument('--N', type=int, default=250, help='Number of iterations')
    parser.add_argument('--M', type=int, default=16, help='Number of attempts')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for improving ELBO')
    parser.add_argument('--sigma', type=float, default=1, help='Standard deviation')
    parser.add_argument('--n', type=int, default=5, help='Burn in period')
    parser.add_argument('--figs', default='./figs', help='Folder to store plots')
    parser.add_argument('--path', default='./data', help='Path to folder where data are stored')
    return parser.parse_args()

def initialize(x,model,K):
    n,d = x.shape

    mu = model.mu
    s = model.sigma
    epsilon = 0.1
    mu0 = mu + epsilon*rng.standard_normal(mu.shape)

    c = np.zeros((n,K))
    for i in range(n):
        # diff = np.empty((K))
        # for k in range(K):
            # diff[k] = np.sum((x[i,:] - mu[k,:])**2)
        # index_closest = np.argmin(diff)
        index_closest = rng.integers(0,2)
        c[i,index_closest] = 1
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

def display(l,x,c,fig,m=2,n=2):
    ax = fig.add_subplot(m,n,l)
    n,d = x.shape
    x_colours = np.empty((n),dtype=np.dtypes.StringDType())
    for i in range(n):
        index = np.argmax(c[i,:])
        x_colours[i] = colours[index]
    ax.scatter(x[:,0],x[:,1],c=x_colours,s=1)

if __name__ == '__main__':
    rcParams.update({
        "text.usetex": True
    })

    args = parse_args()
    rng = np.random.default_rng(args.seed)
    fig = figure(figsize=(10, 5))
    colours = create_colours(args.K)

    model = GaussionMixtureModel(name=get_name(args))
    x = model.load(path=args.path)
    m,s,c = initialize(x,model,args.K)
    display(1,x,c,fig)

    ELBO = get_ELBO(m,s,c,x)
    print (ELBO)
    for i in range(3):#args.N):
        display(i+2,x,c,fig)
        c = get_updated_assignments(m,s,x)
        m,s = get_updated_statistics(m,s,c,x)
        ELBO = get_ELBO(m,s,c,x)
        print (ELBO)

    if args.show:
        show()

