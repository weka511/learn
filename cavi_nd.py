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
from gmm import GaussionMixtureModel, get_name

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--name')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')
    parser.add_argument('--K', type=int, default=3, help='Number of Gaussians')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot displayed')
    parser.add_argument('--N', type=int, default=250, help='Number of Gaussians')
    parser.add_argument('--M', type=int, default=16, help='Number of attempts')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for improving ELBO')
    parser.add_argument('--sigma', type=float, default=1, help='Standard deviation')
    parser.add_argument('--n', type=int, default=5, help='Burn in period')
    parser.add_argument('--figs', default='./figs', help='Folder to store plots')
    parser.add_argument('--path', default='./data', help='Path to folder where data are stored')
    return parser.parse_args()

def get_ELBO(m,s,c,x):
    '''
    Calculate ELBO using equation (18) of my notes
    '''
    def create_offsets():
        n,d = x.shape
        n,K = c.shape
        offsets = np.zeros((n,K))
        for i in range(n):
            for k in range(K):
                for j in range(d):
                    offsets[i,k] += s[k,j]**2 + (x[i,j] - m[k,j])**2
        return
    return -0.5 * ( np.sum(mu**2 + s**2) + np.sum(c * create_offsets()))

if __name__ == '__main__':
    rcParams.update({
        "text.usetex": True
    })

    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model = GaussionMixtureModel(name=get_name(args))
    x = model.load(path=args.path)
    n,d = x.shape

    mu = model.mu
    s = model.sigma
    epsilon = 0.1
    mu0 = mu + epsilon*rng.standard_normal(mu.shape)

    c = np.zeros((n,args.K))
    for i in range(n):
        diff = np.empty((args.K))
        for k in range(args.K):
            diff[k] = np.sum((x[i,:] - mu[k,:])**2)
        index_closest = np.argmin(diff)
        c[i,index_closest] = 1

    ELBO = get_ELBO(mu0,s,c,x)
    z=0

