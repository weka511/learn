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
from time import time
from matplotlib.pyplot import figure, rcParams, show
import numpy as np
from xkcd import generate_xkcd_colours
from gmm import GaussionMixtureModel, get_name, create_colours


class Solution:
    '''
    This class represents the results of one run of CAVI.

    Parameters:
        ELBO    History of ELBO throughout run
        m       Means
        s       Standard deviations
        c       Assigments of points to clusters
    '''
    def __init__(self):
        '''
        Used at the start of a run to initalize parameters
        '''
        self.ELBO = []
        self.m = None
        self.s = None
        self.c = None

    def set_params(self, m, s, c):
        '''
        Used at the end of a run to store results

        Parameters:
            m       Means
            s       Standard deviations
            c       Assigments of points to clusters
        '''
        self.m = m.copy()
        self.s = s.copy()
        self.c = c.copy()

    def append_ELBO(self, ELBO):
        '''
        Used during a run to accumulate ELBO for each iteration

        Parameters:
            ELBO    Current value, to be appended to history
        '''
        self.ELBO.append(ELBO)


def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('name',help='Name of data file')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')
    parser.add_argument('--K', type=int, default=3, help='Number of Gaussians')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot displayed')
    parser.add_argument('--N', type=int, default=100, help='Number of iterations per run')
    parser.add_argument('--M', type=int, default=25, help='Number of runs')
    parser.add_argument('--BURN_IN', type=int, default=3, help='Minimum number of iterations')
    parser.add_argument('--atol', type=float, default=1e-6, help='Tolerance for improving ELBO')
    parser.add_argument('--sigma', type=float, default=1, help='Standard deviation')
    parser.add_argument('--figs', default='./figs', help='Folder to store plots')
    parser.add_argument('--path', default='./data', help='Path to folder where data are stored')
    return parser.parse_args()


def initialize(x, K, rng=np.random.default_rng()):
    '''
    Choose some points at random to be intial values of means.

    Parameters:
        x          Positions of points
        K          Number of clusters
        rng        Randome number generator

    Returns:
        mu        K points, chosen at random to represent cluster centres
        s         Matrix of initial values for cluster centres
        c         Matrix of random assignments of points to cluster centres
    '''
    n, d = x.shape
    s = np.ones((K, d))
    indices = rng.integers(0, n, size=K)
    mu = np.empty((K, d))
    for k in range(K):
        mu[k, :] = x[indices[k], :]

    c = np.zeros((n, K))
    for i in range(n):
        c[i, rng.integers(0, 2)] = 1

    return mu, s, c


def get_ELBO(m, s, c, x):
    '''
    Calculate Evidence Lower Bound [ELBO] using equation (18) of my notes

    Parameters:
        m       Cluster centres
        s       Standard deviations
        c       Assignment of points to clusters
        x       Coordinates of points

    Returns:
        Calculated ELBO
    '''
    def create_offsets():
        '''
        Calculate 2nd term of equation (18)
        '''
        n, d = x.shape
        _, K = c.shape
        offsets = np.zeros((n, K))
        for i in range(n):
            for k in range(K):
                for j in range(d):
                    offsets[i, k] += s[k, j]**2 + (x[i, j] - m[k, j])**2
        return offsets
    return -0.5 * (np.sum(m**2 + s**2) + np.sum(c * create_offsets()))


def get_updated_assignments(m, s, x):
    '''
    Calculate new assignments of points to clusters from equation (26) of Blei et al

    Parameters:
        m       Cluster centres
        s       Standard deviations
        x       Coordinates of points

    Returns:
        c       Assignment of points to clusters
    '''
    n, d = x.shape
    K, _ = m.shape
    log_c = np.zeros((n, K))
    for i in range(n):
        for k in range(K):
            for j in range(d):
                log_c[i, k] += m[k, j] * x[i, j] - 0.5 * (s[k, j]**2 + m[k, j]**2)
    c = np.exp(log_c)
    Z = np.sum(c, axis=1)
    for i in range(n):
        c[i, :] /= Z[i]
    return c


def get_updated_statistics(m0, s0, c, x, sigma=1):
    '''
    Calculate new centres and standard deviations, following equation (34) of Blei et al

    Parameters:
        m0      Cluster centres
        s0      Standard deviations
        c       Assignments of points to clusters
        x       Coordinates of points
        sigma   Hyperparameter

    Returns:
        m      New cluster centres
        s      New standard deviations

    '''
    n, d = x.shape
    K, _ = m0.shape

    denominator = 1 / sigma**2 * np.ones((K))
    for i in range(n):
        for k in range(K):
            denominator[k] += c[i, k]

    m = np.zeros_like(m0)
    for i in range(n):
        for k in range(K):
            for j in range(d):
                m[k, j] += c[i, k] * x[i, j]
    for j in range(d):
        m[:, j] /= denominator

    s = np.empty_like(s0)
    for j in range(d):
        s[:, j] = np.sqrt(1 / denominator)

    return m, s

def create_data_colours(x, c,cluster_colours):
    '''
    Create a set of colours for use when we display points in their clusters

    Parameters:
        x                  Data points
        c                  Assignments of points to clusters
        cluster_colours    Identifies colours assigned to each cluster

    Returns:
        An array of colurs, one for each point, identifying cluster
    '''
    n, d = x.shape
    colours = np.empty((n), dtype=np.dtypes.StringDType())
    for i in range(n):
        index = np.argmax(c[i, :])
        colours[i] = cluster_colours[index]
    return colours


if __name__ == '__main__':
    start  = time()
    rcParams.update({
        "text.usetex": True
    })

    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ELBO_colours = generate_xkcd_colours()
    model = GaussionMixtureModel(name=get_name(args))
    x = model.load(path=args.path)
    Solutions = []
    index_best = -1

    for i in range(args.M):
        print(f'{i+1}/{args.M}')
        m, s, c = initialize(x, args.K, rng=rng)
        Solutions.append(Solution())
        Solutions[-1].append_ELBO(get_ELBO(m, s, c, x))

        for j in range(args.N):
            c = get_updated_assignments(m, s, x)
            m, s = get_updated_statistics(m, s, c, x)
            Solutions[-1].append_ELBO(get_ELBO(m, s, c, x))
            if len(Solutions) > args.BURN_IN and Solutions[-1].ELBO[-1] - Solutions[-1].ELBO[-2] < args.atol:
                break

        Solutions[-1].set_params(m, s, c)
        if index_best == -1 or Solutions[-1].ELBO[-1] > Solutions[index_best].ELBO[-1]:
            index_best = i

    fig = figure(figsize=(12, 12))
    fig.suptitle(f'{args.name}')
    ax1 = fig.add_subplot(2, 1, 1)
    for i in range(args.M):
        label = None
        linestyle = 'dotted'
        if i == index_best:
            label = f'best {Solutions[i].ELBO[-1]:.6}'
            linestyle = 'solid'
        ax1.plot(range(len(Solutions[i].ELBO)),Solutions[i].ELBO, c=next(ELBO_colours), label=label, linestyle=linestyle)
    ax1.legend()
    ax1.set_title(f'ELBO for {args.M} runs')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('ELBO')

    _,d = x.shape
    match d:
        case 1:
            ax2 = fig.add_subplot(2, 1, 2)
            n, _, _ = ax2.hist(x, bins='sturges', color='xkcd:blue', label='x',density=True)
            ax2.vlines(np.ravel(Solutions[index_best].m), 0, max(n), colors='xkcd:red', linestyles='dashed', label='Means (fitted)')
            ax2.set_xlabel('X')
            ax2.set_ylabel('p')
            ax2.legend()

        case 2:
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.scatter(x[:, 0], x[:, 1], c=create_data_colours(x, Solutions[index_best].c, create_colours(args.K)), s=1)
            for k in range(args.K):
                ax2.scatter(Solutions[index_best].m[k, 0], Solutions[index_best].m[k, 1], c='xkcd:black', marker='+', s=25)
            ax2.set_title(f'Solution with best ELBO: {Solutions[index_best].ELBO[-1]:.6} after {args.M} runs')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')

        case 3:
            ax2 = fig.add_subplot(2, 1, 2,projection='3d')
            ax2.scatter(x[:,0],x[:,1],x[:,2],c=create_data_colours(x, Solutions[index_best].c, create_colours(args.K)), s=1)
            ax2.set_title(f'Solution with best ELBO: {Solutions[index_best].ELBO[-1]:.6} after {args.M} runs')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')

    fig.tight_layout(pad=3,h_pad=4)
    fig.savefig(join(args.figs, f'{basename(__file__).split('.')[0]}'))

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    if args.show:
        show()

