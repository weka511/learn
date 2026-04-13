#!/usr/bin/env python

# Copyright (C) 2022-2026 Greenweaves Software Limited

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
    Generate data for the Coordinate Ascent Mean-Field Variational
    Inference (CAVI) example from Section 3 of Blei et al
'''

from argparse import ArgumentParser
from pathlib import Path
from matplotlib.pyplot import figure, rcParams, show
import numpy as np
from utils import generate_xkcd_colours

class GaussionMixtureModel:
    '''
    This class generates data using a Gaussian Mixture Model

    Members:
        mu      Array of means for the clusters. The generated data
                will have the same the same number of dimensions as mu
        sigma   Array of standard deviations for the clusters.
        n       Number of points
        k       Number of clusters
        d       Dimensionalty of space
    '''

    def __init__(self, mu=np.zeros((1)), sigma=np.ones((1)),  n=100):
        try:
            self.k,self.d = mu.shape
        except ValueError:
            self.k = mu.shape[0]
            self.d = 1
        self.mu = mu.copy()
        self.sigma = sigma.copy()
        self.n = n

    def create_data(self):
        '''
        Sample data from specified number of Gaussians
        '''
        self.choice = rng.integers(0, high=self.k, size=self.n)
        samples = rng.standard_normal(size=(self.n,self.d))
        for i in range(self.n):
            samples[i] *= self.sigma[self.choice[i]]
            samples[i] += self.mu[self.choice[i]]
        return samples

    def save(self,path_name):
        '''
        Save generated data and its sufficient statistics

        Parameters:
            path_name   Full path name of file
        '''
        np.savez(path_name,data=self.create_data(),mu=self.mu,sigma=self.sigma,choice=self.choice)

    def load(self,path_name):
        '''
        Load data and its sufficient statistics

        Parameters:
            path_name   Full path name of file
        '''
        with open(path_name, 'rb') as f:
            npzfile = np.load(f)
            self.mu = npzfile['mu']
            self.sigma = npzfile['sigma']
            self.choice = npzfile['choice']
            x = npzfile['data']
            _,d = x.shape
            return x if d > 1 else np.ravel(x)


def get_name(args):
    '''
    Used to establish default file name

    Parameters:
        args     Command line arguments
    '''
    if args.name == None:
        return f'gmm{args.d}-{args.K}'
    else:
        return args.name

def dimensionality(s):
    '''
    Used to verify dimensionality is 1, 2, or 3

    Parameters:
        s         String from command line
    '''
    d = int(s)
    if d in [1,2,3]:
        return d
    else:
        raise ValueError()

def parse_args():
    '''
    Extract parameters from command line
    '''
    K = 3
    n = 1000
    sigma = 1.0
    d = 1
    parser = ArgumentParser(__doc__)
    parser.add_argument('--name', help='Base of name for files')
    parser.add_argument('--K', type=int, default=K, help=f'Number of Gaussians [{K}]')
    parser.add_argument('--n', type=int, default=n, help=f'Number of points [{n}]')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot displayed')
    parser.add_argument('--sigma', type=float, default=sigma, help=f'Standard deviation [{sigma}]')
    parser.add_argument('--data', default='./data', help='Path to folder where data are to be stored')
    parser.add_argument('--figs', default='./figs', help='Location for storing plot files')
    parser.add_argument('--d', type=dimensionality, default=d, help=f'Dimensionality of space [{d}]')
    return parser.parse_args()

def create_colours(K):
    '''
    Create an array containing a specified number of colours

    Parameters:
        K       Number of colours
    '''
    colour_generator = generate_xkcd_colours()
    return np.array([next(colour_generator) for _ in range(K)])

if __name__ == '__main__':
    rcParams.update({'text.usetex': True})

    args = parse_args()
    rng = np.random.default_rng(args.seed)
    shape = args.K if args.d == 1 else (args.K,args.d)
    sigma = args.sigma * np.ones(shape=shape)
    mu = rng.uniform(low=0, high=25, size=shape)
    model = GaussionMixtureModel(mu=mu, sigma=sigma, n=args.n)
    path_name = Path(args.data) / get_name(args)
    model.save(path_name.with_suffix('.npz'))
    print (f'Saved data in {path_name.with_suffix('.npz')}')
    data = model.load(path_name.with_suffix('.npz'))
    
    fig = figure(figsize=(10, 5))
    colours = create_colours(args.K)
    match args.d:
        case 1:
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(data, bins='sturges',density=True,color='xkcd:blue')
            ax.vlines(model.mu, 0, ax.get_ylim()[1], colors='xkcd:red', linestyles='dotted')
        case 2:
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(data[:,0],data[:,1],c=colours[model.choice],s=5)
        case 3:
            ax = fig.add_subplot(1, 1, 1,projection='3d')
            ax.scatter(data[:,0],data[:,1],data[:,2],c=colours[model.choice],s=5)

    ax.set_title(f'{args.d}D Gaussian Mixture Model with {args.K} centres')

    fig.savefig((Path(args.figs) / get_name(args)).with_suffix('.png'))

    if args.show:
        show()
