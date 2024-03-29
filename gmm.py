#!/usr/bin/env python

# Copyright (C) 2022 Greenweaves Software Limited
#
# Simon A. Crase -- simon@greenweaves.nz of +64 210 220 2257

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

'''Generate data for Gaussion Mixture Model'''

from argparse          import ArgumentParser
from matplotlib.pyplot import figure, rcParams, show
from numpy             import array, load, savez, zeros
from numpy.random      import default_rng

class GaussionMixtureModel:
    '''This class generates data using a Gaussian Mixture Model'''
    def __init__(self,
                 mu    = array([0]),
                 sigma = array([1]),
                 name  = 'gmm',
                 n     = 100):
        k           = mu.shape[0]
        self.mu     = mu.copy()
        self.sigma  = sigma.copy()
        self.name   = name
        self.size   = (n,k)

    def save(self,
             rng    = default_rng()):
        n,k    = self.size
        self.choice = rng.integers(0, high = k, size = n)
        savez(self.name,
             data = self.mu[self.choice] + self.sigma[self.choice]* rng.standard_normal(size=(n)),
             mu     = self.mu,
             sigma  = self.sigma,
             choice = self.choice)

    def load(self):
        with open(f'{self.name}.npz', 'rb') as f:
            npzfile     = load(f)
            self.mu     = npzfile['mu']
            self.sigma  = npzfile['sigma']
            self.choice = npzfile['choice']
            return npzfile ['data']

def get_name(args):
    '''Used to establish default file name'''
    if args.name==None:
        return f'gmm{args.K}'
    else:
        return args.name

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--name',                                                    help='Base of name for files')
    parser.add_argument('--K',     type=int,   default=3,                           help='Number of Gaussians')
    parser.add_argument('--n',     type=int,   default=1000,                        help='Number of points')
    parser.add_argument('--seed',  type=int,   default=None,                        help='Seed for random number generator')
    parser.add_argument('--show',              default=False,  action='store_true', help='Controls whether plot displayed')
    parser.add_argument('--sigma', type=float, default = 1.0,                       help='Standard deviation')
    return parser.parse_args()

if __name__=='__main__':
    rcParams.update({
        "text.usetex": True
    })

    args  = parse_args()
    rng   = default_rng(args.seed)
    sigma = array([args.sigma] * args.K)
    mu    = array(sorted(rng.uniform(low  = 0,
                                     high = 25,
                                     size = args.K)))

    model = GaussionMixtureModel(name  = get_name(args),
                                 mu    = mu,
                                 sigma = sigma,
                                 n     = args.n)
    model.save(rng = rng)
    data  = model.load()
    fig   = figure(figsize = (10,5))
    ax    = fig.add_subplot(1,1,1)
    n,_,_ = ax.hist(data,bins='sturges')
    ax.vlines(model.mu,0,ax.get_ylim()[1],colors='xkcd:red',linestyles='dotted')
    ax.set_title(f'Gaussian Mixture Model with {args.K} centres')
    fig.savefig(get_name(args))
    if args.show:
        show()
