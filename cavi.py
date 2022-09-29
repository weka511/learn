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
from gmm               import GaussionMixtureModel
from matplotlib.pyplot import figure, rcParams, show
from numpy.random      import default_rng

class Cavi:
    '''
    Perform Coordinate Ascent Mean-Field Variational Inference

    Parameters:
        K               Number of Gaussians to be fitted
        max_iterations  Maximum number of iterations -- if limit exceeded.
                        we deem cavi to have failed to converge
        tolerance       For assessing convergence
        sigma
        min_iterations  Minimum number of iterations -- don't check for convergence
                        until we have at least this many iterations
    I have borrowed some ideas from Zhiya Zuo's blog--https://zhiyzuo.github.io/VI/
    '''
    def __init__(self,
                 K              = 3,
                 max_iterations = 25,
                 tolerance      = 1e-12,
                 sigma          = 1,
                 min_iterations = 5):
        pass


def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--seed',  type=int,   default=None,                        help='Seed for random number generator')
    parser.add_argument('--show',              default=False,  action='store_true', help='Controls whether plot displayed')

    return parser.parse_args()

if __name__=='__main__':
    rcParams.update({
        "text.usetex": True
    })

    args  = parse_args()
    rng   = default_rng(args.seed)
    model = GaussionMixtureModel()

    data = model.load()
    fig  = figure(figsize = (10,5))
    ax   = fig.add_subplot(1,1,1)
    ax.hist(data,bins=100)
    ax.set_title(f'Gaussian Mixture Model')
    fig.savefig('gmm')
    if args.show:
        show()
