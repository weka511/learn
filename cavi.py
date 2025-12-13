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
'''

from argparse import ArgumentParser
from os.path import basename, join
from matplotlib.pyplot import figure, rcParams, show
import numpy as np
from xkcd import generate_xkcd_colours
from gmm import GaussionMixtureModel, get_name


class Cavi:
    '''
    Perform Coordinate Ascent Mean-Field Variational Inference

    Parameters:
        K               Number of Gaussians to be fitted
        max_iterations  Maximum number of iterations -- if limit exceeded.
                        we deem cavi to have failed to converge
        atol            For assessing convergence
        sigma
        min_iterations  Minimum number of iterations -- don't check for convergence
                        until we have at least this many iterations

    I have borrowed some ideas from Zhiya Zuo's blog--https://zhiyzuo.github.io/VI/
    '''

    def __init__(self, K=3):
        self.K = K

    def infer_hidden_parameters(self, x, rng=np.random.default_rng(),
                                max_iterations=100,
                                atol=1e-6,
                                sigma=1,
                                min_iterations=5):
        m = self.init_means(x, rng)
        s2 = rng.random(self.K)    # Variance of target q(...)
        ELBOs = []
        while (True):
            # Blei et al, equation (26)
            e_mu = np.outer(x, m)                    # Expectation of mu - n x K
            e_mu2 = -0.5 * (m**2 + s2)               # Expectation of mu*mu - K x 1
            phi = np.exp(e_mu + e_mu2[np.newaxis, :])  # Unnormalized - n x K
            phi = phi / phi.sum(1)[:, np.newaxis]

            # Blei et al, equation (34)
            s2 = 1 / (1 / sigma**2 + phi.sum(0))
            m = (phi * x[:, np.newaxis]).sum(0) * s2

            ELBOs.append(self.getELBO(s2, m, sigma, x, phi))

            if len(ELBOs) > min_iterations and abs(ELBOs[-1] / ELBOs[-2] - 1) < atol:
                return Solution(ELBOs, phi, m, np.sqrt(s2))
            if len(ELBOs) > max_iterations:
                raise ELBO_Error(f'ELBO has not converged to within {atol} after {max_iterations} iterations', ELBOs)

    def init_means(self, x, rng=np.random.default_rng()):
        '''
        Initialize means to be roughly the 'K' quantiles, plus random noise
        '''
        quantiles = [np.quantile(x, i / self.K) for i in range(1, self.K + 1)]
        epsilon = min([a - b for (a, b) in zip(quantiles[1:], quantiles[:-1])]) / 6
        return np.array([q * rng.normal(loc=1.0, scale=epsilon) for q in quantiles])

    def getELBO(self, s2, m, sigma, x, phi):
        '''
        Calculate ELBO following Blei et al, equation (21)
        '''
        def get_sum_kK():   # First term in (21) -- sum k in 1:K
            return (np.log(s2) - m / sigma**2).sum()

        def get_sum_iN():   # remaining terms == sum i in 1:N
            t2 = -0.5 * np.add.outer(x**2, s2 + m**2)     # q?
            t2 += np.outer(x, m)
            t2 -= np.log(phi)                          # q?
            t2 *= phi
            return t2.sum()
        return get_sum_kK() + get_sum_iN()


class Solution:
    def __init__(self, ELBOs, phi, m, s):
        self.ELBOs = ELBOs
        self.phi = phi
        self.m = m
        self.s = s


class ELBO_Error(Exception):
    '''
    This class allows us to package the ELBO with an exception, e.g. for plotting
    '''

    def __init__(self, message, ELBOs):
        super().__init__(message)
        self.ELBOs = ELBOs


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
    return parser.parse_args()


if __name__ == '__main__':
    rcParams.update({
        "text.usetex": True
    })

    args = parse_args()

    model = GaussionMixtureModel(name=get_name(args))
    x = model.load()
    cavi = Cavi(K=args.K)

    Solutions = []
    Failures = []
    for i in range(args.M):
        try:
            solution = cavi.infer_hidden_parameters(x,
                                                max_iterations=args.N,
                                                atol=args.tol,
                                                min_iterations=args.n,
                                                sigma=args.sigma,
                                                rng=np.random.default_rng(args.seed))
            Solutions.append(solution)
        except ELBO_Error as e:
            print(e)
            Failures.append(e.ELBOs)

    i_best = np.argmax([solution.ELBOs[-1] for solution in Solutions])

    nrows = 2 if len(Failures) == 0 else 3
    fig = figure(figsize=(10, 10))

    ax1 = fig.add_subplot(nrows, 1, 1)
    n, _, _ = ax1.hist(x, bins='sturges', color='xkcd:blue', label='x')
    ax1.vlines(model.mu, 0, max(n), colors='xkcd:red', label='Means (generated)')
    ax1.vlines(Solutions[i_best].m, 0, max(n), colors='xkcd:green', linestyles='dashed', label='Means (fitted)')
    ax1.set_title(f'Gaussian Mixture Model')
    ax1.legend()

    colours = generate_xkcd_colours(filter=lambda R, G, B: R < 192 and max(R, G, B) > 32)
    best_colour = next(colours)
    ax2 = fig.add_subplot(nrows, 1, 2)
    for i in range(len(Solutions)):
        if i == i_best:
            # label='Best'
            linestyle = 'solid'
            linewidth = 3
            colour = best_colour
        else:
            # label = None
            linestyle = 'dotted'
            linewidth = 1.5
            colour = next(colours)
        ax2.plot(Solutions[i].ELBOs,label=str(i),linestyle=linestyle,linewidth=linewidth,c=colour)

    ax2.set_ylabel('ELBO')
    ax2.legend(title=f'Maximum ELBO {i_best}: {len(Solutions[i_best].ELBOs)} iterations, atol={args.tol}',
               ncol=int(np.sqrt(len(Solutions))))

    if len(Failures) > 0:
        ax3 = fig.add_subplot(nrows, 1, 3)
        for i in range(len(Failures)):
            ax3.plot(Failures[i])
        ax3.set_ylabel('ELBO (Failed)')

    fig.savefig(join(args.figs, f'{basename(__file__).split('.')[0]}'))

    if args.show:
        show()
