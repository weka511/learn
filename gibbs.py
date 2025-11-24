#!/usr/bin/env python

#   Copyright (C) 2020-2024 Simon Crase

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''
Gibbs sampler for the change-point model described in a Cognition cheat sheet titled "Gibbs sampling."

This is a Python implementation of the procedure at https://sites.math.rutgers.edu/~zeilberg/EM20/GibbsYildirim.pdf
Written by Ilker Yildirim, September 2012.
'''

from argparse import ArgumentParser
from os.path import basename, join
from time import time
from matplotlib.pyplot import figure, show
from matplotlib import rc
import numpy as np
from numpy.random import default_rng
from pymdp.maths import softmax


def generate(rng, N=50, a=2, b=1, K=5):
    '''
    Generate a sequence of counts. The first n counts follow a Poisson distribution with mean lambda1, and the remainder
    follow Poisson with mean lambda2. The two lambda are sampled from a Gamma distribution, and n is also random.

    Parameters:
        rng     Random number generator
        N       Number of observations
        a       loc parameter for gamma distribution
        b       1/scale parameter for gamma distribution

    Returns:
        x       Observations
        lambdas Intensity values
    '''
    for k in range(K):
        try:
            n = int(round(rng.uniform() * N)) # Change-point: where the intensity parameter changes.
            lambda1 = rng.gamma(a, scale=1. / b) # Intensity value
            lambda2 = rng.gamma(a, scale=1. / b) # Intensity value

            lambdas = np.empty((N))
            lambdas[0:n] = lambda1
            lambdas[n:N - 1] = lambda2

            x = rng.poisson(lambdas)     # Observations, x_1 ... x_N

            return x, lambdas
        except ValueError as value_error:
            print(f'Failed to generate data\n{value_error}')
            exit(1)


def sample(rng, x, E=5200, BURN_IN=200, freq=100, a=2, b=1, report=lambda epoch: print(f'Epoch={epoch}')):
    '''
    Gibbs sampler. We simulate samples by sweeping through all the posterior conditionals,
    one random variable at a time.

    Parameters:
        rng        Random number generator
        x          Sequence of counts for analysis
        E          Number of epochs for sampling
        BURN_IN    Burn in period (number of epochs that will not be recorded)
        freq       Frequency for reporting
        report     Reporter function called every freq epochs
        a          loc parameter for gamma distribution
        b          1/scale parameter for gamma distribution

    Returns:
        chain_lambda1
        chain_lambda2
        chain_n
    '''
    N = len(x)
    n = rng.choice(N)
    chain_n = np.zeros(E - BURN_IN)
    chain_lambda1 = np.zeros(E - BURN_IN)
    chain_lambda2 = np.zeros(E - BURN_IN)

    for epoch in range(E):
        if epoch % freq == 0: report(epoch)
        lambda1 = rng.gamma(a + sum(x[0:n]), scale=1. / (n + b))  # sample lambda1 from its  posterior conditional,Equation 8
        lambda2 = rng.gamma(a + sum(x[n:N]), scale=1. / (N - n + b)) # sample ambda2 from its posterior conditional, Equation 9
        mult_n = np.zeros(N)          # Now sample n
        for i in range(N): # Equation 10
            mult_n[i] = sum(x[0:i]) * np.log(lambda1) - i * lambda1 + sum(x[i:N]) * np.log(lambda2) - (N - i) * lambda2

        n = np.nonzero(rng.multinomial(1, softmax(mult_n), size=1))[1][0]
        if epoch < BURN_IN: continue
        chain_n[epoch - BURN_IN] = n
        chain_lambda1[epoch - BURN_IN] = lambda1
        chain_lambda2[epoch - BURN_IN] = lambda2

    return (chain_lambda1, chain_lambda2, chain_n)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, help='Seed for random number generator')
    parser.add_argument('--E', type=int, default=5200, help='Number of Epochs')
    parser.add_argument('--BURN_IN', type=int, default=200, help='Burn in: number of Epochs to skip at begining')
    parser.add_argument('--freq', type=int, default=1000, help='For recording progress')
    parser.add_argument('--a', type=float, default=2, help='Paramater for lambda1')
    parser.add_argument('--b', type=float, default=1, help='Paramater for lambda2')
    parser.add_argument('--N', type=int, default=50, help='Number of data points to generate')
    parser.add_argument('--m', type=int, default=5, help='Number of Mont Carlo runs')
    parser.add_argument('--figs', default='./figs', help='Folder to store plots')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be shown')
    return parser.parse_args()


if __name__ == '__main__':
    rc('font', **{'family': 'serif',
                  'serif': ['Palatino'],
                  'size': 8})
    rc('text', usetex=True)
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    x, lambdas = generate(rng, N=args.N, a=args.a, b=args.b)
    fig = figure(figsize=(14, 14))
    fig.tight_layout(pad=3.0)
    fig.suptitle('Gibbs sampler for a change-point model')

    ax1 = fig.add_subplot(args.m, 4, 1)
    ax1.stem(range(len(x)), x, linefmt='b-', markerfmt='bo', label='Counts')
    ax1.plot(range(len(x)), lambdas, 'r--', label=r'$\lambda$')
    ax1.set_ylabel('Counts')
    ax1.set_title('Observations')
    ax1.legend()

    for i in range(args.m):
        chain_lambda1, chain_lambda2, chain_n = sample(rng, x, E=args.E, BURN_IN=args.BURN_IN, a=args.a, b=args.b,
                                                       freq=args.freq, report=lambda epoch: print(f'Run={i},Epoch={epoch}'))

        ax2 = fig.add_subplot(args.m, 4, 4 * i + 2)
        ax2.plot(chain_lambda1, 'b', label=r'$\lambda 1$')
        ax2.plot(chain_lambda2, 'g', label=r'$\lambda 2$')
        if i == args.m - 1:
            ax2.set_xlabel(r'$Epoch$')
        ax2.set_ylabel(r'$\lambda$')
        ax2.set_title(r'$\lambda$')
        ax2.legend()

        ax3 = fig.add_subplot(args.m, 4, 4 * i + 3)
        ax3.hist(chain_lambda2, 20, color='g', label=r'$\lambda 2$', alpha=0.5)
        ax3.hist(chain_lambda1, 20, color='b', label=r'$\lambda 1$', alpha=0.5)
        ax3.set_xlim([0, 12])
        if i == 0:
            ax3.set_title(r'$\lambda$')
        ax3.legend()

        ax4 = fig.add_subplot(args.m, 4, 4 * i + 4)
        ax4.hist(chain_n, args.N, label='n')
        if i == 0:
            ax4.set_title('Histogram for Change Point')
        ax4.set_xlim([0, args.N])
        ax4.legend()

    fig.savefig(join(args.figs, f'{basename(__file__).split('.')[0]}'))

    elapsed = time() - start
    minutes = int(elapsed / 60)
    seconds = elapsed - 60 * minutes
    print(f'Elapsed Time {minutes}m {seconds:.2f}s')

    if args.show:
        show()

