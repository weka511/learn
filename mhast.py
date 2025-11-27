#!/usr/bin/env python

#   Copyright (C) 2020-2025 Simon Crase

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
Metropolis Hasting Sampler after the Cognition cheat sheet
'''

from os.path import basename, join
from argparse import ArgumentParser
from matplotlib.pyplot import figure, show
import numpy as np


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, help='Seed for random number generator')
    parser.add_argument('--E', type=int, default=5200, help='Number of Epochs')
    parser.add_argument('--BURN_IN', type=int, default=200, help='Burn in: number of Epochs to skip at begining')
    parser.add_argument('--frequency', type=int, default=100, help='For recording progress')
    parser.add_argument('--N', type=int, default=1000, help='Number of data points to generate')
    parser.add_argument('--figs', default='./figs', help='Folder to store plots')
    parser.add_argument('--show', default=False, action='store_true', help='Controls whether plot will be shown')
    parser.add_argument('--width', default=0.14, type=float,help='Used to generate candidate values of rho')
    return parser.parse_args()


def generate(N=1000, mean=np.array([0, 0]), cov=np.array([[1, 0.4], [0.4, 1]]), rng=np.random.default_rng()):
    '''
    Generate data

    Parameters:
        N      Number of data points
        mean   Mean for generated data
        cov    Covariance matrix for  for generated data
        rng    Randon number generator
    '''
    data = rng.multivariate_normal(mean, cov, N)
    x = data[:, 0]
    y = data[:, 1]
    return data, x, y


def sample(x, y, E=10000, BURN_IN=0, frequency=100, width=0.14):
    '''
    Metropolis Hasting Sampler

    Parameters:
        x         Data points for first stream
        y         Data points second stream
        E         Number of Epoch
        BURN_IN   Burn in: number of Epochs to skip at begining
        frequency For recording progress
        width     Used to generate candidate values of rho

    Returns:
        acceptance_ratio  Acceptance ratio for sampling
        chain_rho         Chain of estimates for rho
    '''
    def get_acceptance_probability(rho, rho_candidate):
        '''
        Compute the acceptance probability, Equation 8 and Equation 6.
        We will do both equations in log domain to avoid underflow.

        Parameters:
            rho
            rho_candidate
        '''
        def get_acceptance_factor(rho):
            '''
            Compute p(rho) from Equation 6

            Parameters:
                rho     Either an exisiting rho or rho_candidate
            '''
            return (-(3/2) * np.log(1 - rho**2)
                    - N * np.log((1 - rho**2)**(1/2))
                    - sum(1/(2 * (1 - rho**2)) * (x**2 - 2*rho*x*y + y**2)))

        return np.exp(min([0, get_acceptance_factor(rho_candidate) - get_acceptance_factor(rho)]))

    assert E > BURN_IN
    assert len(x) == len(y)

    N = len(x)
    rho = 0    # Starting value assumes x and y are not correlated
    chain_rho = np.empty((E - BURN_IN))
    accepted = 0

    for iteration in range(E):
        # Draw a value from the proposal distribution, Equation 7
        rho_candidate = rng.uniform(low=rho - width / 2, high=rho + width / 2)

        if rng.uniform(0, 1) < get_acceptance_probability(rho, rho_candidate):
            rho = rho_candidate
            if iteration >= BURN_IN:
                accepted += 1

        if iteration >= BURN_IN:
            chain_rho[iteration - BURN_IN] = rho
            if iteration % frequency == 0:
                print(f'At iteration {iteration}')

    return accepted / (E - BURN_IN), chain_rho


if __name__ == '__main__':
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data, x, y = generate(N=args.N, rng=rng)
    acceptance_ratio, chain_rho = sample(x, y, E=args.E, BURN_IN=args.BURN_IN, frequency=args.frequency, width=args.width)
    print('...Summary...')
    print(f'Acceptance ratio is {acceptance_ratio}')
    print(f'Mean rho is {chain_rho.mean()}')
    print(f'Std for rho is {chain_rho.std()}')
    print(f'Compare with numpy.cov function: {np.cov(data.T)}')

    fig = figure(figsize=(10, 10))
    fig.tight_layout(pad=3.0)

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.scatter(x, y, s=1, c='b', marker='o')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Data')

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(chain_rho, 'b', label=r'$\rho$')
    ax2.axhline(y=chain_rho.mean(), color='r', linestyle='dotted', label=fr'$\mu=${chain_rho.mean():.2f}')
    ax2.axhline(y=chain_rho.mean() + chain_rho.std(), color='r', linestyle='dashed', label=r'$\mu+\sigma$')
    ax2.axhline(y=chain_rho.mean() - chain_rho.std(), color='r', linestyle='dashdot', label=r'$\mu-\sigma$')
    ax2.set_ylabel(r'$\rho$')
    ax2.legend()
    ax2.set_title(rf'Chain: acceptance ratio={acceptance_ratio:.2f}, $\sigma=${chain_rho.std():.2f}')

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.hist(chain_rho, 50, color='blue', ec='blue', density=True)
    ax3.set_xlabel(r'$\rho$')
    ax3.set_title(r'Histogram for $\rho$')

    fig.suptitle(__doc__)
    fig.tight_layout(pad=3, h_pad=4, w_pad=3)
    fig.savefig(join(args.figs, f'{basename(__file__).split('.')[0]}'))

    if args.show:
        show()
