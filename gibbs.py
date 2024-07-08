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

This is a Python implementation of the procedure at http://www.cmpe.boun.edu.tr/courses/cmpe58n/fall2009/
Written by Ilker Yildirim, September 2012.
'''

from argparse import ArgumentParser
from os.path import basename, join
from time import time
from matplotlib.pyplot import figure, show
import numpy as np
from numpy.random import default_rng
from scipy.stats import uniform, gamma, poisson

def generate(rng, N = 50, a = 2, b = 1, K = 5):
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
            n = int(round(rng.uniform()*N)) # Change-point: where the intensity parameter changes.

            # Intensity values
            lambda1 = rng.gamma(a,scale=1./b)
            lambda2 = rng.gamma(a,scale=1./b)

            lambdas = np.empty((N))
            lambdas[0:n] = lambda1
            lambdas[n:N-1] = lambda2

            x = rng.poisson(lambdas)     # Observations, x_1 ... x_N

            return x, lambdas
        except ValueError as value_error:
            print (value_error)


def stable_softmax(x):
    '''
    Convert a vector to proababilities, using the stable softmax described in
    https://stackoverflow.com/questions/42599498/numerically-stable-softmax
    and https://www.deeplearningbook.org/contents/numerical.html
    '''
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator/denominator

def gibbs(rng,
          x,
          E  = 5200,
          BURN_IN = 200,
          frequency = 100,
          a = 2,
          b = 1):
    '''Gibbs sampler

    Parameters:
        E
        BURN_IN
        frequency

    Returns:
        chain_lambda1
        chain_lambda2
        chain_n
    '''
    N = len(x)
    n = int(round(rng.uniform()*N))
    lambda1 = rng.gamma(a,scale=1./b)
    lambda2 = rng.gamma(a,scale=1./b)

    chain_n = np.zeros(E - BURN_IN)
    chain_lambda1 = np.zeros(E - BURN_IN)
    chain_lambda2 = np.zeros(E - BURN_IN)

    for epoch in range(E):
        if epoch%frequency==0:
            print (f'Epoch={epoch}')
        # sample lambda1 and lambda2 from their posterior conditionals
        lambda1 = rng.gamma(a+sum(x[0:n]), scale=1./(n+b))  # Equation 8
        lambda2 = rng.gamma(a+sum(x[n:N]), scale=1./(N-n+b)) # Equation 9

        # Now sample n
        mult_n = np.zeros(N)
        for i in range(N): # Equation 10
            mult_n[i] = sum(x[0:i])*np.log(lambda1) - i*lambda1 + sum(x[i:N])*np.log(lambda2) - (N-i)*lambda2

        n = np.nonzero(rng.multinomial(1,stable_softmax(mult_n),size=1))[1][0]
        if epoch >= BURN_IN:
            chain_n[epoch - BURN_IN] = n
            chain_lambda1[epoch - BURN_IN] = lambda1
            chain_lambda2[epoch - BURN_IN] = lambda2

    return (chain_lambda1,chain_lambda2,chain_n)

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, help='Seed for random number generator')
    parser.add_argument('--E',type=int,default = 5200, help='Number of Epochs')
    parser.add_argument('--BURN_IN', type=int, default = 200, help='Burn in: number of Epochs to skip at begining')
    parser.add_argument('--frequency', type=int, default = 100, help='For recording progress')
    parser.add_argument('--a', type=float, default = 2, help='Paramater for lambda1')
    parser.add_argument('--b', type=float,default = 1, help='Paramater for lambda2')
    parser.add_argument('--N', type=int, default = 50, help='Number of data points to generate')
    parser.add_argument('--m', type=int, default = 5, help='Number of Mont Carlo runs')
    parser.add_argument('--figs', default='./figs', help='Folder to store plots')
    parser.add_argument('--show', default = False, action = 'store_true', help='Controls whether plot will be shown')
    return parser.parse_args()

if __name__=='__main__':
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x, lambdas  = generate(rng,N = args.N, a = args.a, b = args.b)
    for i in range(args.m):
        chain_lambda1,chain_lambda2,chain_n = gibbs(rng,x,
                                                    E = args.E,
                                                    BURN_IN =args.BURN_IN,
                                                    frequency = args.frequency,
                                                    a = args.a,
                                                    b = args.b)
        fig = figure(figsize=(10,10))
        fig.tight_layout(pad=3.0)

        ax1 = fig.add_subplot(221)
        ax1.stem(range(len(x)),x,linefmt='b-', markerfmt='bo',label='Counts')
        ax1.plot(range(len(x)),lambdas,'r--',label=r'$\lambda$')
        ax1.set_ylabel('Counts')
        ax1.set_title('Observations')
        ax1.legend()

        ax2 = fig.add_subplot(222)
        ax2.plot(chain_lambda1,'b',label=r'$\lambda 1$')
        ax2.plot(chain_lambda2,'g',label=r'$\lambda 2$')
        ax2.set_xlabel(r'$Epoch$')
        ax2.set_ylabel(r'$\lambda$')
        ax2.legend()

        ax3 = fig.add_subplot(223)
        ax3.hist(chain_lambda2,20,color='g',label=r'$\lambda 2$',alpha=0.5)
        ax3.hist(chain_lambda1,20,color='b',label=r'$\lambda 1$',alpha=0.5)
        ax3.set_xlim([0,12])
        ax3.set_title(r'$\lambda$')
        ax3.legend()

        ax4 = fig.add_subplot(224)
        ax4.hist(chain_n,args.N,label='n')
        ax4.set_title('Histogram for Change Point')
        ax4.set_xlim([0,args.N])
        ax4.legend()

        fig.suptitle(f'Run {i+1}')
        fig.savefig(join(args.figs,f'{basename(__file__).split('.')[0]}{i+1}') )

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes}m {seconds:.2f}s')

    if args.show:
        show()

