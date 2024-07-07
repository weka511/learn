#!/usr/bin/env python

'''
Gibbs sampler for the change-point model described in a Cognition cheat sheet titled "Gibbs sampling."

This is a Python implementation of the procedure at http://www.cmpe.boun.edu.tr/courses/cmpe58n/fall2009/
Written by Ilker Yildirim, September 2012.
'''

from argparse import ArgumentParser
from os.path import basename
from time import time
from matplotlib.pyplot import savefig, show, subplots
import numpy as np
from numpy.random import default_rng
from scipy.stats import uniform, gamma, poisson

def generate(rng, N = 50, a = 2, b = 1):
    '''
    Generate a sequence of counts. The first n counts follow a Poisson distribution with mean lambda1, and the remainder
    follow Poisson with mean lambda2. The two lambda are sampled from a Gamma distribution, and n is also random.

    Parameters:
        N       Number of observations
        a       loc parameter for gamma distribution
        b       1/scale parameter for gamma distribution

    Returns:
        x       Observations
        lambdas Intensity values
    '''

    n = int(round(rng.uniform()*N)) # Change-point: where the intensity parameter changes.

    # Intensity values
    lambda1 = rng.gamma(a,scale=1./b)
    lambda2 = rng.gamma(a,scale=1./b)

    lambdas = np.empty((N))
    lambdas[0:n] = lambda1
    lambdas[n:N-1] = lambda2

    x = rng.poisson(lambdas)     # Observations, x_1 ... x_N

    return x, lambdas

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

        mult_n = np.zeros(N)
        for i in range(N):   # Equation 10
            mult_n[i] = sum(x[0:i])*np.log(lambda1) - i*lambda1 + sum(x[i:N])*np.log(lambda2) - (N-i)*lambda2

        mult_n = np.exp(mult_n - max(mult_n))
        n = np.where(rng.multinomial(1,mult_n/sum(mult_n),size=1)==1)[1][0]

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
    parser.add_argument('--frequency', type=int, default = 100, help='Frequnce for recording progress')
    parser.add_argument('--a', type=float, default = 2, help='')
    parser.add_argument('--b', type=float,default = 1, help='')
    parser.add_argument('--N', type=int, default = 50, help='')
    parser.add_argument('--show', default = False, action = 'store_true', help='')
    return parser.parse_args()

if __name__=='__main__':
    start = time()
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x, lambdas  = generate(rng,N = args.N, a = args.a, b = args.b)
    chain_lambda1,chain_lambda2,chain_n = gibbs(rng,x, E=args.E, BURN_IN=args.BURN_IN, frequency=args.frequency, a = args.a, b = args.b)

    f, axes = subplots(2,2,figsize=(10,10))
    f.tight_layout(pad=3.0)

    axes[0][0].stem(range(len(x)),x,linefmt='b-', markerfmt='bo',label='Counts')
    axes[0][0].plot(range(len(x)),lambdas,'r--',label=r'$\lambda$')
    axes[0][0].set_ylabel('Counts')

    axes[1][0].plot(chain_lambda1,'b',label=r'$\lambda 1$')
    axes[0][0].set_title('Observations')
    axes[0][0].legend()
    axes[1][0].plot(chain_lambda2,'g',label=r'$\lambda 2$')
    axes[1][0].set_xlabel(r'$Epoch$')
    axes[1][0].set_ylabel(r'$\lambda$')
    axes[1][0].legend()

    axes[0][1].hist(chain_lambda2,20,color='g',label=r'$\lambda 2$',alpha=0.5)
    axes[0][1].hist(chain_lambda1,20,color='b',label=r'$\lambda 1$',alpha=0.5)
    axes[0][1].set_xlim([0,12])
    axes[0][1].set_title(r'$\lambda$')
    axes[0][1].legend()

    axes[1][1].hist(chain_n,50,label='n')
    axes[1][1].set_title('Histogram for Change Point')
    # axes[1][1].set_xlim([1,50])
    axes[1][1].legend()

    savefig(basename(__file__).split('.')[0] )

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes}m {seconds:.2f}s')

    if args.show:
        show()

