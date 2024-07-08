#!/usr/bin/env python

# MH sampler for the correlation model described in the Cognition cheat sheet titled "Metropolis-Hastings sampling."
# Written by Ilker Yildirim, September 2012.

from os.path import basename, join
from argparse import ArgumentParser
from matplotlib.pyplot import figure, show
import numpy as np
from numpy.random import default_rng

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, help='Seed for random number generator')
    parser.add_argument('--E',type=int,default = 5200, help='Number of Epochs')
    parser.add_argument('--BURN_IN', type=int, default = 200, help='Burn in: number of Epochs to skip at begining')
    parser.add_argument('--frequency', type=int, default = 100, help='For recording progress')
    parser.add_argument('--N', type=int, default = 1000, help='Number of data points to generate')
    parser.add_argument('--figs', default='./figs', help='Folder to store plots')
    parser.add_argument('--show', default = False, action = 'store_true', help='Controls whether plot will be shown')
    return parser.parse_args()

def generate(rng, N = 1000):
    '''
    Generate data
    '''
    data = rng.multivariate_normal([0,0],[[1, 0.4],[0.4, 1]],N)
    x = data[:,0]
    y = data[:,1]
    return data, x, y

def sample(N,x,y,
           E = 10000,
           BURN_IN = 0,
           frequency = 100):

    rho = 0 # Initialize the chain as if there's no correlation at all.

    # Store the samples
    chain_rho = np.array([0.]*(E - BURN_IN))

    accepted_number = 0.

    for e in range(E):
        if e%frequency==0:
            print (f'At iteration {e}')
        # Draw a value from the proposal distribution, Uniform(rho-0.07,rho+0.07); Equation 7
        rho_candidate = rng.uniform(rho-0.07,2*0.07)

        # Compute the acceptance probability, Equation 8 and Equation 6.
        # We will do both equations in log domain here to avoid underflow.
        accept = (-3./2*np.log(1.-rho_candidate**2)
                  - N*np.log((1.-rho_candidate**2)**(1./2))
                  - sum(1./(2.*(1.-rho_candidate**2))*(x**2-2.*rho_candidate*x*y+y**2)))
        accept = accept - (-3./2*np.log(1.-rho**2)
                           - N*np.log((1.-rho**2)**(1./2))
                           - sum(1./(2.*(1.-rho**2))*(x**2-2.*rho*x*y+y**2)))
        accept = min([0,accept])
        accept = np.exp(accept)

        # Accept rho_candidate with probability accept.
        if rng.uniform(0,1) < accept:
            rho = rho_candidate
            accepted_number = accepted_number + 1

        if e >= BURN_IN:
            chain_rho[e-BURN_IN] = rho

    return accepted_number,chain_rho

if __name__ == '__main__':
    args = parse_args()
    rng = default_rng(args.seed)
    data, x, y = generate(rng, N=args.N)
    accepted_number,chain_rho = sample(args.N,x,y,
                                       E = args.E,
                                       BURN_IN = args.BURN_IN,
                                       frequency = args.frequency)
    print ('...Summary...')
    print (f'Acceptance ratio is {accepted_number/args.E}')
    print (f'Mean rho is {chain_rho.mean()}')
    print (f'Std for rho is {chain_rho.std()}')
    print (f'Compare with numpy.cov function: {np.cov(data.T)}')

    fig = figure(figsize=(10,10))
    fig.tight_layout(pad=3.0)

    ax1 = fig.add_subplot(311)
    ax1.scatter(x,y,s=20,c='b',marker='o')

    ax2 = fig.add_subplot(312)
    ax2.plot(chain_rho,'b')
    ax2.set_ylabel(r'$\rho$')

    ax3 = fig.add_subplot(313)
    ax3.hist(chain_rho,50)
    ax3.set_xlabel('$rho$')
    fig.savefig(join(args.figs,f'{basename(__file__).split('.')[0]}') )
    if args.show:
        show()
