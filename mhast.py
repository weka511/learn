#!/usr/bin/env python

# MH sampler for the correlation model described in the Cognition cheat sheet titled "Metropolis-Hastings sampling."
# Written by Ilker Yildirim, September 2012.

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

def generate(rng, N = 1000):
    '''
    Generate data
    '''
    data = rng.multivariate_normal([0,0],[[1, 0.4],[0.4, 1]],N)
    x = data[:,0]
    y = data[:,1]
    return N, data, x, y

def sample(N,x,y,
           E = 10000,
           BURN_IN = 0):
    # Gibbs sampler


    # Initialize the chain.
    rho = 0 # as if there's no correlation at all.

    # Store the samples
    chain_rho = np.array([0.]*(E-BURN_IN))

    accepted_number = 0.
    interval = 100
    for e in range(E):
        if e%interval==0:
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
        else:
            rho = rho    #WTF?

        # store
        if e >= BURN_IN:
            chain_rho[e-BURN_IN] = rho

    return accepted_number,chain_rho

if __name__ == '__main__':
    rng = default_rng(12345678)
    E = 10000
    N, data, x, y = generate(rng)
    accepted_number,chain_rho = sample(N,x,y,E=E)
    print ('...Summary...')
    print (f'Acceptance ratio is {accepted_number/E}')
    print (f'Mean rho is {chain_rho.mean()}')
    print (f'Std for rho is {chain_rho.std()}')
    print (f'Compare with numpy.cov function: {np.cov(data.T)}')

    f, (ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.scatter(x,y,s=20,c='b',marker='o')

    ax2.plot(chain_rho,'b')
    ax2.set_ylabel('$rho$')
    ax3.hist(chain_rho,50)
    ax3.set_xlabel('$rho$')

    plt.show()
