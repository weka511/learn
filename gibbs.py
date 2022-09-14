#!/usr/bin/env python

# Gibbs sampler for the change-point model described in a Cognition cheat sheet titled "Gibbs sampling."
# This is a Python implementation of the procedure at http://www.cmpe.boun.edu.tr/courses/cmpe58n/fall2009/
# Written by Ilker Yildirim, September 2012.

from matplotlib.pyplot import savefig, show, subplots
from numpy             import log,exp, where, zeros
from numpy.random      import multinomial, seed
from os.path           import basename
from scipy.stats       import uniform, gamma, poisson

def generate(N = 50,
             a = 2,
             b = 1):
    '''
    Generate data

    We generate a sequence of counts. The first n counts follows a Poisson distribution with mean lambda1, and the remainder
    follow Poisson with mean lambda 2. The two lambda are sampled from a Gamma distribution, and n is also random.
    '''

    n = int(round(uniform.rvs()*N)) # Change-point: where the intensity parameter changes.
    print (f'Change point={n}')

    # Intensity values
    # We use 1/b instead of b because of the way Gamma distribution is parametrized in the package random.
    lambda1        = gamma.rvs(a,scale=1./b)
    lambda2        = gamma.rvs(a,scale=1./b)

    lambdas        = [lambda1]*n
    lambdas[n:N-1] = [lambda2]*(N-n)

    x = poisson.rvs(lambdas)     # Observations, x_1 ... x_N

    return x, lambdas, a, b

def gibbs(x,
          E         = 5200,
          BURN_IN   = 200,
          frequency = 100,
          a         = 2,
          b         = 1):
    '''Gibbs sampler'''
    N       = len(x)
    n       = int(round(uniform.rvs()*N))
    lambda1 = gamma.rvs(a,scale=1./b)
    lambda2 = gamma.rvs(a,scale=1./b)

    chain_n       = zeros(E - BURN_IN)
    chain_lambda1 = zeros(E - BURN_IN)
    chain_lambda2 = zeros(E - BURN_IN)

    for e in range(E):
        if e%frequency==0:
            print (f'At iteration {e}')
        # sample lambda1 and lambda2 from their posterior conditionals, Equation 8 and Equation 9, respectively.
        lambda1 = gamma.rvs(a+sum(x[0:n]), scale=1./(n+b))
        lambda2 = gamma.rvs(a+sum(x[n:N]), scale=1./(N-n+b))

        # sample n, Equation 10
        mult_n = zeros(N)
        for i in range(N):
            mult_n[i] = sum(x[0:i])*log(lambda1) - i*lambda1 + sum(x[i:N])*log(lambda2) - (N-i)*lambda2

        mult_n = exp(mult_n - max(mult_n))
        n      = where(multinomial(1,mult_n/sum(mult_n),size=1)==1)[1][0]

        if e>=BURN_IN:
            chain_n[e - BURN_IN]       = n
            chain_lambda1[e - BURN_IN] = lambda1
            chain_lambda2[e - BURN_IN] = lambda2

    return (chain_lambda1,chain_lambda2,chain_n)

if __name__=='__main__':
    seed(123456789)  # fix the random seed for replicability.
    x, lambdas, a, b                    = generate()
    chain_lambda1,chain_lambda2,chain_n = gibbs(x, a=a, b=b)

    f, axes = subplots(3,2,figsize=(10,10))
    f.tight_layout(pad=1.0)

    axes[0][0].stem(range(len(x)),x,linefmt='b-', markerfmt='bo')
    axes[0][0].plot(range(len(x)),lambdas,'r--')
    axes[0][0].set_ylabel('Counts')

    axes[1][0].plot(chain_lambda1,'b',chain_lambda2,'g')
    axes[1][0].set_ylabel('$\lambda$')

    axes[2][0].hist(chain_lambda1,20)
    axes[2][0].set_title('$\lambda_1$')
    axes[2][0].set_xlim([0,12])

    axes[0][1].hist(chain_lambda2,20,color='g')
    axes[0][1].set_xlim([0,12])
    axes[0][1].set_title('$\lambda_2$')

    axes[1][1].hist(chain_n,50)
    axes[1][1].set_title('n')
    axes[1][1].set_xlim([1,50])

    axes[2][1].axis('off')

    savefig(basename(__file__).split('.')[0] )
    show()
