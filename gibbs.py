# Gibbs sampler for the change-point model described in a Cognition cheat sheet titled "Gibbs sampling."
# This is a Python implementation of the procedure at http://www.cmpe.boun.edu.tr/courses/cmpe58n/fall2009/
# Written by Ilker Yildirim, September 2012.

from scipy.stats import uniform, gamma, poisson
import matplotlib.pyplot as plt
import numpy as np
from numpy import log,exp
from numpy.random import multinomial
import os

# Gibbs sampler
def gibbs(x, E = 5200, BURN_IN = 200,  frequency=100):
    
    # Initialize the chain
    n       = int(round(uniform.rvs()*N))
    lambda1 = gamma.rvs(a,scale=1./b)
    lambda2 = gamma.rvs(a,scale=1./b)
    
    # Store the samples
    chain_n       = np.array([0.]*(E-BURN_IN))
    chain_lambda1 = np.array([0.]*(E-BURN_IN))
    chain_lambda2 = np.array([0.]*(E-BURN_IN))
    
    for e in range(E):
        if e%frequency==0:
            print (f'At iteration {e}')
        # sample lambda1 and lambda2 from their posterior conditionals, Equation 8 and Equation 9, respectively.
        lambda1 = gamma.rvs(a+sum(x[0:n]), scale=1./(n+b))
        lambda2 = gamma.rvs(a+sum(x[n:N]), scale=1./(N-n+b))
    
        # sample n, Equation 10
        mult_n = np.array([0]*N)
        for i in range(N):
            mult_n[i] = sum(x[0:i])*log(lambda1) - i*lambda1 + sum(x[i:N])*log(lambda2) - (N-i)*lambda2
        mult_n = exp(mult_n-max(mult_n))
        n      = np.where(multinomial(1,mult_n/sum(mult_n),size=1)==1)[1][0]
    
        # store
        if e>=BURN_IN:
            chain_n[e-BURN_IN]       = n
            chain_lambda1[e-BURN_IN] = lambda1
            chain_lambda2[e-BURN_IN] = lambda2
            
    return (chain_lambda1,chain_lambda2,chain_n)

# fix the random seed for replicability.
np.random.seed(123456789)

# Generate data

# Hyperparameters
N = 50
a = 2
b = 1

n = int(round(uniform.rvs()*N)) # Change-point: where the intensity parameter changes.
print (f'Change point={n}')

# Intensity values
# We use 1/b instead of b because of the way Gamma distribution is parametrized in the package random.
lambda1        = gamma.rvs(a,scale=1./b)
lambda2        = gamma.rvs(a,scale=1./b)

lambdas        = [lambda1]*n
lambdas[n:N-1] = [lambda2]*(N-n)

# Observations, x_1 ... x_N
x = poisson.rvs(lambdas)

# make one big subplots and put everything in it.

f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(6,10))
f.tight_layout(pad=1.0)
# Plot the data
ax1.stem(range(N),x,linefmt='b-', markerfmt='bo')
ax1.plot(range(N),lambdas,'r--')
ax1.set_ylabel('Counts')

chain_lambda1,chain_lambda2,chain_n = gibbs(x)

ax2.plot(chain_lambda1,'b',chain_lambda2,'g')
ax2.set_ylabel('$\lambda$')

ax3.hist(chain_lambda1,20)
ax3.set_title('$\lambda_1$')
ax3.set_xlim([0,12])

ax4.hist(chain_lambda2,20,color='g')
ax4.set_xlim([0,12])
ax4.set_title('$\lambda_2$')

ax5.hist(chain_n,50)
ax5.set_title('n')
ax5.set_xlim([1,50])
plt.savefig(os.path.basename(__file__).split('.')[0] )
plt.show()
