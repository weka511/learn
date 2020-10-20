#    Copyright (C) 2020 Greenweaves Software Limited

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

# A generic Gibbs sampler based on:
# Bayesian Inference: Gibbs Sampling -- http://www2.bcs.rochester.edu/sites/jacobslab/cheat_sheets.html

import numpy as np

def gibbs(x,
          E         = 5200,
          BURN_IN   = 200,
          frequency = 100,
          init      = lambda : [],
          sample    = lambda i,gibbs_sample: []):
    
    gibbs_sample = init()
    m            = len(gibbs_sample)
    chain        = np.zeros((E-BURN_IN,m))
    
    for e in range(E):
        if e%frequency==0:
            print (f'At iteration {e}')
        for i in range(m):
            gibbs_sample[i]    = sample(i,gibbs_sample)
        if e>=BURN_IN:
            chain[e-BURN_IN,:] = gibbs_sample
            
    return chain

if __name__=='__main__':
    from scipy.stats import uniform, gamma, poisson
    import matplotlib.pyplot as plt 
    from numpy import log,exp
    from numpy.random import multinomial
    import os

    def init():
        return[int(round(uniform.rvs()*N)),
               gamma.rvs(a,scale=1./b),
               gamma.rvs(a,scale=1./b)]

    def sample(i,gibbs_sample):
        if i==0:
            return gamma.rvs(a+sum(x[0:int(gibbs_sample[2])]),
                             scale=1./(int(gibbs_sample[2])+b))
        elif i==1:
            return gamma.rvs(a+sum(x[int(gibbs_sample[2]):N]), 
                             scale=1./(N-int(gibbs_sample[2])+b))
        elif i==2:     
            mult_n  = np.array([0]*N)
            for i in range(N):
                mult_n[i] = sum(x[0:i])*log(gibbs_sample[0]) - i*gibbs_sample[0]\
                            + sum(x[i:N])*log(gibbs_sample[1]) - (N-i)*gibbs_sample[1]
            mult_n = exp(mult_n-max(mult_n))
            return np.where(multinomial(1,mult_n/sum(mult_n),size=1)==1)[1][0]    
    np.random.seed(123456789)
    N              = 50
    a              = 2
    b              = 1
    
    n              = int(round(uniform.rvs()*N)) # Change-point: where the intensity parameter changes.
    lambda1        = gamma.rvs(a,scale=1./b)
    lambda2        = gamma.rvs(a,scale=1./b)
    
    lambdas        = [lambda1]*n
    lambdas[n:N-1] = [lambda2]*(N-n)

    print (f'Change point={n}, lambdas=({lambda1},{lambda2})')    
  
 
    x             = poisson.rvs(lambdas)    
    chain         = gibbs(x,init = init, sample= sample)
    chain_lambda1 = chain[:,0]
    chain_lambda2 = chain[:,1]
    chain_n       = chain[:,2]
    
    f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(6,10))
    f.tight_layout(pad=1.0)

    ax1.stem(range(N),x,linefmt='b-', markerfmt='bo')
    ax1.plot(range(N),lambdas,'r--')
    ax1.set_ylabel('Counts')
    
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