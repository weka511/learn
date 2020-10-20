from scipy.stats import uniform, gamma, poisson
import matplotlib.pyplot as plt
import numpy as np
from numpy import log,exp
from numpy.random import multinomial

def gibbs(x,
          E         = 5200,
          BURN_IN   = 200,
          frequency = 100,
          init      = lambda : [],
          sample    = lambda i,gibbs_sample: []):
    
    gibbs_sample = init()
    m            = len(gibbs_sample)
    chain        = np.zeros((E-BURN_IN,n))
    
    for e in range(E):
        if e%frequency==0:
            print (f'At iteration {e}')
        for i in range(m):
            gibbs_sample[i]  = sample(i,gibbs_sample)
        if e>=BURN_IN:
            chain[e-BURN_IN] = gibbs_sample
            
    return chain

if __name__=='__main__':
    N = 50
    a = 2
    b = 1
    
    n = int(round(uniform.rvs()*N)) # Change-point: where the intensity parameter changes.
    print (f'Change point={n}')
    lambda1        = gamma.rvs(a,scale=1./b)
    lambda2        = gamma.rvs(a,scale=1./b)
    
    lambdas        = [lambda1]*n
    lambdas[n:N-1] = [lambda2]*(N-n)
    
    def init():
        return[int(round(uniform.rvs()*N)),
               gamma.rvs(a,scale=1./b),
               gamma.rvs(a,scale=1./b)]
    
    def sample(i,gibbs_sample):
        assert i in [0,1,2]

        if i==0:
            n = int(gibbs_sample[2])
            return gamma.rvs(a+sum(x[0:n]), scale=1./(n+b))
        elif i==1:
            n = int(gibbs_sample[2])
            return gamma.rvs(a+sum(x[n:N]), scale=1./(N-n+b))
        else:
            lambda1 = gibbs_sample[0]
            lambda2 = gibbs_sample[1]            
            mult_n = np.array([0]*N)
            for i in range(N):
                mult_n[i] = sum(x[0:i])*log(lambda1) - i*lambda1 + sum(x[i:N])*log(lambda2) - (N-i)*lambda2
            mult_n = exp(mult_n-max(mult_n))
            n      = np.where(multinomial(1,mult_n/sum(mult_n),size=1)==1)[1][0]
            return n
 
    x = poisson.rvs(lambdas)    
   
    
    f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(6,10))
    f.tight_layout(pad=1.0)
    # Plot the data
    ax1.stem(range(N),x,linefmt='b-', markerfmt='bo')
    ax1.plot(range(N),lambdas,'r--')
    ax1.set_ylabel('Counts')
    
    chain = gibbs(x,init = init, sample= sample)
    chain_lambda1 = chain[:,0]
    chain_lambda2 = chain[:,1]
    chain_n       = chain[:,2]
    
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
    
    plt.show()    