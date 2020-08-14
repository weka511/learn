# Copyright (C) 2020 Greenweaves Software Limited

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#  Coordinate Ascent Mean-Field Variational Inference (CAVI) after
#  David M. Blei, Alp Kucukelbir & Jon D. McAuliffe (2017) Variational Inference: A Review for Statisticians

import random, numpy as np, math


def create_data(n=100,sigma=1,mu=0):
    return [random.gauss(mu,sigma) for _ in range(n)]
 
# cavi
#
# Perform Coordinate Ascent Mean-Field Variational Inference
#
# I have borrowed some ideas from 
# https://suzyahyah.github.io/bayesian%20inference/machine%20learning/variational%20inference/2019/03/20/CAVI.html
def cavi(xs,N=25,N0=5,tolerance=0.000001):
    def calc_elbo():
        x_mu_sq     = sum([(x-mu)*(x-mu) for x in xs]) 
        cov_term    = -0.5 / sigma
        log_p_x     = cov_term * x_mu_sq
        log_p_mu    = cov_term * sum([(x-mu0)*(x-mu0) for x in xs])
        log_p_sigma = (-alpha0-1) * math.log(sigma) - beta0/sigma
        log_q_mu    = cov_term/(len(xs)+1) * (mu - mu_n) * (mu - mu_n)
        log_q_sigma = (alpha_n-1) * math.log(sigma) - 1/sigma * beta_n
        
        return log_p_x + log_p_mu + log_p_sigma - log_q_mu - log_q_sigma
    
    xbar    = np.mean(xs)
    
    elbos   = []
    # hyperpriors
    mu0     = 0
    alpha0  = 0
    beta0   = 0
    sigma0  = 2

    mu_n    = mu0
    alpha_n = alpha0
    beta_n  = beta0
    
    sigma   = sigma0
    mu      = random.gauss(mu0,sigma)

    for _ in range(N):
        if len(elbos)>N0 and abs(elbos[-1]-elbos[-2])<tolerance: return (True,mu,sigma,elbos)
        
        mu_n    = (sum(xs) + mu0)/(len(xs)+1)
        mu      = mu_n
        alpha_n = alpha0 + (len(xs)+1)/2
        beta_n  = beta0 +  0.5*sum(x**2 for x in xs) - xbar * sum(x for x in xs)   \
                  + ((len(xs)+1)/2)*(sigma/len(xs) + xbar**2)- mu0*xbar + 0.5*mu0*mu0
        sigma = (beta_n-1)/alpha_n
        # compute elbo
        elbos.append(calc_elbo())
 
    return (False,mu,sigma,elbos)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    random.seed(1)
    xs = create_data(mu=0.5,sigma=0.5,n=1000)
    _,mu,sigma,elbos = cavi(xs,N0=20)
    
    plt.figure(figsize=(10,10))
    ax1=plt.subplot(211)
    ax1.hist(xs) 
    ax1.set_title (f'Data: mean(xs)={np.mean(xs):.3f}, std(xs)={np.std(xs):.3f}')
    ax2=plt.subplot(212)
    ax2.set_title (f'ELBO: mu={mu:.3f}, sigma={sigma:.3f}')
    ax2.plot(elbos)
    plt.show()
    