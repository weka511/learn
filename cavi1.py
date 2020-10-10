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
#  and
#  Coordinate Ascent Mean-field Variational Inference (Univariate Gaussian Example)
#  https://suzyahyah.github.io/bayesian%20inference/machine%20learning/variational%20inference/2019/03/20/CAVI.html

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import random

# cavi
#
# Perform Coordinate Ascent Mean-Field Variational Inference
#

def cavi(xs,
         N         = 25,
         tolerance = 1e-6):
    
    # calcELBO
    #
    # Calculate ELBO following Blei et al, equation (21)
    def calcELBO():
        log_p_x     = -(0.5 / sigma) * sum([(x-mu)**2 for x in xs]) 
        log_p_mu    = -(0.5 / sigma) * sum([(x-mu_0)**2 for x in xs])
        log_p_sigma = (-alpha_0-1) * math.log(sigma) - beta_0/sigma
        log_q_mu    = -(0.5 / sigma) *(1/(len(xs)+1)) * (mu - mu_n) **2
        log_q_sigma = (alpha_n-1) * math.log(sigma) - 1/sigma * beta_n
        
        return log_p_x + log_p_mu + log_p_sigma - log_q_mu - log_q_sigma
    
    x_bar    = np.mean(xs)
    ndata    = len(xs)
    ELBOs    = [1,2]
    # hyperpriors
    mu_0     = 0
    alpha_0  = 0
    beta_0   = 0
    sigma_0  = 2

    mu_n    = mu_0
    alpha_n = alpha_0
    beta_n  = beta_0
    
    sigma   = sigma_0
    mu      = random.gauss(mu_0,sigma)

    while (abs(ELBOs[-1]/ELBOs[-2]-1)>tolerance):
        mu_n    = (ndata*x_bar + mu_0)/(ndata + 1)
        mu      = mu_n
        alpha_n = alpha_0 + (ndata + 1)/2
        beta_n  = beta_0 +  0.5*sum(x**2 for x in xs) - x_bar * sum(xs)   \
                  + ((ndata+1)/2)*(sigma/ndata + x_bar**2)- mu_0*x_bar + 0.5*mu_0**2
        sigma   = (beta_n-1)/alpha_n
        ELBOs.append(calcELBO())
        if len(ELBOs)>N:
            raise Exception(f'ELBO has not converged to within {tolerance} after {N} iterations')
    return (mu,sigma,ELBOs)

def plot_data(xs,ys,ax=None):
    ax.hist(xs, bins=30, alpha=0.5, label=fr'Data: $\mu=${np.mean(xs):.3f}, $\sigma=${np.std(xs):.3f}',color='r') 
    ax.hist(ys, bins=30, alpha=0.5, label=fr'Samples: $\mu=${mu:.3f}, $\sigma=${sigma:.3f}',color='b') 
    ax.set_title ('Data and Samples')
    ax.legend()

def plot_ELBO(ELBOs,ax=None):
    ax.set_title ('ELBO')
    ax.plot(range(1,len(ELBOs)-1),ELBOs[2:])
    ax.set_xticks(range(1,len(ELBOs)-1))
    ax.set_ylabel('ELBO')
    ax.set_xlabel('Iteration')
    
if __name__=='__main__':
    
    N  = 1000
    plt.rcParams.update({
        "text.usetex": True
    })     
    
    xs             = np.random.normal(loc=0.5,scale=0.5,size=N)
    mu,sigma,ELBOs = cavi(xs) 
    plt.figure(figsize=(10,10))
    plot_data(xs,
              np.random.normal(loc=mu,scale=sigma,size=N),
              ax=plt.subplot(211))
    plot_ELBO(ELBOs,ax=plt.subplot(212))
    
    plt.savefig(os.path.basename(__file__).split('.')[0] )
    plt.show()
    