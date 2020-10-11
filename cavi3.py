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

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.stats as stats

def create_data(mu,n=1000,sigmas=[]):
    def create_datum():
        i = random.randrange(len(mu))
        return (i,random.gauss(mu[i],sigmas[i]))
    return list(zip(*[create_datum() for _ in range(n)]))
 
# cavi
#
# Perform Coordinate Ascent Mean-Field Variational Inference
#
# I have borrowed some ideas from Zhiya Zuo's blog--https://zhiyzuo.github.io/VI/
def cavi(x,K=3,N=25,sigmas=1,tolerance=1e-6):
    def get_uniform_vector():
        sample = [random.random() for _ in range(K)]
        return [s/sum(sample) for s in sample]
       
    # calcELBO
    #
    # Calculate ELBO following Blei et al, equation (21)
    def calcELBO():
        log_p_x     = 1 #TBP
        log_p_mu    = 0 #TBP
        log_p_sigma = 0 #TBP
        log_q_mu    = 0 #TBP
        log_q_sigma = 0 #TBP
        
        return log_p_x + log_p_mu + log_p_sigma - log_q_mu - log_q_sigma
    
    phi   = [get_uniform_vector() for _ in x]
    m     = [random.gauss(np.mean(x),sigma) for _ in range(K)]
    s     = [random.random()*np.std(x) for _ in range(K)]
    ELBOs = [1,2]
    
    while (abs(ELBOs[-1]/ELBOs[-2]-1)>tolerance):
        # update cluster assignment
        phi_update = [[math.exp(-0.5*m[k]*m[k] + s[k]*s[k] + x[i]*m[k]) for k in range(K)]  for i in range(len(x))]
        denoms     = [sum([phi_update[i][k]                             for k in range(K)]) for i in range(len(x))]
        phi        = [[phi_update[i][k]/denoms[i]                       for k in range(K)]  for i in range(len(x))]
        # update variational density
        numerator   = [sum([phi[i][k]*x[i]              for i in range(len(x))]) for k in range(K)]
        denominator = [(1/(sigma*sigma))+sum([phi[i][k] for i in range(len(x))]) for k in range(K)]
        m           = [n/d for (n,d) in zip(numerator,denominator)]
        s           = [1/d for d in denominator]
        print (m,s)
        ELBOs.append(calcELBO())
        if len(ELBOs)>N:
            raise Exception(f'ELBO has not converged to within {tolerance} after {N} iterations')
    return (N,phi,m,s)

def plot_data(xs,cs,mu=[],sigmas=[],colours=['r','g','b', 'c', 'm', 'y']):
    plt.figure(figsize=(10,10))
    plt.hist(xs,
             bins  = 25,
             label = 'Full Dataset',
             alpha = 0.5)
    
    for i in range(len(mu)):
        colour   = colours[i]
        x0s      = [xs[j] for j in range(len(xs)) if cs[j]==i ]
        n,bins,_ = plt.hist(x0s,
                            bins      = 25,
                            label     = fr'$\mu=${mu[i]:.3f}, $\sigma=${sigmas[i]:.1f}',
                            alpha     = 0.5,
                            facecolor = colour)
        x_values = np.arange(min(xs), max(xs), 0.1)
        y_values = stats.norm(mu[i], sigmas[i])
        plt.plot(x_values,
                 [y*max(n)*2 for y in y_values.pdf(x_values)],
                 c = colour)

    plt.legend()
    plt.title('Raw data')
    plt.xlabel('X')
    plt.ylabel('N')
    plt.savefig(os.path.basename(__file__).split('.')[0] )
    
if __name__=='__main__':
    plt.rcParams.update({
        "text.usetex": True
    })         
    parser = argparse.ArgumentParser('CAVI')
    parser.add_argument('--K', type=int, default=3, help='Number of Gaussians')
    parser.add_argument('--n', type=int, default=1000, help='Number of points')
    args = parser.parse_args()
    
    random.seed(1)
    
    sigma  = 3
    sigmas = [0.5]*args.K
    mu     = sorted(np.random.normal(scale=sigma,size=args.K))
    cs,xs  = create_data(mu,sigmas=sigmas,n=args.n)

    plot_data(xs,cs,mu=mu,sigmas=sigmas)
    iteration,phi,m,s = cavi(xs,K=args.K,N=100,sigmas=sigmas)
    print (phi)
    print (m)
    print (s)
    
    plt.show()
    