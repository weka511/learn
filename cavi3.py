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
def cavi(x,K=3,N=25,tolerance=1e-12,sigma=1):
    def get_uniform_vector():
        sample = [random.random() for _ in range(K)]
        Z      = sum(sample)
        return [s/Z for s in sample]
       
    # calcELBO
    #
    # Calculate ELBO following Blei et al, equation (21)
    def calcELBO():
        t1 = np.log(s2) - m/sigma2
        t1 = t1.sum()
        t2 = -0.5*np.add.outer(x**2, s2+m**2)
        t2 += np.outer(x, m)
        t2 -= np.log(phi)
        t2 *= phi
        t2 = t2.sum()
        return t1 + t2 #log_p_x + log_p_mu + log_p_sigma - log_q_mu - log_q_sigma
    
    phi   =  np.random.dirichlet([np.random.random()*np.random.randint(1, 10)]*K, len(x))
    m      = np.array([np.quantile(x,q/K) for q in range(1,K+1)])
    s2    =  np.ones(K) * np.random.random(K)
    print('Init mean', m)
    print('Init s2', s2) 
    sigma2 = sigma**2
    ELBOs = []
    
    while (len(ELBOs)<5 or ( abs(ELBOs[-1]/ELBOs[-2]-1)>tolerance)):
        # update cluster assignment
        t1 = np.outer(x, m)
        t2 = -(0.5*m**2 + 0.5*s2)
        exponent = t1 + t2[np.newaxis, :]
        phi = np.exp(exponent)
        phi = phi / phi.sum(1)[:, np.newaxis]        
        # update variational density

        m = (phi*x[:, np.newaxis]).sum(0) * (1/sigma2 + phi.sum(0))**(-1)
        assert m.size == K
        s2 = (1/sigma2 + phi.sum(0))**(-1)
        assert s2.size == K
        
        print (f'm={m}, s2={s2}')
        ELBOs.append(calcELBO())
        if len(ELBOs)>N:
            raise Exception(f'ELBO has not converged to within {tolerance} after {N} iterations')
    return (ELBOs,phi,m,[math.sqrt(s) for s in s2])

def plot_data(xs,cs,mu=[],sigmas=[],m=0,s=1,colours=['r','g','b', 'c', 'm', 'y']):
    ax = plt.subplot(2,1,1)
    plt.hist(xs,
             bins  = 25,
             label = 'Full Dataset',
             alpha = 0.5)
    
    indices = np.argsort(m)
    m       = [m[i] for i in indices]
    s       = [s[i] for i in indices]    
    for i in range(len(mu)):
        x0s           = [xs[j] for j in range(len(xs)) if cs[j]==i ]
        n,bins,_      = plt.hist(x0s,
                            bins      = 25,
                            label     = fr'$\mu=${mu[i]:.3f}, $\sigma=${sigmas[i]:.1f}',
                            alpha     = 0.5,
                            facecolor = colours[i])
        x_values      = np.arange(min(xs), max(xs), 0.1)
        y_values      = stats.norm(mu[i], sigmas[i])
        y_values_cavi = stats.norm(m[i], s[i])
        ys            = y_values.pdf(x_values)
        ys_cavi       = y_values_cavi.pdf(x_values)
        plt.plot(x_values,
                 [y*max(n)/max(ys) for y in ys],
                 c = colours[i])
        
        plt.plot(x_values,
                [y*max(n)/max(ys_cavi) for y in ys_cavi],
                label     = fr'$\mu=${m[i]:.3f}, $\sigma=${s[i]:.3f}',
                c = colours[i],
                linestyle='--')        

    plt.legend()
    plt.title('Raw data')
    plt.xlabel('X')
    plt.ylabel('N')

def plot_ELBO(ELBOs):
    plt.plot(ELBOs,label='ELBO')
    plt.xlim(1,len(ELBOs)+1)
    plt.xlabel('Iteration')
    plt.legend()
    
if __name__=='__main__':
    plt.rcParams.update({
        "text.usetex": True
    })         
    parser = argparse.ArgumentParser('CAVI')
    parser.add_argument('--K',         type=int,   default=3,      help='Number of Gaussians')
    parser.add_argument('--n',         type=int,   default=1000,   help='Number of points')
    parser.add_argument('--N',         type=int,   default=25,     help='Number of iterations')
    parser.add_argument('--tolerance', type=float, default=1.0e-6, help='Convergence criterion')
    args = parser.parse_args()
    
    random.seed(1)
    
    sigma     = 1
    sigmas    = [1.0]*args.K
    mu        = sorted(np.random.uniform(low=0,high=5,size=args.K))
    cs,xs     = create_data(mu,sigmas=sigmas,n=args.n)

    ELBOs,phi,m,s = cavi(x         = np.asarray(xs),
                         K         = args.K,
                         N         = args.N,
                         tolerance = args.tolerance)
    plt.figure(figsize=(10,10))
    plot_data(xs,cs,
              mu     = mu,
              sigmas = sigmas,
              m      = m,
              s      = s)
    plt.subplot(2,1,2)
    plot_ELBO(ELBOs)

    plt.savefig(os.path.basename(__file__).split('.')[0] )
    plt.show()
    