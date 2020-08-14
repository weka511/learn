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


def create_data(n=10000,sigma=1,mu=0):
    return [random.gauss(mu,sigma) for _ in range(n)]
 
# cavi
#
# Peerform Coordinate Ascent Mean-Field Variational Inference
#
# I have borrowed some ideas from see https://zhiyzuo.github.io/VI/
def cavi(xs,N=25,tolerance=1.01):
    def calc_elbo():
        x_mu_sq   = sum([(x-mu)*(x-mu) for x in xs]) 
        cov_term  = -0.5 / sigma
        log_p_x   = cov_term * x_mu_sq
        log_p_mu  = cov_term * sum([(x-mu0)*(x-mu0) for x in xs])
        log_p_sigma = (-alpha0-1) * math.log(sigma) - beta0/sigma
        log_q_mu    = cov_term/(len(xs)+1) * (mu - mu_n) * (mu - mu_n)
        log_q_sigma = (alpha_n-1) * math.log(sigma) - 1/sigma * beta_n
        return log_p_x + log_p_mu + log_p_sigma - log_q_mu - log_q_sigma
    
    elbos   = []
    mu0     = 0
    sigma0  = 2
    alpha0  = 0
    beta0   = 0
    mu_n    = mu0
    alpha_n = alpha0
    beta_n  = beta0
    
    mu      = random.gauss(mu0,sigma0)
    sigma   = sigma0
    
    for iteration in range(N):
        # update cluster assignment

        # update variational density
 
        # compute elbo
        elbos.append(calc_elbo())
        if len(elbos)>2 and elbos[-1]<elbos[-2]*tolerance:
            return (True,iteration,elbos)
    return (False,N,elbos)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    random.seed(1)
    plt.subplot(211)
    xs         = create_data()
    plt.hist(xs)
    plt.subplot(212)
    _,_,elbos = cavi(xs)
    plt.plot(range(len(elbos)),elbos)
    plt.show()
    