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

import math
import matplotlib.pyplot as plt
import numpy as np
import random

def create_parameter_set(k=6,sigma=1):
    return (k,sigma,[random.gauss(0,sigma) for _ in range(k)])

def split(data):
    return ([c for c,_ in data],[x for _,x in data])

def create_data(parameter_set,n=100,sigma=1):
    def create_datum():
        i = random.randrange(k)
        return (i,random.gauss(mu[i],sigma))
    k,_,mu=parameter_set
    return split([create_datum() for _ in range(n)])
 
# cavi
#
# Peerform Coordinate Ascent Mean-Field Variational Inference
#
# I have borrowed some ideas from see https://zhiyzuo.github.io/VI/
def cavi(x,k=6,N=25,sigma=1):
    def get_uniform_vector():
        sample = [random.random() for _ in range(k)]
        return [s/sum(sample) for s in sample]
    
    def converged():
        pass
    
    phi = [get_uniform_vector() for _ in x]
    m   = [random.gauss(np.mean(x),sigma) for _ in range(k)]
    s   = [random.random()*np.std(x) for _ in range(k)]
    for iteration in range(N):
        # update cluster assignment
        phi_update = [[math.exp(-0.5*m[_k]*m[_k] + s[_k]*s[_k] + x[i]*m[_k]) for _k in range(k)] for i in range(len(x))]
        denoms     = [sum([phi_update[i][_k] for _k in range(k)] ) for i in range(len(x))]
        phi        = [[phi_update[i][_k]/denoms[i] for _k in range(k)] for i in range(len(x))]
        # update variational density
        numerator   = [sum([phi[i][_k]*x[i] for i in range(len(x))]) for _k in range(k)]
        denominator =  [(1/(sigma*sigma))+sum([phi[i][_k] for i in range(len(x))]) for _k in range(k)]
        m           = [n/d for (n,d) in zip(numerator,denominator)]
        s           = [1/d for d in denominator]
        print (m,s)
        # compute elbo
        if converged():
            return (True,iteration,phi,m,s)
    return (False,N,phi,m,s)

if __name__=='__main__':
    
    random.seed(1)
    parameter_set = create_parameter_set(sigma=1)
    cs,xs         = create_data(parameter_set)
    plt.hist(xs)
    k,sigma,_     = parameter_set
    success,iteration,phi,m,s = cavi(xs,k=k,N=100,sigma=sigma)
    print (phi)
    print (m)
    print (s)
    plt.show()
    