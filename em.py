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

# Notes on the EM Algorithm for Gaussian Mixtures: CS 274A, Probabilistic Learning 
# Padhraic Smyth 
# https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf

import math
from scipy.stats import norm

# maximize_likelihood
#
# Get best GMM fit
def maximize_likelihood(xs,mus=[],sigmas=[],alphas=[],K=3,N=25,tolerance=1.0e-6):

    # has_converged
    #
    # Used to check whether likelhood has stopped improving

    def has_converged():
        return len(likelihoods)>1 and abs(likelihoods[-1]/likelihoods[-2]-1)<tolerance

    # get_log_likelihood
    #
    # Log of liklehood that xs were generated by current set of parameters

    def get_log_likelihood(mus=[],sigmas=[],alphas=[]):
        return sum([math.log(sum([alphas[k]*norm.pdf(xs[i],loc=mus[k],scale=sigmas[k]) for k in range(K)])) for i in range(len(xs))])

    # e_step
    #
    # Perform E step from EM algorithm. The ws represent the numerators in Padhraic Smyth's formulae;
    # the Zs are the denominators, which aren't they same as Smyth's zs.
    #
    # The function returns Smyth's ws.
    def e_step(mus=[],sigmas=[],alphas=[]):
        ws      = [[norm.pdf(xs[i],loc=mus[k],scale=sigmas[k])*alphas[k] for i in range(len(xs))] for k in range(K)] 
        Zs      = [sum([ws[k][i] for k in range(K)]) for i in range(len(xs))] #Normalizers 
        return [[ws[k][i]/Zs[i] for i in range(len(xs))] for k in range(K)]

    # m_step
    #
    # Perform M step from EM algorithm: calculate new values for alphas, mus, and sigmas    
    def m_step(ws):
        # Number of data points assigned to each k; since it is calculted from weights, and unrounded, the
        # values aren't exact integers.
        N       = [sum([ws[k][i] for i in range(len(xs))] ) for k in range(K)]
        # Proportion of data points assigned to each k
        alphas  = [n/sum(N) for n in N]
        mus     = [sum([ws[k][i]*xs[i] for i in range(len(xs))] )/N[k] for k in range(K)]
        sigmas  = [math.sqrt(sum([ws[k][i]*(xs[i]-mus[k])**2 for i in range(len(xs))] )/N[k]) for k in range(K)]
        return (alphas,mus,sigmas)

    # Update alphas, mus, and sigmas as long as likelihoods keep improving

    likelihoods=[]

    while len(likelihoods)<N and not has_converged():
        alphas,mus,sigmas = m_step(e_step(mus=mus,sigmas=sigmas,alphas=alphas))
        likelihoods.append(get_log_likelihood(mus=mus,sigmas=sigmas,alphas=alphas))

    return likelihoods,alphas,mus,sigmas