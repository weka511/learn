# Copyright (C) 2020 Greenweaves Software Limited

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>

# Notes on the EM Algorithm for Gaussian Mixtures: CS 274A, Probabilistic Learning 
# Padhraic Smyth 
# https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf

import math,numpy as np
from scipy.stats import multivariate_normal

# sqdist
#
# Calculate the squared distance between two points
#
#    Parameters:
#        p1 One point
#        p2 The other point
#        d  Number of dimensions for space

def sqdist(p1,p2,d=3):
    return sum ([(p1[i]-p2[i])**2 for i in range(d)])

# maximize_likelihood
#
# Get best GMM fit, using 
# Notes on the EM Algorithm for Gaussian Mixtures: CS 274A, Probabilistic Learning 
# Padhraic Smyth 
# https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
#
# Parameters:
#     xs      x coordinates for all points (FSC-H)
#     ys      y coordinates for all points (SSC-H)
#     zs      z coordinates for all points (FSC-Width)
#     mus     Means--one triplet (x,y,z) for each  component in GMM
#     Sigmas  Covaraince--one matrix  for each  component in GMM
#     alphas  Proportion of points assigned  to each  component in GMM
#     K       Number of components in GMM
#     N       Max number of iterations
#     limit   Used to decide whether we have converged (ratio between the last two likelihoods is this close to 1). 

def maximize_likelihood(xs,ys,zs,mus=[],Sigmas=[],alphas=[],K=2,N=25,limit=1.0e-6):
    
    # has_converged
    #
    # Verify that the ratio between the last two likelihoods is close to 1
    def has_converged():
        return len(likelihoods)>1 and abs(likelihoods[-1]/likelihoods[-2]-1)<limit 
    
    # get_log_likelihood
    #
    # Calculate log likelihood
    #
    # Parameters:
    #     ps     matrix of Probabilies ps[k][i]--the proability of point (xs[i],ys[i],zs[i]) given cluster k
    
    def get_log_likelihood(ps):
        return sum([math.log(sum([alphas[k]*ps[k][i] for k in range(K)])) for i in range(len(xs))])
    
    # e_step
    #
    # Calculate ws for the E-step 
    #
    # Returns:
    #     ws    weights
    #     ps    For use in get_log_likelihood(...)
    def e_step():
        var = [multivariate_normal(mean=mus[k], cov=Sigmas[k]) for k in range(K)]
        ps  = [[var[k].pdf([xs[i],ys[i],zs[i]]) for i in range(len(xs))] for k in range(K)] 
        ws  = [[ps[k][i] * alphas[k] for i in range(len(xs))] for k in range(K)] # Not normalized
        Zs  = [sum([ws[k][i] for k in range(K)]) for i in range(len(xs))]
        return [[ws[k][i]/Zs[i] for i in range(len(xs))] for k in range(K)],ps
    
    # m_step
    #
    # Peform M-step
    #
    # Parameters:
    #     ws       weights
    #
    # Returns:  alphas,mus,Sigmas
    
    def m_step(ws):
        N      = [sum([ws[k][i] for i in range(len(xs))] ) for k in range(K)]
        alphas = [n/sum(N) for n in N]
        mus    = [[np.average(xs,weights=ws[k]),np.average(ys,weights=ws[k]),np.average(zs,weights=ws[k])] for k in range(K)]
        Sigmas = [np.cov([xs,ys,zs],rowvar=True,aweights=ws[k]) for k in range(K)]      
        return (alphas,mus,Sigmas)    
    
    likelihoods=[]
    try:
        while len(likelihoods)<N and not has_converged():
            ws,ps             = e_step()
            alphas,mus,Sigmas = m_step(ws)
            likelihoods.append(get_log_likelihood(ps))
        return True,likelihoods,ws,alphas,mus,Sigmas
    except(ValueError):
        return False, likelihoods,ws,alphas,mus,Sigmas
    
if __name__=='__main__':
    pass
