#!/usr/bin/env python

# Copyright (C) 2022 Greenweaves Software Limited

'''Generate data for Gaussion Mixture Model'''

from numpy        import array, zeros
from numpy.random import default_rng

class GaussionMixtureModel:
    '''This class generates data using a Gaussian Mixture Model'''
    def __init__(self,
                 seed  = None,
                 mu    = array([0]),
                 sigma = array([1])):
        self.rng    = default_rng(seed)
        self.mu     = mu.copy()
        self.sigma  = sigma.copy()

    def generate(self,m):
        k = self.mu.shape[0]
        for i in range(m):
            choice = self.rng.integers(0, high=k)
            yield self.mu[choice] + self.sigma[choice] * self.rng.standard_normal()

if __name__=='__main__':
    model = GaussionMixtureModel(mu=array([0,2]),sigma=array([1,1]))
    for x in model.generate(250):
        print (x)
