#!/usr/bin/env python

# Copyright (C) 2022 Greenweaves Software Limited
#
# Simon A. Crase -- simon@greenweaves.nz of +64 210 220 2257

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://github.com/weka511/learn/blob/master/LICENSE or
# <http://www.gnu.org/licenses/>.

'''Generate data for Gaussion Mixture Model'''

from numpy        import array, load, save, zeros
from numpy.random import default_rng

class GaussionMixtureModel:
    '''This class generates data using a Gaussian Mixture Model'''
    def __init__(self,
                 seed  = None,
                 mu    = array([0]),
                 sigma = array([1]),
                 name  = 'gmm',
                 m     = 100):
        k           = mu.shape[0]
        self.rng    = default_rng(seed)
        self.mu     = mu.copy()
        self.sigma  = sigma.copy()
        self.name   = name
        self.size   = (m,k)

    def save(self):
        m,k    = self.size
        choice = self.rng.integers(0, high = k, size = m)
        save(self.name, self.mu[choice] + self.sigma[choice]* self.rng.standard_normal(size=(k,m)))

    def load(self):
        with open(f'{self.name}.npy', 'rb') as f:
            return load(f)

    def generate(self,m):
        for i in range(m):
            choice = self.rng.integers(0, high=k)
            yield self.mu[choice] + self.sigma[choice] * self.rng.standard_normal()


if __name__=='__main__':
    model = GaussionMixtureModel(mu    = array([-10,20]),
                                 sigma = array([1,1]))
    model.save()
    data =model.load()
    print (data.shape)
    # for x in model.generate(250):
        # print (x)
