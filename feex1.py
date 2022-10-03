#!/usr/bin/env python

# Copyright (C) 2020-2022 Greenweaves Software Limited

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

'''
   Exercise 1, posterior probabilities, from A tutorial on the free-energy
   framework for modelling perception and learning, by Rafal Bogacz
'''

from scipy.stats       import norm
from matplotlib.pyplot import figure, show
from numpy             import linspace
from scipy.integrate   import quad


def g(v):
    return v*v

def p_u_v(u,v,Sigma_u  = 1):
    return norm(g(v),Sigma_u).pdf(u)

def p_v(v, vp = 3, Sigma_p  = 1):
    '''Prior expectation of size'''
    return norm(vp, Sigma_p).pdf(v)

def p_u(u):
    return quad(lambda v:p_v(v)*p_u_v(u,v),0,6,epsabs=0.0001)[0]

def get_posterior(v,u):
    evidence      = p_u(u)
    return p_v(v)*p_u_v(u,v)/evidence

u             = 2
Sizes         = linspace(0,5,num=500)
evidence      = p_u(u)
Posterior     = get_posterior(Sizes,u)


fig = figure(figsize=(10,10))
ax  = fig.add_subplot(1,1,1)
ax.scatter(Sizes,Posterior,
           s     = 5,
           c     = 'xkcd:blue',
           label = 'posterior probability for size')
ax.set_xlabel('v')
ax.set_ylabel('p(v|u)')
ax.legend()
ax.set_title('Exercise 1')
fig.savefig('feex1')
show()
