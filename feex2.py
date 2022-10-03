#!/usr/bin/env python

# Copyright (C) 2020-22 Greenweaves Software Limited

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

# Exercise 2--most likely size-- from A tutorial on the free-energy
# framework for modelling perception and learning, by Rafal Bogacz

from matplotlib.pyplot import figure, show
from matplotlib import rc
from numpy      import array, linspace

rc('text', usetex=True)

vp       = 3
dt       = 0.01
Sigma_p  = 1
Sigma_u  = 1
u        = 2

def g(v):
    return v*v

def g_prime(v):
    return 2*v

def new_phi(phi0,N=500):
    yield phi0
    phi = phi0
    for i in range(N):
        df = (vp-phi)/Sigma_p + (u-g(phi))*g_prime(phi)/Sigma_u
        phi += dt*df
        yield phi


Ts   = linspace(0,5,num=501)
Phis = array(list(new_phi(vp)))

fig = figure(figsize=(10,10))
ax  = fig.add_subplot(1,1,1)
ax.scatter(Ts,Phis,
           s     = 1,
           c     = 'xkcd:blue',
           label = 'Most likely size of food item')
ax.set_title('Exercise 2')
ax.set_ylim(0,3)
ax.set_xlabel('t')
ax.set_ylabel(r'$\phi$')
ax.legend()
fig.savefig('feex2')
show()
