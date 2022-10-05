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
   Exercise 5--learn variance-- from A tutorial on the free-energy
   framework for modelling perception and learning, by Rafal Bogacz
'''

from math              import sqrt
from matplotlib.pyplot import figure, show
from matplotlib        import rc
from random            import random

rc('text', usetex=True)

phi_mean  = 5
phi_sigma = 2
phi_above = 5

dt = 0.01
MaxT = 20
N = 1000
LRate = 0.01

Sigma = [1]

for i in range(N):
    error = [1]
    e     = [0]
    phi = phi_mean + sqrt(phi_sigma) * random()
    for i in range(int(MaxT/dt)):
        error1 = error[-1]+dt*(phi-phi_above - e[-1])
        e1     = e[-1] + dt *(Sigma[-1] * error[-1] - e[-1])
        error.append(error1)
        e.append(e1)
    Sigma.append(Sigma[-1] + LRate *(error[-1]*error[-1]-1))

fig = figure(figsize=(10,10))
ax  = fig.add_subplot(1,1,1)

ax.plot(Sigma)
ax.set_xlabel('Trial')
ax.set_ylabel(r'$\Sigma$')
show()
