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

# Exercise 2 from A tutorial on the free-energy framework for modelling perception
# and learning, by Rafal Bogacz

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

vp       = 3
phis     = [vp]
ts       = [0]
dt       = 0.01
Sigma_p  = 1
Sigma_u  = 1
u        = 2

def g(v):
    return v*v

def g_prime(v):
    return 2*v

def new_phi(phi):
    df = (vp-phi)/Sigma_p + (u-g(phi))*g_prime(phi)/Sigma_u
    return phi + dt*df

for t in range(1,501):
    ts.append(dt*t)
    phis.append(new_phi(phis[-1]))

plt.scatter(ts,phis,s=1,label='Most likely size of food item')
plt.title('Exercise 2')
plt.ylim(0,3)
plt.xlabel('t')
plt.ylabel(r'$\phi$')
plt.legend()
plt.show()
