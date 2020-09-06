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

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

def g(v):
    return v*v

def g_prime(v):
    return 2*v

v_p         = 3
phi        = v_p
Sigma_p    = 1
Sigma_u    = 1
u          = 2
epsilon_p  = (phi-v_p)/Sigma_p
epsilon_u  = (u-g(phi))/Sigma_u
phis       = [v_p]
epsilon_us = [epsilon_u]
epsilon_ps = [epsilon_p]
ts         = [0]

dt         = 0.01

for t in range(1,501):
    epsilon_p     = (phi-v_p)/Sigma_p
    epsilon_u     = (u-g(phi))/Sigma_u
    phi_dot       = epsilon_u*g_prime(phi) - epsilon_p
    epsilon_p_dot = phi - v_p    - Sigma_p *epsilon_p
    epsilon_u_dot = u -   g(phi) - Sigma_u * epsilon_u
    phi          += dt*phi_dot
    epsilon_p    += dt*epsilon_p_dot
    epsilon_u    += dt*epsilon_u_dot
    ts.append(dt*t)
    phis.append(phi)
    epsilon_us.append(epsilon_u)
    epsilon_ps.append(epsilon_p)
    
plt.scatter(ts,phis,s=1,label=r'$\phi$')
plt.scatter(ts,epsilon_us,s=1,label=r'$\epsilon_u$')
plt.scatter(ts,epsilon_ps,s=1,label=r'$\epsilon_p$')
plt.legend()
plt.title('Exercise 3')
plt.show()