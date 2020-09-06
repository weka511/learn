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

import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def g(v):
    return v*v

mu_u     = 2
Sigma_u  = 1
#p_u      = stats.norm(u, Sigma_u)

def p_u_v(u,v):
    return stats.norm(g(v),Sigma_u).pdf(u)

# Prior expectation of size
vp       = 3
Sigma_p  = 1
def p_v(v):
    return stats.norm(vp, Sigma_p).pdf(v)

def p_u(u):
    return integrate.quad(lambda v:p_v(v)*p_u_v(u,v),0,6,epsabs=0.0001)[0]

vs = [0.1*i for i in range(60)]
us = [0.1*i for i in range(100)]

u = 2
sizes = [0.01 * i for i in range(1,501)]
evidence = p_u(u)

plt.scatter(sizes,[p_v(v)*p_u_v(u,v)/evidence for v in sizes],s=5,label='posterior probability')
plt.xlabel('v')
plt.ylabel('p(v|u)')
plt.ylim(0,1.4)
plt.legend()
plt.title('Exercise 1')
plt.show()


