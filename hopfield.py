# Copyright (C) 2021 Greenweaves Software Limited

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

from numpy import array, copy, dot, equal, fill_diagonal, outer, zeros
from random import shuffle

def theta(x):
    return 1 if x>0 else -1

class Hopfield:
    def __init__(self,n,init=lambda m: zeros((m,m))):
        self.n = n
        self.W = init(n)

    def store(self,y):
        self.W += outer(y,y)

    def zero_diagonal(self):
        fill_diagonal(self.W,0)
        print (self.W)

    def evaluate(self,y):
        y0 = copy(y)
        y1,E = self.step(y0)
        Es   = [E]
        while True:
            if equal(y1,y0).all(): return y0,Es
            y0 = copy(y1)
            y1,E = self.step(y0)
            Es.append(E)

    def step(self,y):
        deltaE  = 0
        indices = list(range(self.n))
        shuffle(indices)
        for i in indices:
            product = theta(dot(self.W[i,:],y))
            if y[i]*product < 0:
                deltaE += y[i]*product
                y[i]*=-1
        return y,deltaE

if __name__=='__main__':
    hopfield = Hopfield(9)
    hopfield.store(array([ 1, -1,  1, -1, 1, -1,  1, -1,  1]))
    hopfield.store(array([ 1,  1, -1, -1, 1,  1, -1, -1,  1]))
    hopfield.store(array([-1,  1, -1, -1, 1, -1, -1,  1, -1]))
    hopfield.zero_diagonal()

    y = array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    print (hopfield.evaluate(y))

