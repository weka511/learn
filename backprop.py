# Copyright (C) 2017 Greenweaves Software Pty Ltd

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

import math,numpy as np

def sigmoid(z):
    return 1.0/(1.0+math.exp(-z))

def predict(Thetas, X,fn=np.vectorize(sigmoid)):
    XX=X
    m=len(X)
    for Theta in Thetas:
        XX=np.append(XX,np.ones((m,1)),axis=1)
        XT=np.dot(Theta,XX)
        Activation=fn(np.sum(XT,axis=1))
        m=len(Activation)
        XX=Activation.reshape(m,1)
    return XX


if __name__=='__main__':
    Theta1=np.array([[1,10,100,1000],[1,1,1,1],[1,1,1,1]])
    Theta2=np.array([[1,1,1],[1,1,1]])
    X0=np.array([1,2,3,4])
    m=len(X0)
    X2=X0.reshape((m,1))
    print (predict([Theta1,Theta2],X2))