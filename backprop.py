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

def predict(Thetas, Inputs,fn=np.vectorize(sigmoid)):
    X=Inputs
    m=len(X)
    for Theta in Thetas:
        X_with_bias=np.append(X,[1],axis=0)
        XR=np.reshape(X_with_bias,(m+1,1))
        X=fn(np.sum(np.dot(Theta,XR),axis=1))
        m=len(X)
        
    return X

def create_thetas(layer_spec):
    def create_theta(a,b):
        eps=math.sqrt(6)/math.sqrt(a+b+1)
        theta=np.random.rand(b,a+1)
        return np.subtract(np.multiply(theta,2*eps),eps)
    return [create_theta(a,b) for a,b in zip(layer_spec[:-1],layer_spec[1:])]

        
if __name__=='__main__':
    M=400
    N=25
    L=10
    Thetas=create_thetas([M,N,L])
    print (Thetas)
    print (predict(Thetas,[0 for x in range(M)]))