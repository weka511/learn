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

sigmoid_v=np.vectorize(sigmoid)

def predict(Theta1, Theta2, X):
    m1=len(X)
    XX=np.append(X,np.ones((m1,1)),axis=1)
    XT=np.dot(Theta1,XX)
    #XT1=np.sum(XT,axis=1)
    A=sigmoid_v(np.sum(XT,axis=1))
    #print (A)
    m2=len(A)
    #print (m2,np.ones((m2,1)))
    XX2=np.append(A.reshape(m2,1),np.ones((m2,1)),axis=1)
    #print (XX2)
    XTT=np.dot(Theta2,XX2)
    A2=sigmoid_v(np.sum(XTT,axis=1))
    #print (A2)
    return A2

if __name__=='__main__':
    Theta1=np.array([[1,10,100,1000],[1,1,1,1],[1,1,1,1]])
    Theta2=np.array([[1,1,1],[1,1,1]])
    #print(Theta1,Theta2)
    X0=np.array([1,2,3,4])
    m=len(X0)
    X2=X0.reshape((m,1))
    print (predict(Theta1,Theta2,X2))