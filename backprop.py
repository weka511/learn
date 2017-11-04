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

import math,numpy as np,random,unittest

def sigmoid(z):
    return 1.0/(1.0+math.exp(-z))

def predict(Thetas, Inputs,fn=np.vectorize(sigmoid)):
    X=Inputs
    m=len(X)
    derivatives=[]
    for Theta in Thetas:
        X_with_bias=np.append(X,[1],axis=0)
        X=fn(np.dot(X_with_bias,Theta))
        derivatives.append([g*(1-g) for g in X])
        m=len(X)       
    return (X,derivatives)

def create_thetas(layer_spec):
    def create_theta(a,b):
        eps=math.sqrt(6)/math.sqrt(a+b+1)
        theta=np.random.rand(b,a+1)
        return np.subtract(np.multiply(theta,2*eps),eps)
    return [create_theta(a,b) for a,b in zip(layer_spec[:-1],layer_spec[1:])]

def train(Thetas, Inputs,Outputs,fn=np.vectorize(sigmoid)):
    a_k,derivatives=predict(Thetas, Inputs,fn=fn)
    delta_k=[a - o for (a,o) in zip(a_k,Outputs)]
    print (delta_k)
    Theta2=Thetas[-1]
    print (Theta2)
    print (derivatives[-1])
    
if __name__=='__main__':
    class TestEvaluation(unittest.TestCase):
        '''Tests based on https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/'''
        def test_forward(self):
            '''A test to verify correct calculation of feedforward network'''
            Theta1=np.array([[0.15,0.25],[0.2,0.3],[0.35,0.35]])
            Theta2=np.array([[0.4,0.50],[0.45,0.55],[0.6,0.6]])
            z,_=predict([Theta1,Theta2],[0.05,0.1])
            self.assertAlmostEqual(0.75136507,z[0],delta=0.0000001)
            self.assertAlmostEqual(0.772928465,z[1],delta=0.0000001)
    
    unittest.main()
