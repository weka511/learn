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

import math,numpy as np,random,time,os,glob

def sigmoid(z):
    try:
        return 1.0/(1.0+math.exp(-z))
    except OverflowError as e:
        print (e,z)
        return 1 if z>0 else 0

def predict(Thetas, Inputs,fn=np.vectorize(sigmoid)):
    X=Inputs
    activations=[]
    Xs=[]
    for Theta in Thetas:
        X_with_bias=np.append(X,[1],axis=0)
        Xs.append(X_with_bias)
        X=fn(np.dot(X_with_bias,Theta))
        activations.append(X)    
    return (X,activations,Xs)

def create_thetas(layer_spec):
    def create_theta(a,b):
        eps=math.sqrt(6)/math.sqrt(a+b+1)
        theta=np.random.rand(a+1,b)
        return np.subtract(np.multiply(theta,2*eps),eps)
    return [create_theta(a,b) for a,b in zip(layer_spec[:-1],layer_spec[1:])]

def error(target,output):
    return 0.5*sum([(t-o)*(t-o) for t,o in zip(target,output)])
    
def delta_weights(target,output,activations,Xs,Thetas):
    errors=np.subtract(output,target)
    deltas=[]
    for g,X,theta in zip(activations[::-1],Xs[::-1],Thetas[::-1]):
        delta = np.outer(X,
                         np.multiply(np.multiply(g,
                                                 np.subtract(1,g)),errors))
        deltas.append(delta)
        g2=np.multiply(g,np.subtract(1,g))
        partialEPartialNet=np.multiply(g2,errors)
        errors=np.sum(np.multiply(partialEPartialNet,theta),axis=1)[:-1]

    return deltas



def gradient_descent(Thetas,
                     data_source=None,
                     eta=0.5,
                     print_interval=100,
                     output=lambda i,maximum_error,average_error,Thetas: print ('{0} {1:9.3g} {2:9.3g}'.format(i,maximum_error,average_error))):


    total_error=0
    maximum_error=0
    i=0
    m=0
    for target,Input in data_source:
        z,activations,Xs=predict(Thetas,Input)
        err=error(target,z)
        total_error+=err
        if err>maximum_error:
            maximum_error=err
        m+=1
        deltas_same_sequence_thetas=delta_weights(target,z,activations,Xs,Thetas)[::-1] #NB - reversed!
        Thetas[:]=[np.subtract(Theta,np.multiply(eta,delta))  for (Theta,delta) in zip(Thetas,deltas_same_sequence_thetas)]
        if i%print_interval==0:
            output(i,maximum_error,total_error/m,Thetas) 
            total_error=0
            maximum_error=0
            m=0 
        i+=1

    return (Thetas,err)

def get_status_file_name(run='nn',ext='txt',path='./weights'):
    return os.path.join(path,'{0}.{1}'.format(run,ext))

def save_status(i,maximum_error,average_error,run='nn',ext='txt',path='./weights'):
    with open (get_status_file_name(run=run,ext=ext,path=path),'a') as status_file:
        status_file.write('{0},{1},{2:.3g},{3:.3g}\n'.format(time.strftime('%Y-%m-%d-%H-%M-%S',time.gmtime()),i,maximum_error,average_error))

def load(run='nn',ext='npy',path='./weights'):
    matches=glob.glob('{0}*.{1}'.format(os.path.join(path,run),ext))
    matches.sort()
    print (matches)
    return np.load(matches[len(matches)-1])

def save(Thetas,run='nn',ext='npy',path='./weights',max_files=3):
    np.save(os.path.join(path,
                         '{0}-{1}.{2}'.format(run,
                                              time.strftime('%Y-%m-%d-%H-%M-%S',time.gmtime()), 
                                              ext)),
            Thetas)
    matches=glob.glob('{0}*.{1}'.format(os.path.join(path,run),ext))
    if len(matches)>max_files:
        matches.sort()
        for file in matches[:-max_files]:
            os.remove(file)
    
if __name__=='__main__':
    import unittest
    
    class TestEvaluation(unittest.TestCase):
        '''Tests based on https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/'''
        def test_forward(self):
            '''A test to verify correct calculation of feedforward network'''
            Theta1=np.array([[0.15,0.25],[0.2,0.3],[0.35,0.35]])
            Theta2=np.array([[0.4,0.50],[0.45,0.55],[0.6,0.6]])
            z,_,_=predict([Theta1,Theta2],[0.05,0.1])
            self.assertAlmostEqual(0.75136507,z[0],delta=0.0000001)
            self.assertAlmostEqual(0.772928465,z[1],delta=0.0000001)
            
        def test_error(self):
            self.assertAlmostEqual(0.2983171109,error([0.01,0.99],[0.75136507,0.772928465]),delta=0.0001)
            
        def test_errors_output(self):
            Theta1=np.array([[0.15,0.25],[0.2,0.3],[0.35,0.35]])
            Theta2=np.array([[0.4,0.50],[0.45,0.55],[0.6,0.6]])
            Thetas=[Theta1,Theta2]
            z,activations,Xs=predict(Thetas,[0.05,0.1]) 
            deltas=delta_weights(np.array([0.01,0.99]),z,activations,Xs,Thetas)
            
            difference2=np.subtract(Theta2,np.multiply(0.5,deltas[0]))
            self.assertAlmostEqual(0.35891648,difference2[0][0],delta=0.00001)
            self.assertAlmostEqual(0.511301270,difference2[0][1],delta=0.00001)
            self.assertAlmostEqual(0.408666186,difference2[1][0],delta=0.00001)
            self.assertAlmostEqual(0.56137012,difference2[1][1],delta=0.00001)
            
            difference1=np.subtract(Theta1,np.multiply(0.5,deltas[1]))
            self.assertAlmostEqual(0.149780716,difference1[0][0],delta=0.00001)
            self.assertAlmostEqual(0.24975114,difference1[0][1],delta=0.00001)
            self.assertAlmostEqual(0.19956143,difference1[1][0],delta=0.00001)
            self.assertAlmostEqual(0.29950229,difference1[1][1],delta=0.00001)
            

            
        def test_grad_descent(self):
            def ggen(n):
                i=0
                while i<n:
                    r=random.random()*0.01
                    if i%2==0:
                        yield np.array([0.01,0.99]),[0.05+r,0.1-r]
                    else:
                        yield np.array([0.99,0.01]),[0.1+r, 0.05-r]
                    i+=1

            print ("new")
            Theta1=np.array([[0.15,0.25],[0.2,0.3],[0.35,0.35]])
            Theta2=np.array([[0.4,0.50],[0.45,0.55],[0.6,0.6]])
            Thetas=[Theta1,Theta2]
            Thetas,_=gradient_descent(Thetas,data_source=ggen(20000))
            z,_,_=predict(Thetas,[0.05,0.1])
            print (z)
            z,_,_=predict(Thetas,[0.1,0.05])
            print (z)            
    
     
    class TestXOR(unittest.TestCase):
        '''Tests based on /'''
        def test_xor(self):
            '''A test to verify training for XOR'''
            def ggen(n):
                i=0
                while i<n:
                    r1=random.normalvariate(0,0.01)
                    r2=random.normalvariate(0,0.01)
                    if i%4==0:
                        yield np.array([0,1]),[r1,r2]
                    elif i%4==1:
                        yield np.array([1,0]),[r1,1+r2]
                    elif i%4==2:
                        yield np.array([1,0]),[1+r1,r2]                        
                    else:
                        yield np.array([0,1]),[1+r1,1+r2]
                    i+=1
            
            Thetas=create_thetas([2,5,2])
            Thetas,_=gradient_descent(Thetas,data_source=ggen(400000),n=100000,print_interval=1000)
            z,_,_=predict(Thetas,[0,0])
            print (z)
            z,_,_=predict(Thetas,[1,0])
            print (z) 
            z,_,_=predict(Thetas,[0,1])
            print (z)
            z,_,_=predict(Thetas,[1,1])
            print (z) 
            
    class  TestFiles(unittest.TestCase):
        def test1(self):
            Theta1=np.array([[0.15,0.25],[0.2,0.3],[0.35,0.35]])
            Theta2=np.array([[0.4,0.50],[0.45,0.55],[0.6,0.6]])
            Thetas=[Theta1,Theta2]
            save(Thetas)
            
    unittest.main()
