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

import backprop as bp,argparse 
from mnist import MNIST

def create_target(label):
    target=[0]*10
    target[label]=1
    return target

def data_gen(images,labels):
    i=0
    while i<len(labels):
        yield create_target(labels[i]),[0 if ii<64 else 1 for ii in images[i]]
        i+=1

def digit(l,threshold=0.5):
    for i in range(len(l)):
        if l[i]>threshold:
            return i
    return -1

def test_weights(Thetas,images,labels):
    total_errors=0
    for i in range(len(labels)):
        target=create_target(labels[i])        
        z,_,_=bp.predict(Thetas,[0 if ii<64 else 1 for ii in images[i]])
        err=bp.error(target,z)
        dd=digit(z)
        print (labels[i],dd,err)
        if dd!=labels[i]:
            total_errors+=1
    print ('Error count ={0}, percentage ={1}'.format(total_errors,100*total_errors/len(labels_test)))
    
if __name__=='__main__':
    parser=argparse.ArgumentParser('Train NN')
    parser.add_argument('-d','--data',action='store',default='./data',help='Path where data can be found')
    parser.add_argument('-n','--name',action='store',default='nn',help='Name for files')
    args = parser.parse_args()
    print ('Reading data from {0}'.format(args.data))
    mndata=MNIST(args.data)
    images_test,labels_test=mndata.load_testing()
    Thetas=bp.load(run=args.name)
    test_weights(Thetas,images_test,labels_test)