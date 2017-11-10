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

import backprop as bp,argparse,train as tr,operator,numpy as np 
from mnist import MNIST

def interpret(values,n_sigma=2.0):
    max_index, max_value = max(enumerate(values), key=operator.itemgetter(1))
    others=[values[i] for i in range(len(values)) if i!=max_index]
    return max_index if max_value > np.mean(others) + n_sigma*np.std(others) else -1

def test_weights(Thetas,images,labels):
    total_errors=0
    total_missed = 0
    for i in range(len(labels)):
        target=tr.create_target(labels[i])        
        z,_,_=bp.predict(Thetas,tr.scale_input(images[i]))
        err=bp.error(target,z)
        dd=interpret(z)
        if dd!=labels[i]:
            total_errors+=1
        if dd==-1:
            total_missed+=1
    print ('Error count ={0}, percentage ={1}. Includes {2} unrecognized'.format(total_errors,100*total_errors/len(labels_test),total_missed))
    
if __name__=='__main__':
    parser=argparse.ArgumentParser('Train NN')
    parser.add_argument('-d','--data',action='store',default='./data',help='Path where data can be found')
    parser.add_argument('-n','--name',action='store',default='nn',help='Name for files')
    args = parser.parse_args()
    mndata=MNIST(args.data)
    images_test,labels_test=mndata.load_testing()
    Thetas=bp.load(run=args.name)
    test_weights(Thetas,images_test,labels_test)