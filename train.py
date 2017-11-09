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

def output(i,maximum_error,average_error,Thetas,run):
    print ('{0} {1:9.3g} {2:9.3g}'.format(i,maximum_error,average_error))
    bp.save_status(i,maximum_error,average_error,run=run)
    bp.save(Thetas,run=run)
    
if __name__=='__main__':
    parser=argparse.ArgumentParser('Train NN')
    parser.add_argument('-d','--data',action='store',default='./data',help='Path where data can be found')
    parser.add_argument('-n','--name',action='store',default='nn',help='Name for files')
    parser.add_argument('-N','--number',action='store',type=int,default='10',help='Number of iterations')
    parser.add_argument('-l','--layers',action='store',type=int,nargs='+',help='Number of nodes in each layer')
    parser.add_argument('-e','--eta',action='store',type=float,default=0.2,help='Eta for training')
    parser.add_argument('-p','--print',action='store',type=int,default=1000,help='Interval for printing')
    args = parser.parse_args()
    
    Thetas=None
    
    try:
        with open(bp.get_status_file_name(run=args.name,ext='txt')) as status_file:
            print ('Resuming')
            saved=status_file.read().splitlines()
            eta=float(saved[1].split('=')[1])
            interval= int(saved[2].split('=')[1])
            Thetas=bp.load(run=args.name)
            print(Thetas)
    except FileNotFoundError:
        with open(bp.get_status_file_name(run=args.name,ext='txt'),'w') as status_file:
            print ('Starting')
            status_file.write(','.join(str(l) for l in args.layers)+'\n')
            eta=args.eta
            status_file.write('eta={0}\n'.format(eta))
            interval=args.print
            status_file.write('Interval={0}\n'.format(interval))
            print ('Training. Eta={0}'.format(args.eta))
            print ('Network has {0} layers'.format(len(args.layers)))
            for i in range(len(args.layers)):
                if i==0:
                    title='Input Layer'
                elif i==len(args.layers)-1:
                    title='Output layer'
                else:
                    title = 'Hidden Layer {0}'.format(i)
                    
                print ('{0} has {1} nodes'.format(title,args.layers[i]))            
            Thetas=bp.create_thetas(args.layers)
    try:
        print ('Reading data from {0}'.format(args.data))
        mndata=MNIST(args.data)
        images_training,labels_training=mndata.load_training()
        for i in range(args.number):
            Thetas,_=bp.gradient_descent(Thetas,
                                         data_source=data_gen(images_training,labels_training),
                                         eta=args.eta,
                                         print_interval=args.print,
                                         output=lambda i,maximum_error,average_error,Thetas: output(i,maximum_error,average_error,Thetas,args.name))
    except FileNotFoundError as err:
        print (err)
