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

'''
Train network using MNIST data
'''

import backprop as bp,argparse,operator,numpy as np 
from mnist import MNIST


def create_target(label):
    '''
    Convert a target from a single digit to an array compatible with output from the network.
    E.G. 3->[0,0,0,1,0,0,0,0,0,0]
    '''    
    target=[0]*10
    target[label]=1
    return target

def scale_input(image,scale_factor=256):
    '''
    Convert grey scale image to numbers in range [0.0,1.0)
    '''
    return [ii/scale_factor for ii in image]
    
def data_gen(images,labels,randomize=True):
    '''
    Generator to supply input to bp.gradient_descent
    
        Parameters:
            images    Images from mnist
            labels    digits represents ing targets for images
            randomize Randomize order of data each iteration
    '''
    i=0
    if randomize:
        indices= np.random.permutation(len(labels))
    while i<len(labels):
        j=indices[i] if randomize else i
        yield create_target(labels[j]),scale_input(images[j])
        i+=1

def interpret(values,n_sigma=2.5):
    '''
    Interpet output from network as a digit. We take the index of the value that is maximal,
    then test to see if it is sufficiently far from the mean of all the others
    
        Parameters:
            values      Output from network
            n_sigma     Maximal value needs to be at least n_sigma standard deviations above mean
    '''
    max_index, max_value = max(enumerate(values), key=operator.itemgetter(1))
    others=[values[i] for i in range(len(values)) if i!=max_index]
    ns=(max_value - np.mean(others))/np.std(others)
    return (max_index,ns) if ns>n_sigma else (-1,ns)

def test_weights(Thetas,images,labels,n_sigma=2.5):
    '''
    Test network against the MNIST test datset to see how we are doing.
    
        Parameters:
            Thetas
            images
            labels
            n_sigma
    '''
    total_errors=0
    total_missed = 0
    n_sigmas=[]
    for i in range(len(labels)):
        target=create_target(labels[i])        
        z,_,_=bp.predict(Thetas,scale_input(images[i]))
        err=bp.error(target,z)
        dd,n_s=interpret(z,n_sigma=n_sigma)
        if dd!=labels[i]:
            total_errors+=1
        if dd==-1:
            total_missed+=1
            n_sigmas.append(n_s)
    return ('Error count ={0}, percentage ={1}. Includes {2} unrecognized {3}'.format(total_errors,
                                                                                  100*total_errors/len(labels),
                                                                                  total_missed,
                                                                                  ','.join([str(ns) for ns in n_sigmas])))

def output(i,maximum_error,average_error,Thetas,run):
    '''
    Used for progress reports during training.
    
        Parameters:
            i
            maximum_error
            average_error
            Thetas
            run    
    '''
    print ('{0} {1:9.3g}'.format(i,average_error))
    bp.save_status(i,maximum_error,average_error,run=run)
    bp.save(Thetas,run=run)
    
if __name__=='__main__':
    parser=argparse.ArgumentParser('Train NN')
    parser.add_argument('-d','--data',action='store',default='./data',
                        help='Path where data can be found')
    parser.add_argument('-n','--name',action='store',default='nn',
                        help='Name for files')
    parser.add_argument('-N','--number',action='store',type=int,default='10',
                        help='Number of iterations of training against dataset')
    parser.add_argument('-l','--layers',action='store',type=int,nargs='+',help='Number of nodes in each layer')
    parser.add_argument('-e','--eta',action='store',type=float,default=0.5,help='Learning rate')
    parser.add_argument('-a','--alpha',action='store',type=float,default=0.7,help='Momentum for training')
    parser.add_argument('-p','--print',action='store',type=int,default=1000,help='Interval for printing')
    parser.add_argument('-t','--test',action='store',type=int,default=2,
                        help='Every TEST runs execute network against test dataset')
    parser.add_argument('-s','--nsigma',action='store',type=float,default=2.5,
                        help='Interpret output if maximum is more that n_sigma standard deviations above mean of other outputs')
    parser.add_argument('-r','--randomize',action='store_true',default=False,help='Randomize order of input for each training step')
    args = parser.parse_args()
    
    
    Thetas=None
    
    try:
        with open(bp.get_status_file_name(run=args.name,ext='txt')) as status_file:
            print ('Resuming')
            saved=status_file.read().splitlines()
            eta=float(saved[1].split('=')[1])
            print_interval= int(saved[2].split('=')[1])
            alpha=float(saved[3].split('=')[1])
            n_sigma=float(saved[4].split('=')[1])
            randomize=bool(saved[5].split('=')[1])
            Thetas=bp.load(run=args.name)
            
    except FileNotFoundError:
        with open(bp.get_status_file_name(run=args.name,ext='txt'),'w') as status_file:
            print ('Starting')
            eta=args.eta
            print_interval=args.print
            alpha=args.alpha
            n_sigma=args.nsigma
            randomize=args.randomize
            
            status_file.write(','.join(str(l) for l in args.layers)+'\n')
            status_file.write('eta={0}\n'.format(eta))
            status_file.write('Interval={0}\n'.format(print_interval))
            status_file.write('alpha={0}\n'.format(alpha))           
            status_file.write('n sigma={0}\n'.format(n_sigma))
            status_file.write('randomize={0}\n'.format(randomize))
            print ('Training. Eta={0},alpha={1}'.format(args.eta,args.alpha))
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
        images_test,labels_test=mndata.load_testing()
        for i in range(args.number):
            Thetas,_=bp.gradient_descent(Thetas,
                                         data_source=data_gen(images_training,labels_training,randomize=randomize),
                                         eta=eta,
                                         alpha=alpha,
                                         print_interval=print_interval,
                                         output=lambda i,maximum_error,average_error,Thetas: output(i,maximum_error,average_error,Thetas,args.name))
            if i>0 and i%args.test==0:
                text=test_weights(Thetas,images_test,labels_test,n_sigma)
                print (text)
                bp.save_text(text,run=args.name)
                
    except FileNotFoundError as err:
        print (err)
