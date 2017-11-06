import backprop as bp, random
from mnist import MNIST

# http://yann.lecun.com/exdb/mnist/

mndata = MNIST(r'.\data')

images, labels = mndata.load_training()

# Hidden node computed following 
# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

Thetas=bp.Thetas=bp.create_thetas([784,37,10])

def data_gen():
        i=0
        while i<len(labels):
                target=[0]*10
                target[labels[i]]=1
                yield target,[0 if ii<64 else 1 for ii in images[i]]
                i+=1
    
bp.gradient_descent(Thetas,data_source=data_gen(),eta=0.2)