import backprop as bp, random
from mnist import MNIST

# http://yann.lecun.com/exdb/mnist/

mndata = MNIST(r'.\data')

images, labels = mndata.load_training()

# Hidden node computed following 
# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

Thetas=bp.Thetas=bp.create_thetas([784,37,10])

def data_gen(n):
    def data():
        i=0
        while i<n:
            j=random.randint(0,len(labels)-1)
            target=[0]*10
            target[labels[j]]=1
            yield target,[0 if ii<64 else 1 for ii in images[j]]
            i+=1
    return data
    
bp.gradient_descent(Thetas,data_source=data_gen(512),eta=0.2,n=10000,print_interval=10)