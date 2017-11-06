import backprop as bp, random
from mnist import MNIST

mndata = MNIST(r'.\data')

images, labels = mndata.load_training()

Thetas=bp.Thetas=bp.create_thetas([784,80,10])

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
    
bp.gradient_descent(Thetas,data_source=data_gen(12),n=1000,print_interval=10)