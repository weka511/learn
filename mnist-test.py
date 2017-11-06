import backprop as bp, random
from mnist import MNIST

# http://yann.lecun.com/exdb/mnist/
def digit(l,threshold=0.9):
        for i in range(len(l)):
                if l[i]>threshold:
                        return i
                
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
    
Thetas,_=bp.gradient_descent(Thetas,data_source=data_gen(),eta=0.2,print_interval=1000)

images_test, labels_test = mndata.load_testing()


                

for i in range(len(labels_test)):
        total_errors=0
        target=[0]*10
        target[labels_test[i]]=1        
        z,_,_=bp.predict(Thetas,[0 if ii<64 else 1 for ii in images_test[i]])
        err=bp.error(target,z)
        dd=digit(z)
        print (labels_test[i],dd,err)
        if dd!=labels_test[i]:
                total_errors+=1
print ('Error count ={0}'.format(total_errors))