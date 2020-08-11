import random

def create_parameter_set(k=6,sigma=1):
    return (k,sigma,[random.gauss(0,sigma) for _ in range(k)])

def split(data):
    return ([c for c,_ in data],[x for _,x in data])

def create_data(parameter_set,n=100,sigma=1):
    def create_datum():
        i = random.randrange(k)
        return (i,random.gauss(mu[i],sigma))
    k,_,mu=parameter_set
    return split([create_datum() for _ in range(n)])
   
def cavi(xs,k=6,N=25,sigma=1):
    q_mu = [random.gauss(0,sigma) for _ in range(k)]
    q_c = [random.randrange(k) for _ in range(k)]
    for _ in range(N):
        pass
    return False

if __name__=='__main__':
    import matplotlib.pyplot as plt
    random.seed(1)
    parameter_set = create_parameter_set(sigma=100)
    print (parameter_set)
    cs,xs = create_data(parameter_set)
    plt.hist(xs)
    k,sigma,_ = parameter_set
    cavi(xs,k=k,sigma=sigma)
    plt.show()
    