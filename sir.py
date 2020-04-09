import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

k = 1.1
l = 0.1

def sir(y,t,k,l):
    x_,y_ = y
    return [-k*x_*y_,k*x_*y_ - l*y_]

t = np.linspace(0, 100, 101)
y0 = [1,0.00001]

sol = odeint(sir, y0, t, args=(k, l))

plt.figure(figsize=(20,6))
plt.plot(t, sol[:, 0], 'b', label='S')
plt.plot(t, sol[:, 1], 'g', label='I')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()