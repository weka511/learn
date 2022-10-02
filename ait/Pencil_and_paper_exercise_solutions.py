#!/usr/bin/env python

from numpy import array, exp, log, dot

def softmax(z):
    exp_z = exp(z)
    return exp_z/exp_z.sum()

print ('Example 1')
D = array([0.5, 0.5])

A = array([[0.9, 0.3],
           [0.1, 0.7]
           ])

o = array([1,0])
print (f'D={D}')
print (f'A={A}')
print (f'o={o}')
s = softmax(log(D) + dot(log(A.transpose()),o))
print (f's={s}')

print ('Exercise 1')
D = array([0.75, 0.25])

A = array([[0.8, 0.2],
           [0.2, 0.8]
           ])

o = array([1,0])
print (f'D={D}')
print (f'A={A}')
print (f'o={o}')
s = softmax(log(D) + dot(log(A.transpose()),o))
print (f's={s}')
