#!/usr/bin/env python

# Copyright (C) 2020-2022 Greenweaves Software Limited

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from argparse import ArgumentParser
from numpy    import array, dot, exp, log

def softmax(z):
    exp_z = exp(z)
    return exp_z/exp_z.sum()

EPSILON = exp(-16)
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('exercise', type=int,choices=[1,2])
    args   = parser.parse_args()
    if args.exercise==1:
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

    if args.exercise==2:
        print ('Example 2 (WIP)')
        D = array([0.75,0.25])
        A = array([[0.8,0.2],[0.2,0.8]])
        B = array([[0,1],[1,0]])
        o = array([[1,0],[0,1]])
        print (f'D={D}')
        print (f'A={A}')
        print (f'B={B}')
        print (f'o={o}')
        s = array([[0.5,0.5],[0.5,0.5]])
        print (f's={s}')
