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

'''Get best GMM fit as described in:
   Notes on the EM Algorithm for Gaussian Mixtures: CS 274A, Probabilistic Learning
   Padhraic Smyth
   https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
'''

from argparse          import ArgumentParser
from math              import log, sqrt
from scipy.stats       import norm
from math              import sqrt, log
from matplotlib.pyplot import figure, rcParams, savefig, show, subplot
from numpy             import mean, std
from numpy.random      import normal
from os.path           import basename
from random            import choice, seed
from time              import time

def maximize_likelihood(xs,
                        mus       = [],
                        sigmas    = [],
                        alphas    = [],
                        K         = 3,
                        N         = 25,
                        tolerance = 1.0e-6,
                        n_burn_in = 3):

    def has_converged():
        ''' Used to check whether likelhood has stopped improving'''
        return len(likelihoods)>n_burn_in and abs(likelihoods[-1]/likelihoods[-2]-1)<tolerance

    def get_log_likelihood(mus    = [],
                           sigmas = [],
                           alphas = []):
        '''Log of likelihood that xs were generated by current set of parameters'''
        return sum([log(sum([alphas[k]*norm.pdf(xs[i],loc=mus[k],scale=sigmas[k]) for k in range(K)])) for i in range(len(xs))])

    def e_step(mus    = [],
               sigmas = [],
               alphas = []):
        '''Perform E step from EM algorithm. The ws represent the numerators in Padhraic Smyth's formulae;
           the Zs are the denominators, which aren't they same as Smyth's zs.
           The function returns Smyth's ws.
        '''
        ws      = [[norm.pdf(xs[i],
                             loc   = mus[k],
                             scale = sigmas[k])*alphas[k] for i in range(len(xs))] for k in range(K)]
        Zs      = [sum([ws[k][i] for k in range(K)]) for i in range(len(xs))] #Normalizers
        return [[ws[k][i]/Zs[i] for i in range(len(xs))] for k in range(K)]

    def m_step(ws):
        '''Perform M step from EM algorithm: calculate new values for alphas, mus, and sigmas'''
        # Number of data points assigned to each k; since it is calculted from weights, and unrounded, the
        # values aren't exact integers.
        N       = [sum([ws[k][i] for i in range(len(xs))] ) for k in range(K)]
        # Proportion of data points assigned to each k
        alphas  = [n/sum(N) for n in N]
        mus     = [sum([ws[k][i]*xs[i] for i in range(len(xs))] )/N[k] for k in range(K)]
        sigmas  = [sqrt(sum([ws[k][i]*(xs[i]-mus[k])**2 for i in range(len(xs))] )/N[k]) for k in range(K)]
        return (alphas,mus,sigmas)

    # Use E-step and M- Step to update alphas, mus, and sigmas as long as likelihoods keep improving

    likelihoods = []

    while len(likelihoods)<N and not has_converged():
        ws                = e_step(mus    = mus,
                                   sigmas = sigmas,
                                   alphas = alphas)
        alphas,mus,sigmas = m_step(ws)
        likelihoods.append(get_log_likelihood(mus    = mus,
                                              sigmas = sigmas,
                                              alphas = alphas))
    return likelihoods,alphas,mus,sigmas

def plot_data(xs,mu,sigma,ax=None):
    def normalize(ys):
        factor = max(n)/max(ys)
        return [factor*y for y in ys]

    n,bins,_ = ax.hist(xs, bins=50,
                     label=fr'1: Data $\mu=${mean(xs):.3f}, $\sigma=${std(xs):.3f}',
                     color='b')
    bins_mid = [0.5*(bins[i-1]+bins[i]) for i in range(1,len(bins))]
    ax.plot(bins_mid,normalize([norm.pdf(x,loc=mu,scale=sigma) for x in bins_mid]),
            color = 'c',
            label = fr'2: EM $\mu=${mu[0]:.3f}, $\sigma=${sigma[0]:.3f}')

    ax.set_title ('Data compared to  EM')

    # sort both labels and handles by labels
    # snarfed from https://stackoverflow.com/questions/22263807/how-is-order-of-items-in-matplotlib-legend-determined
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels)

def plot_Likelihoods(Likelihoods,ax=None):
    ax.set_title ('Progress')
    ax.plot(Likelihoods)
    ax.set_xticks(range(1,len(Likelihoods)))
    ax.set_ylabel('log Likelihood')
    ax.set_xlabel('Iteration')

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--N',     type = int,            default = 5000,  help = 'Dataset size')
    parser.add_argument('--mean',  type = float,          default = 0.5,   help = 'Mean for dataset')
    parser.add_argument('--sigma', type = float,          default = 0.5,   help = 'Standard deviation')
    parser.add_argument('--seed',  type = int,            default = None,  help = 'Seed for random number generator')
    parser.add_argument('--show',  action = 'store_true', default = False, help = 'Show plots')
    return parser.parse_args()

if __name__=='__main__':
    rcParams.update({
        "text.usetex": True
    })

    args=parse_args()

    seed(args.seed)
    start                    = time()
    xs                       = normal(loc   = args.mean,
                                      scale = args.sigma,
                                      size  = args.N)

    Likelihoods,_,mus,sigmas = maximize_likelihood(xs,
                                                   mus    = [choice(xs)],
                                                   sigmas = [2],
                                                   alphas = [1],
                                                   K      = 1)
    figure(figsize = (10,10))
    plot_data(xs, mus, sigmas, ax = subplot(211))

    plot_Likelihoods(Likelihoods, ax = subplot(212))
    savefig(basename(__file__).split('.')[0] )

    print (f'N={args.N},  EM: {(time() - start):.3f} sec')
    if args.show:
        show()
