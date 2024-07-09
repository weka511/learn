#!/usr/bin/env python

# Copyright (C) 2020-2024 Greenweaves Software Limited

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
'''
 CAVI for single Gaussian
 Coordinate Ascent Mean-Field Variational Inference (CAVI) after
 David M. Blei, Alp Kucukelbir & Jon D. McAuliffe (2017) Variational Inference: A Review for Statisticians
 and
 Coordinate Ascent Mean-field Variational Inference (Univariate Gaussian Example)
 https://suzyahyah.github.io/bayesian%20inference/machine%20learning/variational%20inference/2019/03/20/CAVI.html
'''

from argparse import ArgumentParser
from em  import maximize_likelihood
from math import sqrt, log
from matplotlib.pyplot import figure, rcParams, show
from numpy import mean, std
from numpy.random import normal
from os.path import basename, join
from random import gauss, seed
from scipy.stats import norm
from time import time

# cavi
#
# Perform Coordinate Ascent Mean-Field Variational Inference
#

def cavi(xs,
         N         = 25,
         tolerance = 1e-6):

    # calcELBO
    #
    # Calculate ELBO following Blei et al, equation (21)
    def calcELBO():
        log_p_x     = -(0.5 / sigma) * sum([(x-mu)**2 for x in xs])
        log_p_mu    = -(0.5 / sigma) * sum([(x-mu_0)**2 for x in xs])
        log_p_sigma = (-alpha_0-1) * log(sigma) - beta_0/sigma
        log_q_mu    = -(0.5 / sigma) *(1/(len(xs)+1)) * (mu - mu_n) **2
        log_q_sigma = (alpha_n-1) * log(sigma) - beta_n/sigma

        return log_p_x + log_p_mu + log_p_sigma - log_q_mu - log_q_sigma

    x_bar    = mean(xs)
    ndata    = len(xs)
    ELBOs    = [1,2]
    # hyperpriors
    mu_0     = 0
    alpha_0  = 0
    beta_0   = 0
    sigma_0  = 2

    mu_n    = mu_0
    alpha_n = alpha_0
    beta_n  = beta_0

    sigma   = sigma_0
    mu      = gauss(mu_0,sigma)

    while (abs((ELBOs[-1]-ELBOs[-2])/ELBOs[-2])>tolerance):
        mu_n    = (ndata*x_bar + mu_0)/(ndata + 1)
        mu      = mu_n
        alpha_n = alpha_0 + (ndata + 1)/2
        beta_n  = beta_0 +  0.5*sum(x**2 for x in xs) - x_bar * sum(xs)   \
                  + ((ndata+1)/2)*(sigma/ndata + x_bar**2) - mu_0*x_bar + 0.5*mu_0**2
        sigma   = sqrt((beta_n-1)/alpha_n)
        ELBOs.append(calcELBO())
        if len(ELBOs)>N:
            raise Exception(f'ELBO has not converged to within {tolerance} after {N} iterations')
    return (mu,sigma,ELBOs)

def plot_data(xs,mu,sigma,mus_em,sigmas_em,ax=None):
    def normalize(ys):
        factor = max(n)/max(ys)
        return [factor*y for y in ys]

    n,bins,_ = ax.hist(xs, bins=50,
                     label=fr'1: Data $\mu=${mean(xs):.3f}, $\sigma=${std(xs):.3f}',
                     color='b')
    bins_mid = [0.5*(bins[i-1]+bins[i]) for i in range(1,len(bins))]
    ax.plot(bins_mid,normalize([norm.pdf(x,loc=mu,scale=sigma) for x in bins_mid]),
            color = 'c',
            label = fr'2: CAVI $\mu=${mu:.3f}, $\sigma=${sigma:.3f}')
    if len(mus_em)>0:
        ax.plot(bins_mid,normalize([norm.pdf(x,loc=mus_em[0],scale=sigmas_em[0]) for x in bins_mid]),
                color = 'm',
                label = fr'3: EM $\mu$={mus_em[0]:.3f}, $\sigma=${sigmas_em[0]:.3f}')
    ax.set_title ('Data compared to CAVI and EM')

    # sort both labels and handles by labels
    # snarfed from https://stackoverflow.com/questions/22263807/how-is-order-of-items-in-matplotlib-legend-determined
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels)

def plot_ELBO(ELBOs,ax=None):
    ax.set_title ('ELBO')
    ax.plot(range(1,len(ELBOs)-1),ELBOs[2:])
    ax.set_xticks(range(1,len(ELBOs)-1))
    ax.set_ylabel('ELBO')
    ax.set_xlabel('Iteration')


if __name__=='__main__':
    rcParams.update({
        "text.usetex": True
    })

    parser = ArgumentParser(__doc__)
    parser.add_argument('--N',     type = int,            default = 5000,  help = 'Dataset size')
    parser.add_argument('--mean',  type = float,          default = 0.5,   help = 'Mean for dataset')
    parser.add_argument('--sigma', type = float,          default = 0.5,   help = 'Standard deviation')
    parser.add_argument('--seed',  type = int,            default = None,  help = 'Seed for random number generator')
    parser.add_argument('--show',  action = 'store_true', default = False, help = 'Show plots')
    parser.add_argument('--em',    action = 'store_true', default = False, help = 'Uses Expectation maximization')
    parser.add_argument('--figs', default='./figs', help='Folder to store plots')
    args = parser.parse_args()

    seed(args.seed)
    start                    = time()
    xs                       = normal(loc   = args.mean,
                                      scale = args.sigma,
                                      size  = args.N)
    mu,sigma,ELBOs           = cavi(xs,tolerance = 1e-6)
    time_cavi                = time()
    mus_em                   = []
    sigmas_em                = []
    if args.em:
        _,_,mus_em,sigmas_em = maximize_likelihood(xs,
                                                   mus    = [mean(xs)],
                                                   sigmas = [2],
                                                   alphas = [1],
                                                   K = 1)
    time_em                 = time()
    fig = figure(figsize=(10,10))
    plot_data(xs,
              mu,sigma,
              mus_em,sigmas_em,
              ax = fig.add_subplot(211))
    plot_ELBO(ELBOs,ax=fig.add_subplot(212))

    fig.savefig(join(args.figs,f'{basename(__file__).split('.')[0]}') )
    elapsed_cavi = time_cavi - start
    elapsed_em   = time_em   - time_cavi
    print (f'N={args.N}, CAVI: {elapsed_cavi:.3f} sec, EM: {(elapsed_em):.3f} sec, ratio={(elapsed_em/elapsed_cavi):.1f}')
    if args.show:
        show()
