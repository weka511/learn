#!/usr/bin/env python

# Copyright (C) 2020-2025 Simon Crase  simon@greenweaves.nz

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>.
'''

 CAVI for single Gaussian
 Coordinate Ascent Mean-Field Variational Inference (CAVI) after
 David M. Blei, Alp Kucukelbir & Jon D. McAuliffe (2017) Variational Inference: A Review for Statisticians
 and
 Coordinate Ascent Mean-field Variational Inference (Univariate Gaussian Example)
 https://suzyahyah.github.io/bayesian%20inference/machine%20learning/variational%20inference/2019/03/20/CAVI.html
'''

from argparse import ArgumentParser
from os.path import basename, join
from time import time
from matplotlib.pyplot import figure, rcParams, show
import numpy as np
from scipy.stats import norm
from em import maximize_likelihood


def cavi(xs, n=100, m=5,rtol=1e-8, rng=np.random.default_rng(),
         mu_0 = 10, alpha_0 = 0, beta_0 = 0, sigma_0 = 20):
    '''
    Perform Coordinate Ascent Mean-Field Variational Inference

    Parameters:
        xs       Data to be fitted
        n        Maximum number of iterations
        m        Minimum number of iterations (before we start testing value)
        rtol     Relative tolerance: stop iterating if improvement less than this value
        rng      Random number generator, used for initialization
        mu_0     hyperprior: initial value for mu
        alpha_0  hyperprior: initial value for alpha
        beta_0   hyperprior: initial value for beta
        sigma_0  hyperprior: initial value for sigma
    '''

    def get_ELBO():
        '''
        Calculate ELBO following Blei et al, equation (21)
        '''
        log_p_x = -(0.5 / sigma) * np.sum((xs - mu)**2)
        log_p_mu = -(0.5 / sigma) * np.sum((xs - mu_0)**2)
        log_p_sigma = (-alpha_0 - 1) * np.log(sigma) - beta_0 / sigma
        log_q_mu = -(0.5 / sigma) * (1 / (len(xs) + 1)) * (mu - mu_n) ** 2
        log_q_sigma = (alpha_n - 1) * np.log(sigma) - beta_n / sigma

        return log_p_x + log_p_mu + log_p_sigma - log_q_mu - log_q_sigma

    x_bar = np.mean(xs)
    ndata = len(xs)
    ELBOs = np.empty((n))
    mu_n = mu_0
    alpha_n = alpha_0
    beta_n = beta_0

    sigma = sigma_0
    mu = rng.normal(mu_0, scale=sigma)

    for i in range(n):
        mu_n = (ndata * x_bar + mu_0) / (ndata + 1)
        mu = mu_n
        alpha_n = alpha_0 + (ndata + 1) / 2
        beta_n = (beta_0 + 0.5 * np.sum(xs**2) - x_bar * np.sum(xs)
                  + ((ndata + 1) / 2) * (sigma / ndata + x_bar**2) - mu_0 * x_bar + 0.5 * mu_0**2)
        sigma = np.sqrt((beta_n - 1) / alpha_n)
        ELBOs[i] = get_ELBO()
        if i > m and (abs((ELBOs[i] - ELBOs[i-1]) / ELBOs[i-1]) < rtol):
            return (mu, sigma, np.resize(ELBOs,i))

    raise Exception(f'ELBO has not converged to within {rtol} after {n} iterations')

def sort_handles_and_labels(ax):
    '''
    sort both labels and handles by labels
    snarfed from https://stackoverflow.com/questions/22263807/how-is-order-of-items-in-matplotlib-legend-determined
    '''
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    return labels,handles

def plot_data(xs, mu, sigma, mus_em, sigmas_em, ax=None):
    def normalize(ys,y_scale):
        '''
        Used to plot data using the same scale as some other dataset

        Parameters:
            ys          Data to be normalized
            y_other     The data that sets the scale
        '''
        return (max(y_scale) / ys.max())*ys

    n, bins, _ = ax.hist(xs, bins=50,color='b',label=fr'1: Data $\mu=${np.mean(xs):.3f}, $\sigma=${np.std(xs):.3f}')

    midpoints = 0.5 * (bins[:-1] + bins[1:])
    ax.plot(midpoints, normalize(norm.pdf(midpoints, loc=mu, scale=sigma),n),
            color='c',label=fr'2: CAVI $\mu=${mu:.3f}, $\sigma=${sigma:.3f}')

    if len(mus_em) > 0:
        ax.plot(midpoints, normalize(norm.pdf(midpoints, loc=mus_em[0], scale=sigmas_em[0]),n),
                color='m',label=fr'3: EM $\mu$={mus_em[0]:.3f}, $\sigma=${sigmas_em[0]:.3f}')
    ax.set_title('Data compared to CAVI and EM')

    labels,handles = sort_handles_and_labels(ax)
    ax.legend(handles, labels)


def plot_ELBO(ELBOs, ax=None):
    ax.set_title('Evidence Lower Bound (ELBO)')
    ax.plot(list(range(len(ELBOs))),ELBOs)
    ax.set_xticks(list(range(len(ELBOs))))
    ax.set_ylabel('ELBO')
    ax.set_xlabel('Iteration')

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--N', type=int, default=5000, help='Dataset size')
    parser.add_argument('--n', type=int, default=50, help='Maximum number of iterations')
    parser.add_argument('--m', type=int, default=3, help='Minimum number of iterations (before we start testing value)')
    parser.add_argument('--mean', type=float, default=0.5, help='Mean for dataset')
    parser.add_argument('--rtol', type=float, default=1e-8, help='Relative tolerance: stop iterating if improvement less than this value')
    parser.add_argument('--sigma', type=float, default=0.5, help='Standard deviation')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')
    parser.add_argument('--show', action='store_true', default=False, help='Show plots')
    parser.add_argument('--figs', default='./figs', help='Folder to store plots')
    return parser.parse_args()

if __name__ == '__main__':
    rcParams.update({
        "text.usetex": True
    })

    args = parse_args()
    rng = np.random.default_rng(args.seed)
    start = time()
    xs = rng.normal(loc=args.mean, scale=args.sigma, size=args.N)
    mu, sigma, ELBOs = cavi(xs, m=args.m, n=args.n, rtol=args.rtol, rng=rng)
    time_cavi = time()
    _, _, mu_em, sigma_em = maximize_likelihood(xs,mu=[np.mean(xs)],sigma=[2],alpha=[1],K=1)
    time_em = time()
    fig = figure(figsize=(10, 10))
    plot_data(xs, mu, sigma, mu_em, sigma_em, ax=fig.add_subplot(211))
    plot_ELBO(ELBOs, ax=fig.add_subplot(212))

    fig.savefig(join(args.figs, f'{basename(__file__).split('.')[0]}'))
    elapsed_cavi = time_cavi - start
    elapsed_em = time_em - time_cavi
    print(f'N={args.N}, CAVI: {elapsed_cavi:.3f} sec, EM: {(elapsed_em):.3f} sec, ratio={(elapsed_em / elapsed_cavi):.1f}')
    if args.show:
        show()
