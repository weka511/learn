#!/usr/bin/env python

# Copyright (C) 2022 Greenweaves Software Limited
#
# Simon A. Crase -- simon@greenweaves.nz of +64 210 220 2257

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://github.com/weka511/learn/blob/master/LICENSE or
# <http://www.gnu.org/licenses/>.

'''Generate data for Gaussion Mixture Model'''

from argparse          import ArgumentParser
from gmm               import GaussionMixtureModel
from matplotlib.pyplot import figure, rcParams, show
from numpy             import add, array, exp, log, newaxis, outer, quantile, sqrt
from numpy.random      import default_rng

class Cavi:
    '''
    Perform Coordinate Ascent Mean-Field Variational Inference

    Parameters:
        K               Number of Gaussians to be fitted
        max_iterations  Maximum number of iterations -- if limit exceeded.
                        we deem cavi to have failed to converge
        tolerance       For assessing convergence
        sigma
        min_iterations  Minimum number of iterations -- don't check for convergence
                        until we have at least this many iterations
    I have borrowed some ideas from Zhiya Zuo's blog--https://zhiyzuo.github.io/VI/
    '''


    def __init__(self, K = 3):
        self.K = K

    def infer_hidden_parameters(self,x,rng = default_rng(),
                                max_iterations = 100,
                                tolerance      = 1e-6,
                                sigma          = 1,
                                min_iterations = 5):
        m      = self.init_means(x,rng)
        s2     = rng.random(self.K)    # Variance of target q(...)
        ELBOs  = []
        while (True):
            # Blei et al, equation (26)
            e_mu     = outer(x, m)                    # Expectation of mu - n x K
            e_mu2    = -0.5*(m**2 + s2)               # Expectation of mu*mu - K x 1
            phi      = exp(e_mu + e_mu2[newaxis, :])  # Unnormalized - n x K
            phi      = phi / phi.sum(1)[:, newaxis]

            # Blei et al, equation (34)
            s2       = 1/(1/sigma**2 + phi.sum(0))
            m        = (phi*x[:, newaxis]).sum(0) * s2

            ELBOs.append(self.getELBO(s2,m,sigma,x,phi))

            if len(ELBOs)> min_iterations :
                print (abs(ELBOs[-1]/ELBOs[-2]-1))
            if len(ELBOs)> min_iterations and abs(ELBOs[-1]/ELBOs[-2]-1)<tolerance:
                return (ELBOs,phi,m,sqrt(s2))
            if len(ELBOs)>max_iterations:
                raise ELBO_Error(f'ELBO has not converged to within {tolerance} after {max_iterations} iterations',ELBOs)

    def init_means(self,x,rng = default_rng()):
        '''
        Initialize means to be roughly the 'K' quantiles, plus random noise
        '''
        quantiles = [quantile(x,i/self.K) for i in range(1,self.K+1)]
        epsilon   = min([a-b for (a,b) in zip(quantiles[1:],quantiles[:-1])] )/6
        return array([q * rng.normal(loc   = 1.0,
                                     scale = epsilon) for q in quantiles])
    def getELBO(self,s2,m,sigma,x,phi):
        '''
        Calculate ELBO following Blei et al, equation (21)
        '''
        def get_sum_kK():   # First term in (21) -- sum k in 1:K
            return (log(s2) - m/sigma**2).sum()
        def get_sum_iN():   # remaining terms == sum i in 1:N
            t2  = -0.5*add.outer(x**2, s2+m**2)     # q?
            t2 += outer(x, m)
            t2 -= log(phi)                          # q?
            t2 *= phi
            return t2.sum()
        return get_sum_kK() + get_sum_iN()

class ELBO_Error(Exception):
    '''
    This class allows us to package the ELBO with an exception, e.g. for plotting
    '''
    def __init__(self,message,ELBOs):
        super().__init__(message)
        self.ELBOs = ELBOs

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--name',              default ='gmm')
    parser.add_argument('--seed',  type=int,   default=None,                        help='Seed for random number generator')
    parser.add_argument('--K',     type=int,   default=3,                           help='Number of Gaussians')
    parser.add_argument('--show',              default=False,  action='store_true', help='Controls whether plot displayed')
    parser.add_argument('--N',     type=int,   default=250,                           help='Number of Gaussians')
    parser.add_argument('--tol',     type=float,   default=1e-6,                           help='Number of Gaussians')
    parser.add_argument('--sigma',     type=float,   default=1,                           help='Number of Gaussians')
    parser.add_argument('--n',     type=int,   default=5,                           help='Number of Gaussians')
    return parser.parse_args()

if __name__=='__main__':
    rcParams.update({
        "text.usetex": True
    })

    args  = parse_args()

    model = GaussionMixtureModel(name=args.name)
    x     = model.load()
    cavi  = Cavi(K=args.K)
    fig   = figure(figsize = (10,5))
    ax1   = fig.add_subplot(2,1,1)
    ax2   = fig.add_subplot(2,1,2)
    elbo_best = - float('inf')
    phi_best = None
    m_best   = None
    s_best   = None
    for i in range(10):
        ELBOs,phi,m,s = cavi.infer_hidden_parameters(x,
                                                     max_iterations=args.N,
                                                     tolerance=args.tol,
                                                     min_iterations=args.n,
                                                     sigma=args.sigma,
                                                     rng = default_rng(args.seed))
        if ELBOs[-1]<elbo_best:
            elbo_best = - float('inf')
            phi_best = phi
            m_best   = m
            s_best   = s
        ax2.plot(ELBOs)
    ax1.hist(x,
            bins = 'sturges',
            color = 'xkcd:blue',
            label = 'x')
    ax1.vlines(model.mu,0,ax1.get_ylim()[1],
              colors    = 'xkcd:red',
              linestyles = 'dotted',
              label      = 'Means (generated)')
    ax1.vlines(m,0,ax1.get_ylim()[1],
              colors     = 'xkcd:green',
              linestyles = 'dashed',
              label      = 'Means (fitted)')
    ax1.set_title(f'Gaussian Mixture Model')
    ax1.legend()
    fig.savefig('gmm')
    if args.show:
        show()
