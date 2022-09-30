#!/usr/bin/env python

#    Copyright (C) 2020-2022 Greenweaves Software Limited

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''
   The Coordinate Ascent Mean-Field Variational Inference (CAVI) example from Section 3 of
   David M. Blei, Alp Kucukelbir & Jon D. McAuliffe (2017) Variational Inference: A Review for Statisticians
'''

from argparse          import ArgumentParser
from math              import sqrt
from matplotlib.pyplot import figure, rcParams, show
from numpy             import add, arange, argsort, array, asarray, exp, log, newaxis, outer, quantile, random
from numpy.random      import dirichlet, normal, random, randint, uniform
from os.path           import basename
from random            import gauss, randrange, seed
from re                import split
from scipy.stats       import norm


class ELBO_Error(Exception):
    '''
    This class allows us to package the ELBO with an exception, e.g. for plotting
    '''
    def __init__(self,message,ELBOs):
        super().__init__(message)
        self.ELBOs = ELBOs


def create_data(mu,
                n      = 1000,
                sigmas = []):
    '''
    Create test data following Gaussian Mixture Model
    '''
    def create_datum():
        i = randrange(len(mu))
        return (i,gauss(mu[i],sigmas[i]))
    return list(zip(*[create_datum() for _ in range(n)]))

def getELBO(s2,m,sigma,x,phi):
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
    return get_sum_kK() + get_sum_iN() #log_p_x + log_p_mu + log_p_sigma - log_q_mu - log_q_sigma

def init_means(x,K):
    '''
    Initialize means to be roughly the 'K' quantiles, plus random noise
    '''
    quantiles = [quantile(x,i/K) for i in range(1,K+1)]
    epsilon   = min([a-b for (a,b) in zip(quantiles[1:],quantiles[:-1])] )/6
    return array([q * normal(loc   = 1.0,
                             scale = epsilon) for q in quantiles])

def cavi(x,
         K              = 3,
         max_iterations = 25,
         tolerance      = 1e-12,
         sigma          = 1,
         min_iterations = 5):
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

    m      = init_means(x,K)
    s2     = random(K)    # Variance of target q(...)
    ELBOs  = []

    # Perform update -- Blei et al, section 3.1
    while (True):
        # Blei et al, equation (26)
        e_mu     = outer(x, m)                    # Expectation of mu - n x K
        e_mu2    = -0.5*(m**2 + s2)               # Expectation of mu*mu - K x 1
        phi      = exp(e_mu + e_mu2[newaxis, :])  # Unnormalized - n x K
        phi      = phi / phi.sum(1)[:, newaxis]

        # Blei et al, equation (34)
        s2       = 1/(1/sigma**2 + phi.sum(0))
        m        = (phi*x[:, newaxis]).sum(0) * s2

        ELBOs.append(getELBO(s2,m,sigma,x,phi))

        if len(ELBOs)> min_iterations and abs(ELBOs[-1]/ELBOs[-2]-1)<tolerance:
            return (ELBOs,phi,m,[sqrt(s) for s in s2])
        if len(ELBOs)>max_iterations:
            raise ELBO_Error(f'ELBO has not converged to within {tolerance} after {max_iterations} iterations',ELBOs)

def plot_data(xs,
              cs,
              mu      = [],
              sigmas  = [],
              m       = 0,
              s       = [1],
              nbins   = 25,
              colours = ['r', 'g', 'b'],
              ax      = None):
    '''
    Plot raw data and estimates of sufficient statistics
    '''
    def sort_stats(m,s):
        '''Reorder both m and s so that m is ascending '''
        indices = argsort(m)
        return ([m[i] for i in indices], [s[i] for i in indices])

    ax.hist(xs,
             bins  = nbins,
             alpha = 0.5)

    m,s = sort_stats(m,s)

    for i in range(len(mu)):
        x0s           = [xs[j] for j in range(len(xs)) if cs[j]==i ]
        n,bins,_      = ax.hist(x0s,
                            bins      = 25,
                            alpha     = 0.5,
                            facecolor = colours[i])
        x_values      = arange(min(xs), max(xs), 0.1)
        y_values      = norm(mu[i], sigmas[i])
        y_values_cavi = norm(m[i], s[i])
        ys            = y_values.pdf(x_values)
        ys_cavi       = y_values_cavi.pdf(x_values)
        ax.plot(x_values,
                 [y*max(n)/max(ys) for y in ys],
                 c     = colours[i],
                 label = fr'$\mu=${mu[i]:.3f}, $\sigma=${sigmas[i]:.1f}')

        ax.plot(x_values,
                [y*max(n)/max(ys_cavi) for y in ys_cavi],
                label     = fr'$\mu=${m[i]:.3f}, $\sigma=${s[i]:.3f} (CAVI)',
                c         = colours[i],
                linestyle = '--')

    ax.legend()
    ax.set_title('Coordinate Ascent Mean-Field Variational Inference (CAVI)')
    ax.set_ylabel('N')

def plotELBO(ELBOs,
             ax    = None,
             title = 'Convergence'):
    '''
    Illustrate progress
    '''
    def get_tick_freq(maximum_ticks=25, modulus=10):
        '''
        Used to avoid cluttering x axis if calculation has gone on for too many iterations
        '''
        freq   = 1
        nticks = len(ELBOs)
        if nticks<= maximum_ticks: return nticks
        while nticks>maximum_ticks:
            freq   *= modulus
            nticks /= modulus
        return freq

    ax.plot(ELBOs,label='ELBO')
    ax.set_xlim(0,len(ELBOs))
    ax.set_xticks(range(0,len(ELBOs),get_tick_freq()))
    ax.set_ylim(min(ELBOs),max(ELBOs)+1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$\log(P)$')
    ax.set_title(title)
    ax.legend()

def create_xkcd_colours(file_name = 'rgb.txt',
                        prefix    = 'xkcd:',
                        filter    = lambda R,G,B:True):
    '''  Create list of XKCD colours
         Keyword Parameters:
            file_name Where XKCD colours live
            prefix    Use to prefix each colour with "xkcd:"
            filter    Allows us to exclude some colours based on RGB values
    '''
    with open(file_name) as colours:
        for row in colours:
            parts = split(r'\s+#',row.strip())
            if len(parts)>1:
                rgb  = int(parts[1],16)
                B    = rgb%256
                rest = (rgb-B)//256
                G    = rest%256
                R    = (rest-G)//256
                if filter(R,G,B):
                    yield f'{prefix}{parts[0]}'

def parse_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--K',         type=int,   default=2,                           help='Number of Gaussians')
    parser.add_argument('--n',         type=int,   default=1000,                        help='Number of points')
    parser.add_argument('--N',         type=int,   default=250,                         help='Number of iterations')
    parser.add_argument('--tolerance', type=float, default=1.0e-6,                      help='Convergence criterion')
    parser.add_argument('--seed',      type=int,   default=None,                        help='Seed for random number generator')
    parser.add_argument('--show',                  default=False,  action='store_true', help='Controls whether plot displayed')
    return parser.parse_args()

if __name__=='__main__':
    rcParams.update({
        "text.usetex": True
    })

    args = parse_args()
    seed(args.seed)

    sigmas = [1.0]*args.K
    mu     = sorted(uniform(low  = 0,
                            high = 5,
                            size = args.K))
    cs,xs = create_data(mu,
                        sigmas = sigmas,
                        n      = args.n)
    fig = figure(figsize = (10,10))
    try:
        ELBOs,phi,m,s = cavi(x              = asarray(xs),
                             K              = args.K,
                             max_iterations = args.N,
                             tolerance      = args.tolerance)

        plot_data(xs,cs,
                  mu      = mu,
                  sigmas  = sigmas,
                  m       = m,
                  s       = s,
                  ax      = fig.add_subplot(2,1,1),
                  colours = [colour for colour in create_xkcd_colours(filter = lambda R,G,B:R<192 and max(R,G,B)>32)][::-1])

        plotELBO(ELBOs,
                 ax = fig.add_subplot(2,1,2))

    except ELBO_Error as e:
        print (e)
        figure(figsize=(10,10))
        plotELBO(e.ELBOs,
                 ax    = fig.add_subplot(1,1,1),
                 title = str(e))

    fig.savefig(basename(__file__).split('.')[0] )

    if args.show:
        show()
