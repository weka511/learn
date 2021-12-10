#    Copyright (C) 2020-2021 Greenweaves Software Limited

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

#  Coordinate Ascent Mean-Field Variational Inference (CAVI) after
#  David M. Blei, Alp Kucukelbir & Jon D. McAuliffe (2017) Variational Inference: A Review for Statisticians

from argparse          import ArgumentParser
from math              import sqrt
from matplotlib.pyplot import figure, hist, rcParams, savefig, show, subplot
import numpy as np
from numpy.random      import dirichlet, normal, random, randint
from os.path           import basename
from random            import gauss, randrange, seed
from scipy.stats       import norm

# ELBO_Error
#
# This class ellows us to package the ELBO with an exception,
# e.g. for plotting

class ELBO_Error(Exception):
    def __init__(self,message,ELBOs):
        super().__init__(message)
        self.ELBOs = ELBOs

# create_data
#
# Create test data following Gaussian Mixture Model

def create_data(mu,n=1000,sigmas=[]):
    def create_datum():
        i = randrange(len(mu))
        return (i,gauss(mu[i],sigmas[i]))
    return list(zip(*[create_datum() for _ in range(n)]))

# cavi
#
# Perform Coordinate Ascent Mean-Field Variational Inference
#
# I have borrowed some ideas from Zhiya Zuo's blog--https://zhiyzuo.github.io/VI/
def cavi(x,K=3,N=25,tolerance=1e-12,sigma=1,min_iterations=5):

    # init_means
    #
    # Initialize means to be roughly the 'K' quantiles, plus random noise
    def init_means():
        quantiles = [np.quantile(x,q/K) for q in range(1,K+1)]
        epsilon   = min([a-b for (a,b) in zip(quantiles[1:],quantiles[:-1])] )/6
        return np.array([q * normal(loc=1.0,scale=epsilon) for q in quantiles])

    # getELBO
    #
    # Calculate ELBO following Blei et al, equation (21)
    def getELBO():
        t1  = np.log(s2) - m/sigma2
        t1  = t1.sum()
        t2  = -0.5*np.add.outer(x**2, s2+m**2)
        t2 += np.outer(x, m)
        t2 -= np.log(phi)
        t2 *= phi
        t2  = t2.sum()
        return t1 + t2 #log_p_x + log_p_mu + log_p_sigma - log_q_mu - log_q_sigma

    phi    = dirichlet([random()*randint(1, 10)]*K, len(x))
    m      = init_means()
    s2     = random(K)
    sigma2 = sigma**2
    ELBOs  = []

    # Perform update -- Blei et al, section 3.1
    while (len(ELBOs)<min_iterations or abs(ELBOs[-1]/ELBOs[-2]-1)>tolerance):
        # Blei et al, equation (26)
        e_mu     = np.outer(x, m)                       # Expectation of mu - n x K
        e_mu2    = -0.5*(m**2 + s2)                     # Expectation of mu*mu - K x 1
        phi      = np.exp(e_mu + e_mu2[np.newaxis, :])  # Unnormalized - n x K
        phi      = phi / phi.sum(1)[:, np.newaxis]

        # Blei et al, equation (34)
        s2       = 1/(1/sigma2 + phi.sum(0))
        m        = (phi*x[:, np.newaxis]).sum(0) * s2

        ELBOs.append(getELBO())
        if len(ELBOs)>N:
            raise ELBO_Error(f'ELBO has not converged to within {tolerance} after {N} iterations',ELBOs)

    return (ELBOs,phi,m,[sqrt(s) for s in s2])

# plot_data
#
# Plot raw data and estimates of sufficient statistics
def plot_data(xs,
              cs,
              mu      = [],
              sigmas  = [],
              m       = 0,
              s       = 1,
              nbins   = 25,
              colours = ['r','g','b', 'c', 'm', 'y'],
              ax      = None):
    def sort_stats():
        indices = np.argsort(m)
        return ([m[i] for i in indices], [s[i] for i in indices])

    ax.hist(xs,
             bins  = nbins,
             alpha = 0.5)

    m,s = sort_stats()

    for i in range(len(mu)):
        x0s           = [xs[j] for j in range(len(xs)) if cs[j]==i ]
        n,bins,_      = hist(x0s,
                            bins      = 25,
                            alpha     = 0.5,
                            facecolor = colours[i])
        x_values      = np.arange(min(xs), max(xs), 0.1)
        y_values      = norm(mu[i], sigmas[i])
        y_values_cavi = norm(m[i], s[i])
        ys            = y_values.pdf(x_values)
        ys_cavi       = y_values_cavi.pdf(x_values)
        ax.plot(x_values,
                 [y*max(n)/max(ys) for y in ys],
                 c = colours[i],
                 label     = fr'$\mu=${mu[i]:.3f}, $\sigma=${sigmas[i]:.1f}')

        ax.plot(x_values,
                [y*max(n)/max(ys_cavi) for y in ys_cavi],
                label     = fr'$\mu=${m[i]:.3f}, $\sigma=${s[i]:.3f} (CAVI)',
                c = colours[i],
                linestyle='--')

    ax.legend()
    ax.set_title('Coordinate Ascent Mean-Field Variational Inference (CAVI)')
    ax.set_ylabel('N')

# plotELBO
#
# Illustrate progress
def plotELBO(ELBOs,
             ax    = None,
             title = 'Convergence'):
    # get_tick_freq
    #
    # Used to avoid cluttering x axis if calculation has gone on for too many iterations
    def get_tick_freq(maximum_ticks=25, modulus=10):
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

if __name__=='__main__':
    rcParams.update({
        "text.usetex": True
    })
    parser = ArgumentParser('The Coordinate Ascent Mean-Field Variational Inference (CAVI) example from Section 3 of Blei et al')
    parser.add_argument('--K',         type=int,   default=2,      help='Number of Gaussians')
    parser.add_argument('--n',         type=int,   default=1000,   help='Number of points')
    parser.add_argument('--N',         type=int,   default=250,    help='Number of iterations')
    parser.add_argument('--tolerance', type=float, default=1.0e-6, help='Convergence criterion')
    parser.add_argument('--seed',      type=int,   default=None,   help='Seed for random number generator')
    args = parser.parse_args()

    seed(args.seed)

    sigma         = 1
    sigmas        = [1.0]*args.K
    mu            = sorted(np.random.uniform(low  = 0,
                                             high = 5,
                                             size = args.K))
    cs,xs         = create_data(mu,sigmas=sigmas,n=args.n)

    try:
        ELBOs,phi,m,s = cavi(x         = np.asarray(xs),
                             K         = args.K,
                             N         = args.N,
                             tolerance = args.tolerance)

        figure(figsize=(10,10))
        plot_data(xs,cs,
                  mu     = mu,
                  sigmas = sigmas,
                  m      = m,
                  s      = s,
                  ax     = subplot(2,1,1))

        plotELBO(ELBOs,
                 ax = subplot(2,1,2))

    except ELBO_Error as e:
        print (e)
        figure(figsize=(10,10))
        plotELBO(e.ELBOs,
                 ax    = subplot(1,1,1),
                 title = str(e))

    savefig(basename(__file__).split('.')[0] )
    show()
