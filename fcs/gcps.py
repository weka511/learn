# Copyright (C) 2020 Greenweaves Software Limited

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>

# Notes on the EM Algorithm for Gaussian Mixtures: CS 274A, Probabilistic Learning 
# Padhraic Smyth 
# https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf

import argparse
import fcsparser
from matplotlib import rc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import re
from scipy import stats
import scipy.stats as stats,math
import sys
import em
import fcs
import standards


# get_boundaries

def get_boundaries(n,K=3):
    def get_segment(c):
        for numerator in range(1,K):
            if c<numerator/K:
                return numerator-1
        return K-1
    
    total      = sum(n)
    freqs      = [c/total for c in n]
    cumulative = [sum(freqs[0:i]) for i in range(len(freqs))]   
    segments   = [get_segment(c) for c in cumulative] 
    return [segments.index(k) for k in range(K)]

def get_gaussian(segment,n=100,bins=[]):
    mu      = np.mean(segment)
    sigma   = np.std(segment)
    rv      = stats.norm(loc = mu, scale = sigma)
    pdf     = rv.pdf(bins)
    max_pdf = max(pdf)
    return mu,sigma,rv,[y*n/max_pdf for y in pdf]


def get_p(x,mu=0,sigma=1):
    return (math.exp(-0.5*((x-mu)/sigma)**2)) / (math.sqrt(2*math.pi)*sigma)


# maximize_likelihood
#
# Get best GMM fit, using 
# Notes on the EM Algorithm for Gaussian Mixtures: CS 274A, Probabilistic Learning 
# Padhraic Smyth 
# https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
def maximize_likelihood(xs,mus=[],sigmas=[],alphas=[],K=3,N=25,limit=1.0e-6):

    def has_converged():
        return len(likelihoods)>1 and abs(likelihoods[-1]/likelihoods[-2]-1)<limit
    
    def get_log_likelihood(mus=[],sigmas=[],alphas=[]):
        return sum([math.log(sum([alphas[k]*get_p(xs[i],mus[k],sigmas[k]) for k in range(K)])) for i in range(len(xs))])
 
    def e_step(mus=[],sigmas=[],alphas=[]):
        ws      = [[get_p(xs[i],mus[k],sigmas[k])*alphas[k] for i in range(len(xs))] for k in range(K)] 
        Zs      = [sum([ws[k][i] for k in range(K)]) for i in range(len(xs))]
        return [[ws[k][i]/Zs[i] for i in range(len(xs))] for k in range(K)]
    
    def m_step(ws):
        N       = [sum([ws[k][i] for i in range(len(xs))] ) for k in range(K)]
        alphas  = [n/sum(N) for n in N]
        mus     = [sum([ws[k][i]*xs[i] for i in range(len(xs))] )/N[k] for k in range(K)]
        sigmas  = [math.sqrt(sum([ws[k][i]*(xs[i]-mus[k])**2 for i in range(len(xs))] )/N[k]) for k in range(K)]
        return (alphas,mus,sigmas)
    
    likelihoods=[]
    
    while len(likelihoods)<N and not has_converged():
        alphas,mus,sigmas = m_step(e_step(mus=mus,sigmas=sigmas,alphas=alphas))
        likelihoods.append(get_log_likelihood(mus=mus,sigmas=sigmas,alphas=alphas))
        
    return likelihoods,alphas,mus,sigmas

# get_selector
#
# Used to select weightiest data 
#
# Parameters:
#      ws       Data to select from
#      q        Quantile to use (default is median)
#      K        Used to select from 2D array os ws

def get_selector(ws,q=0.5,K=0): # hyperparameter
    quantile = np.quantile(ws[K], q) # hyperparameter
    return [w>quantile for w in ws[K]]    

# filter_doublets
#
# Used to split beads into two clusters, preumably real data and a set of doublets
#
# Parameters:
#     xs
#     ys
#     zs
#     mult   hyperparameter
def filter_doublets(xs,ys,zs,mult=0.25):
    mu     = [np.mean(xs),np.mean(ys),np.mean(zs)]
    sigma  = [np.std(xs),np.std(ys),np.std(zs)]
    return em.maximize_likelihood(
            xs,ys,zs,
            mus    = [[mu[i]+ direction*mult*sigma[i] for i in range(len(mu))] for direction in [-1,+1]],
            Sigmas = [np.cov([xs,ys,zs],rowvar=True),
                      np.cov([xs,ys,zs],rowvar=True)],
            alphas = [0.5,0.5]) 
     
if __name__=='__main__':    
    rc('text', usetex=True)

    parser   = argparse.ArgumentParser('Fit Gaussian Mixture Model to GCP wells')
    parser.add_argument('-r','--root',
                        default=r'\data\cytoflex\Melbourne',
                        help='Root for fcs files')
    parser.add_argument('-p','--plate',
                        default='all',
                        nargs='+',
                        help='Name of plate to be processed (omit for all)')
    parser.add_argument('-w','--well', 
                        default=['G12','H12'],
                        nargs='+',
                        help='Names of wells to be processed (omit for all)')
    parser.add_argument('-N','--N',
                        default=25, 
                        type=int, 
                        help='Number of attempts for iteration')
    parser.add_argument('-K','--K', 
                        default=[3,4],
                        nargs='+', type=int, 
                        help='Number of peaks to search for')
    parser.add_argument('-t', '--tolerance',
                        default=1.0e-6,
                        type=float, 
                        help='Iteration stops when ratio between likelihoods is this close to 1.')
    parser.add_argument('-s', '--show',
                        default=False,
                        action='store_true',
                        help='Display graphs')
    parser.add_argument('-f', '--fixup',
                        default=False,
                        action='store_true',
                        help='Decide whether to keep K=3 or 4')
    
    parser.add_argument('-d','--doublets',
                        default=False,
                        action='store_true',
                        help='Strip doublets')
    
    parser.add_argument('--properties',
                        default=r'\data\properties',
                        help='Root for properties files')
    
    args   = parser.parse_args()
    show   = args.show or args.plate!='all'
    
    Konfusion = []
    
    references = standards.create_standards(args.properties)
    
    for plate,well,df,meta,location in fcs.fcs(args.root,
                                               plate = args.plate,
                                               wells = 'controls'):                     

        df1      = df[(0               < df['FSC-H']) & \
                      (df['FSC-H']     < 1000000  )   & \
                      (0               < df['SSC-H']) & \
                      (df['SSC-H']     < 1000000)     & \
                      (df['FSC-Width'] < 2000000)]
                        
        if well in args.well:
            print (f'{plate} {well}')
            kstats = []         # Store statistics for each K so we can tune hyperparameter
            xs     = df1['FSC-H'].values
            ys     = df1['SSC-H'].values
            zs     = df1['FSC-Width'].values                             
            for K in args.K:
                plt.figure(figsize=(10,10))
                plt.suptitle(f'{plate} {well}')
                singlet = [True for i in range(len(xs))]                                
                ax1 = plt.subplot(2,2,1,projection='3d')
                if args.doublets:
                    maximized,likelihoods,ws,alphas,mus,Sigmas = filter_doublets(xs,ys,zs)             
                    
                    if maximized:
                        plt.colorbar(ax1.scatter(xs,ys,zs,
                                             s=1,
                                             c=ws[0],
                                             cmap=plt.cm.get_cmap('RdYlBu') ))
                        singlet = get_selector(ws)
                    else:
                        ax1.scatter(xs,ys,zs,s=1,c='m')
                else:
                    ax1.scatter(xs,ys,zs,s=1,c='g')
                
                ax1.set_xlabel('FSC-H')
                ax1.set_ylabel('SSC-H')
                ax1.set_zlabel('FSC-Width')  
                
                ax2             = plt.subplot(2,2,2)
                intensities     = np.log(df1['Red-H'][singlet]).values
                n,bins,_        = ax2.hist(intensities,facecolor='g',bins=100,label='From FCS')
                indices         = get_boundaries(n,K=K)
                indices.append(len(n))
                segments = [[r for r in intensities if bins[indices[k]]<r and r < bins[indices[k+1]]] 
                            for k in range(K)]
                mus      = []
                sigmas   = []
                heights      = []
                for k in range(K):
                    mu,sigma,_,y = get_gaussian(segments[k],n=max(n[i] for i in range(indices[k],indices[k+1])),bins=bins)
                    mus.append(mu)
                    sigmas.append(sigma)
                    heights.append(y)
                
                for k in range(K):    
                    if k==0:
                        ax2.plot(bins, heights[k], c='c', label='GMM')
                    else:
                        ax2.plot(bins, heights[k], c='c')
                        
                    ax2.fill_between(bins, heights[k], color='c', alpha=0.5)
        
                sums  = [sum(zz) for zz in zip(*heights)] 
                a,b = ax2.get_ylim()
                cn  = 0.5*(a+b)
                for k in range(K):
                    ax2.plot(bins, [cn*c for c in [y/z for (y,z) in zip(heights[k],sums)]],  label=f'c{k}')
                 
                ax2.set_title('Initialization')
                ax2.set_xlabel('log(Red-H)')
                ax2.set_ylabel('N')
                ax2.legend()
                
                alphas = [len(segments[k]) for k in range(K)]
                alpha_norm = sum(alphas)
                for k in range(K):
                    alphas[k] /= alpha_norm
                likelihoods,alphas,mus,sigmas =\
                    maximize_likelihood(
                        intensities,
                        mus    = mus,
                        sigmas = sigmas,
                        alphas = alphas,
                        N      = args.N,
                        limit  = args.tolerance,
                        K      = K)
                
                if K==3:
                    barcode,levels = standards.lookup(plate,references)
                    _, _, r_value, _, _ = stats.linregress(levels,[math.exp(y) for y in mus])
                    print (f'Using standard for {barcode}, r_value={r_value}')
        
                ax3 = plt.subplot(2,2,3)
                
                ax3.plot(range(len(likelihoods)),likelihoods)
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Log Likelihood')
                
                ax4 = plt.subplot(2,2,4)
                
                n,bins,_          = ax4.hist(intensities,facecolor='g',bins=100,label='From FCS')
                for k in range(K):
                    ax4.plot(bins,[max(n)*alphas[k]*get_p(x,mu=mus[k],sigma=sigmas[k]) for x in bins],
                             #c='c',
                             label=fr'$\mu=${mus[k]:.3f}, $\sigma=${sigmas[k]:.3f}')
                
                ax4.legend(framealpha=0.5)
                
                kstats.append((K,mus,sigmas))
                plt.savefig(
                    fcs.get_image_name(
                        script = os.path.basename(__file__).split('.')[0],
                        plate  = plate,
                        well   = well,
                        K      = K))
        
                print (f'K={K}, max sigma={max(sigmas)}, min dist={min([b-a for (a,b) in zip(mus[:-1],mus[1:])])}')
                if not show:
                    plt.close()
                    
            if args.fixup:   # Find optimum K
                K_preferred      = None
                sigmas_preferred = sys.float_info.max
                diffs_preferred = -1
                K_preferred_diffs = None
                for K,mus,sigmas in kstats:
                    indices = np.argsort(mus)
                    mus     = [mus[i] for i in indices]
                    diffs   = [b-a for (a,b) in zip(mus[:-1],mus[1:])]
                    sigmas = [sigmas[i] for i in indices]
        
                    if max(sigmas)<sigmas_preferred:
                        K_preferred      = K
                        sigmas_preferred = max(sigmas)
                    if min(diffs)>diffs_preferred:
                        diffs_preferred = min(diffs)
                        K_preferred_diffs = K
                if K_preferred != K_preferred_diffs:
                    Konfusion.append((plate,well))
                    break
                
                for K,_,_ in kstats:
                    file_name =  fcs.get_image_name(
                                     script = os.path.basename(__file__).split('.')[0],
                                     plate  = plate,
                                     well   = well,
                                     K      = K)
                    if K==K_preferred:
                        new_file_name = fcs.get_image_name(
                                      script = os.path.basename(__file__).split('.')[0],
                                      plate  = plate,
                                      well   = well)
                        if os.path.exists(new_file_name  ):
                            os.remove(new_file_name  )
                        os.rename(file_name, new_file_name  )
                    else:
                        os.remove(file_name )
    
    if len(Konfusion)>0:
        print ('Could not set hyperparameter K for the following wells')
        for plate,well in Konfusion:
            print (f'\t{plate} {well}')
    if show:
        plt.show()    