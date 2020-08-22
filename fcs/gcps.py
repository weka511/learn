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

import fcsparser, matplotlib.pyplot as plt,numpy as np,scipy.stats as stats,math

# get_well_name
#
# Extract well name from tube name

def get_well_name(tbnm):
    return tbnm[-3:]

#  get_bounds
#
# Used to clip data into band near to mean

def get_bounds(df,channel,nsigma=3):
    mean   = np.mean(df[channel])
    std    = np.std(df[channel])
    return (mean-nsigma*std,mean+nsigma*std)

# gate_data
#
# Used to clip data

def gate_data(df,nsigma=3,nw=2):
    fsc_min,fsc_max = get_bounds(df,'FSC-H',nsigma=nsigma)
    ssc_min,ssc_max = get_bounds(df,'SSC-H',nsigma=nsigma)
    _,fsc_width_max = get_bounds(df,'FSC-Width',nsigma=nw)
    
    return df[(fsc_min         < df['FSC-H']) & \
              (df['FSC-H']     < fsc_max)     & \
              (ssc_min         < df['SSC-H']) & \
              (df['SSC-H']     < ssc_max)     & \
              (df['FSC-Width'] < fsc_width_max)]

# purge_outliers
#

def purge_outliers(df,nsigma=3,nw=2,max_iterations=float('inf')):
    nr0,_ = df.shape
    df1   = gate_data(df,nsigma=nsigma)
    nr1,_ = df1.shape
    i     = 0
    while nr1<nr0 and i<max_iterations:
        nr0    = nr1
        df1    = gate_data(df1,nsigma=nsigma,nw=nw)
        nr1,_  = df1.shape
        i     += 1
    return df1

# get_boundaries

def get_boundaries(bins,n,K=3):
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
 
def get_image_name(script=None,plate=None,well=None,K=None):
    return os.path.join('figs',
                        f'{script}-{plate}-{well}' if K==None else f'{script}-{plate}-{well}-{K}')

if __name__=='__main__':
    import os, re, argparse
    from matplotlib import rc
    rc('text', usetex=True)

    parser   = argparse.ArgumentParser('Fit Gaussian mixture model to GCP wells')
    parser.add_argument('-r','--root',       default=r'\data\cytoflex\Melbourne', help='Root for fcs files')
    parser.add_argument('-p','--plate',      default='all',          nargs='+', help='Name of plate to be processed')
    parser.add_argument('-w','--well',       default=['G12','H12'],  nargs='+', help='Names of wells to be processed')
    parser.add_argument('-N','--N',          default=25,             type=int, help='Number of attempts for iteration')
    parser.add_argument('-K','--K',          default=3,              type=int, help='Number of peaks to search for')
    parser.add_argument('-t', '--tolerance', default=1.0e-6,         type=float, 
                        help='Iteration stops when ratio between likelihoods is this close to 1.')
    parser.add_argument('-s', '--show',      default=False,          action='store_true', help='Display graphs')
    
    args   = parser.parse_args()
    show   = args.show or args.plate!='all'
    for root, dirs, files in os.walk(args.root):
        path  = root.split(os.sep)
        match = re.match('.*(((PAP)|(RTI))[A-Z]*[0-9]+[r]?)',path[-1])
        if match:
            plate = match.group(1)
            if args.plate=='all' or plate in args.plate:
                for file in files:
                    if re.match('.*[GH]12.fcs',file):
                        path     = os.path.join(root,file)
                        meta, df = fcsparser.parse(path, reformat_meta=True)
                        
                        date     = meta['$DATE']
                        tbnm     = meta['TBNM']
                        btim     = meta['$BTIM']
                        etim     = meta['$ETIM']
                        cyt      = meta['$CYT']
                        df1      = purge_outliers(df)
                        well     = get_well_name(tbnm)
 
                        if well in args.well:
                            plt.figure(figsize=(10,10))
                            plt.suptitle(f'{plate} {well}')
                            
                            ax1 = plt.subplot(2,2,1)
                            ax1.scatter(df1['FSC-H'],df1['SSC-H'],s=1,c='g')
                            ax1.set_xlabel('FSC-H')
                            ax1.set_ylabel('SSC-H')
                            
                            ax2             = plt.subplot(2,2,2)
                            intensities     = np.log(df1['Red-H']).values
                            n,bins,_        = ax2.hist(intensities,facecolor='g',bins=100,label='From FCS')
                            indices         = get_boundaries(bins,n,K=args.K)
                            indices.append(len(n))
                            segments = [[r for r in intensities if bins[indices[k]]<r and r < bins[indices[k+1]]] 
                                        for k in range(args.K)]
                            mus      = []
                            sigmas   = []
                            ys       = []
                            for k in range(args.K):
                                mu,sigma,_,y = get_gaussian(segments[k],n=max(n[i] for i in range(indices[k],indices[k+1])),bins=bins)
                                mus.append(mu)
                                sigmas.append(sigma)
                                ys.append(y)
                            
                            for k in range(args.K):    
                                if k==0:
                                    ax2.plot(bins, ys[k], c='c', label='GMM')
                                else:
                                    ax2.plot(bins, ys[k], c='c')
                                ax2.fill_between(bins, ys[k], color='c', alpha=0.5)
   
                            zs  = [sum(zz) for zz in zip(*ys)] 
                            a,b = ax2.get_ylim()
                            cn  = 0.5*(a+b)
                            for k in range(args.K):
                                ax2.plot(bins, [cn*c for c in [y/z for (y,z) in zip(ys[k],zs)]],  label=f'c{k}')
                             
                            ax2.set_title('Initialization')
                            ax2.set_xlabel('log(Red-H)')
                            ax2.set_ylabel('N')
                            ax2.legend()
                            
                            alphas = [len(segments[k]) for k in range(args.K)]
                            alpha_norm = sum(alphas)
                            for k in range(args.K):
                                alphas[k] /= alpha_norm
                            likelihoods,alphas,mus,sigmas =\
                                maximize_likelihood(
                                    intensities,
                                    mus    = mus,
                                    sigmas = sigmas,
                                    alphas = alphas,
                                    N      = args.N,
                                    limit  = args.tolerance,
                                    K      = args.K)
                            
                            ax3 = plt.subplot(2,2,3)
                            
                            ax3.plot(range(len(likelihoods)),likelihoods)
                            ax3.set_xlabel('Iteration')
                            ax3.set_ylabel('Log Likelihood')
                            
                            ax4 = plt.subplot(2,2,4)
                            
                            n,bins,_          = ax4.hist(intensities,facecolor='g',bins=100,label='From FCS')
                            for k in range(args.K):
                                ax4.plot(bins,[max(n)*alphas[k]*get_p(x,mu=mus[k],sigma=sigmas[k]) for x in bins],
                                         #c='c',
                                         label=fr'$\mu=${mus[k]:.3f}, $\sigma=${sigmas[k]:.3f}')
                            
                            ax4.legend(framealpha=0.5)
                            
                            plt.savefig(
                                get_image_name(
                                    script = os.path.basename(__file__).split('.')[0],
                                    plate  = plate,
                                    well   = well,
                                    K      = args.K))
                            
                            if not show:
                                plt.close()
                       
    if show:
        plt.show()    