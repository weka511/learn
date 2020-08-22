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

def get_boundaries(bins,n):
    def get_segment(c):
        if c<1/3:
            return 0
        elif c<2/3:
            return 1
        else:
            return 2
    
    freqs     = [c/sum(n) for c in n]
    counts    = [sum(freqs[0:i]) for i in range(len(freqs))]   
    segments  = [get_segment(c) for c in counts] 
    i1        = segments.index(1)
    i2        = segments.index(2)
    return (i1,i2)

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
        
if __name__=='__main__':
    import os, re, argparse
    from matplotlib import rc
    rc('text', usetex=True)
  
    script   = os.path.basename(__file__).split('.')[0]
    parser   = argparse.ArgumentParser('Plot GCP wells')
    parser.add_argument('-r','--root',       default=r'\data\cytoflex\Melbourne')
    parser.add_argument('-p','--plate',      default='all',          nargs='+')
    parser.add_argument('-w','--well',       default=['G12','H12'],  nargs='+')
    parser.add_argument('-N','--N',          default=25,             type=int)
    parser.add_argument('-t', '--tolerance', default=1.0e-6,         type=float)
    parser.add_argument('-s', '--show',      default=False,          action='store_true')
    
    args   = parser.parse_args()
    show   = args.show or args.plate!='all'
    for root, dirs, files in os.walk(args.root):
        path  = root.split(os.sep)
        match = re.match('.*(((PAP)|(RTI))[A-Z]*[0-9]+)',path[-1])
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
                            i1,i2           = get_boundaries(bins,n)
                            segment0        = [r for r in intensities if r < bins[i1]]
                            segment1        = [r for r in intensities if bins[i1]<r and r<bins[i2]]
                            segment2        = [r for r in intensities if  bins[i2]<r]
                            mu0,sigma0,_,y0 = get_gaussian(segment0,n=max(n[i] for i in range(i1)),bins=bins)
                            mu1,sigma1,_,y1 = get_gaussian(segment1,n=max(n[i] for i in range(i1,i2)),bins=bins)
                            mu2,sigma2,_,y2 = get_gaussian(segment2,n=max(n[i] for i in range(i2,len(n))),bins=bins)
                            ax2.plot(bins, y0, c='c', label='GMM')
                            ax2.fill_between(bins, y0, color='c', alpha=0.5)
                            ax2.plot(bins, y1, c='c')
                            ax2.fill_between(bins, y1, color='c', alpha=0.5)
                            ax2.plot(bins, y2, c='c')
                            ax2.fill_between(bins, y2, color='c', alpha=0.5)
                            zz  = zip(y0,y1,y2)
                            zs  = [z0 +z1 + z2 for (z0,z1,z2) in zip(y0,y1,y2)]
                            a,b = ax2.get_ylim()
                            cn  = 0.5*(a+b)
                            c0  = [y/z for (y,z) in zip(y0,zs)]
                            c1  = [y/z for (y,z) in zip(y1,zs)]
                            c2  = [y/z for (y,z) in zip(y2,zs)]
                            ax2.plot(bins, [cn*c for c in c0], c='m', label='c0')
                            ax2.plot(bins, [cn*c for c in c1], c='y', label='c1')
                            ax2.plot(bins, [cn*c for c in c2], c='b', label='c2')                            
                            ax2.set_title('Initialization')
                            ax2.set_xlabel('log(Red-H)')
                            ax2.set_ylabel('N')
                            ax2.legend()
                            
                            likelihoods,alphas,mus,sigmas =\
                                maximize_likelihood(
                                    intensities,
                                    mus    = [mu0,mu1,mu2],
                                    sigmas = [sigma0,sigma1,sigma2],
                                    alphas = [len(segment0)/(len(segment0)+len(segment1)+len(segment2)),
                                              len(segment1)/(len(segment0)+len(segment1)+len(segment2)),
                                              len(segment2)/(len(segment0)+len(segment1)+len(segment2))],
                                    N      = args.N,
                                    limit  = args.tolerance)
                            
                            ax3 = plt.subplot(2,2,3)
                            
                            ax3.plot(range(len(likelihoods)),likelihoods)
                            ax3.set_xlabel('Iteration')
                            ax3.set_ylabel('Log Likelihood')
                            
                            ax4 = plt.subplot(2,2,4)
                            
                            n,bins,_          = ax4.hist(intensities,facecolor='g',bins=100,label='From FCS')
                            ax4.plot(bins,[max(n)*alphas[0]*get_p(x,mu=mus[0],sigma=sigmas[0]) for x in bins],
                                     c='c',
                                     label=fr'$\mu=${mus[0]:.3f}, $\sigma=${sigmas[0]:.3f}')
                            ax4.plot(bins,[max(n)*alphas[1]*get_p(x,mu=mus[1],sigma=sigmas[1]) for x in bins],
                                     c='m',
                                     label=fr'$\mu=${mus[1]:.3f}, $\sigma=${sigmas[1]:.3f}')
                            ax4.plot(bins,[max(n)*alphas[2]*get_p(x,mu=mus[2],sigma=sigmas[2]) for x in bins],
                                     c='y',
                                     label=fr'$\mu=${mus[2]:.3f}, $\sigma=${sigmas[2]:.3f}')
                            
                            ax4.legend(framealpha=0.5)
                            
                            plt.savefig(os.path.join('figs',f'{script}-{plate}-{well}'))
                            
                            if not show:
                                plt.close()
                       
    if show:
        plt.show()    