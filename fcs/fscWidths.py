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

from abc import ABC,abstractmethod
import argparse
import fcsparser
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os
import seaborn as sns
import pandas as pd
import random 
import re
import scipy.stats as stats
from time import gmtime, strftime, time

import fcs
import gcps
import standards

# Tracker
#
# An abstract parent for various logging classes
class Tracker(ABC):
    def __init__(self,path='tracker.csv'):
        self.plates = []
        self.refs   = None
        self.path   = path
    def accumulate(self,plate):
        self.plates.append(plate)
    @abstractmethod
    def build(self):
        pass
    def save(self):
        if self.refs == None:
            self.build()
        self.refs.to_csv(self.path,index=False)   

# RegressionTracker
#
# This class is responsible for keeping track of
# regression coefficients      
class RegressionTracker(Tracker):
    def __init__(self,path='r_values.csv'):
        super().__init__(path=path)
        self.wells       = []
        self.r_values    = []
        self.s1s         = []
        self.s2s         = []
        self.s3s         = []
    def accumulate(self,plate,well,levels,r_value):
        super().accumulate(plate)
        self.wells.append(well)
        self.s1s.append(levels[0])
        self.s2s.append(levels[1])
        self.s3s.append(levels[2])     
        self.r_values.append(r_value)
    def build(self):
        self.refs = pd.DataFrame({ 
            'Plate'   : self.plates,
            'Well'    : self.wells,
            'S1'      : self.s1s,
            'S2'      : self.s2s,
            'S3'      : self.s3s,
            'r_value' : self.r_values})

        
# MappingBuilder
#
# This class is responsible for keeping track of
# the location and instrument used for each plate
class MappingBuilder(Tracker):
    def __init__(self,path='mapping.csv'):
        super().__init__(path=path)
        self.cytsns    = []
        self.locations = []
        
    def accumulate(self,plate,cytsn,location):
        if not plate in self.plates:
            super().accumulate(plate)
            self.cytsns.append(cytsn)
            self.locations.append(location)
            
    # Build mapping between Plate, serial number, and location 
    def build(self):
        self.refs = pd.DataFrame({ 
            'Plate'    : self.plates,
            'CYTSN'    : self.cytsns,
            'Location' : self.locations})
    

# suppress_y_labels
#
# Used to suppress display of y label when I've twinned an axis - see
# https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots

def suppress_y_labels(ax):
    for xlabel_i in ax.get_yticklabels():
        xlabel_i.set_fontsize(0.0)
        xlabel_i.set_visible(False) 

# enlarge_symbols_in_legend
#
# Make symbols in legend larger - see Bruno Morais contribution:
# https://stackoverflow.com/questions/24706125/setting-a-fixed-size-for-points-in-legend
def enlarge_symbols_in_legend(legend,size=6.0):
    for handle in legend.legendHandles:
        handle.set_sizes([size])  
        
# plot_fsc_ssc_width
#
# Plot FSC-H and SSC-H, with colour showing FSC-Width

def plot_fsc_ssc_width(df,
                       ax=None,
                       title=''):
    sns.scatterplot(x       = df['FSC-H'],
                    y       = df['SSC-H'],
                    hue     = df['FSC-Width'],
                    palette = sns.color_palette('icefire', as_cmap=True),
                    s       = 1,
                    ax      = ax)
    ax.set_title(title)

# plot_fsc_width_histogram
#
# Plot histogram for FSC-Width, accompanied by plot of Gausian Mixture Model

def plot_fsc_width_histogram(df,
                   ax     = None, 
                   mus    = [0,0],
                   sigmas = [1,1],
                   alphas = [0.5,0.5]):
    sns.histplot(df, x  = 'FSC-Width', ax = ax, label='FSC-Width')
    ax.legend(loc='lower right')
    ax.set_title('Gaussian Mixture Model for FSC-Width')
    
    ax2 = ax.twinx()
    xs  = df['FSC-Width'].values
    ys  = [[alphas[i]*gcps.get_p(w,mus[i],sigmas[i]) for w in xs] for i in range(len(alphas))]
    for i in range(len(alphas)):
        ax2.scatter(xs,ys[i],
                    s     = 1,
                    c     = ['r', 'g'][i],
                    alpha = 0.5,
                    label = rf'$\mu$={mus[i]:.0f}, $\sigma$={sigmas[i]:.0f}')
 
    ax2.set_ylim((0,max(ys[0])))
    suppress_y_labels(ax2)
    legend = ax2.legend()
    enlarge_symbols_in_legend(legend)


# resample_widths
#
# Create a sample, of the same size as the original data, whose widths match the 
# first Gaussian in GMM

def resample_widths(df,mu=0,sigma=1):
    widths   = df['FSC-Width'].values
    selector = []
    nrows    = len(df.index)
    while len(selector)<nrows:
        candidate = random.randint(0,nrows-1)
        x         = widths[candidate]
        p         = math.exp(-0.5*((x-mu)/sigma)**2)
        test      = random.random()
        if p>test:
            selector.append(candidate)
    return df.iloc[selector] 
 
# is_gcp
#
# Verify thaat well is a GCP well

def is_gcp(well):
    return re.match('[GH]12',well)

# create_segments

def create_segments(intensities):
    n,bins    = np.histogram(intensities,bins=100)
    indices     = gcps.get_boundaries(n,K=3)
    indices.append(len(n))
    segments = [[r for r in intensities if bins[indices[k]]<r and r < bins[indices[k+1]]] 
                            for k in range(3)]
    mus      = []
    sigmas   = []
    heights  = []
    for k in range(3):
        mu,sigma,_,y = gcps.get_gaussian(segments[k],
                                         n=max(n[i] for i in range(indices[k],indices[k+1])),bins=bins)
        mus.append(mu)
        sigmas.append(sigma)
        heights.append(y)

    return mus, sigmas, segments

# fit_reds

def fit_reds(segments=[],intensities=[],mus=[],sigmas=[],N=25,tolerance=1e-5):
    alphas = [len(segments[k]) for k in range(3)]
    alpha_norm = sum(alphas)
    for k in range(3):
        alphas[k] /= alpha_norm
    likelihoods,alphas,mus,sigmas =\
        gcps.maximize_likelihood(
            intensities,
            mus    = mus,
            sigmas = sigmas,
            alphas = alphas,
            N      = args.N,
            limit  = args.tolerance,
            K      = 3)  
    barcode,levels = standards.lookup(plate,references)
    _, _, r_value, _, _ = stats.linregress(levels,[math.exp(y) for y in mus])
    return alphas,mus,sigmas,levels,r_value

# plot_GMM_for_reds

def plot_GMM_for_reds(intensities=[],alphas=[],mus=[],sigmas=[],levels=[],r_value=0,ax=None):
    n,bins,_ = ax.hist(intensities,facecolor='g',bins=100,alpha=0.5)
    ax.set_xlabel(r'$\log(Red)$')
    ax2      = ax.twinx()
    for k in range(3):
        ax2.plot(bins,
                 [max(n)*alphas[k]*gcps.get_p(x,mu=mus[k],sigma=sigmas[k]) for x in bins],
                 label=fr'{levels[k]}, $\mu=${mus[k]:.3f}, $\sigma=${sigmas[k]:.3f}')
    _,ymax = ax2.get_ylim()
    ax2.set_ylim(0,ymax)
    ax2.legend(framealpha=0.5,title=f'$r^2=${r_value:.8f}')
    ax2.set_title('GMM for Red')
    suppress_y_labels(ax2)
   
class Logger:
    def __init__(self,path):
        self.path = path
    def __enter__(self):
        self.file = open(self.path,'w')
        return self
    def log(self,text):
        print (text)
        self.file.write(f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} {text}\n')    
    def __exit__(self,etype, value, traceback):
        if traceback is not None:
            print(f'{etype}, {value}, {traceback}')
        try:
            self.file.close()
        except:
            e = sys.exc_info()[0]
            print (e)

# fit_gmm_to_widths
#
# Fit Gaussiam Mixture Model with 2 peaks to FSC-Width

def fit_gmm_to_widths(widths,N=25,tolerance=1e-5):
    q_05 = np.quantile(widths,0.05)
    q_95 = np.quantile(widths,0.95)
    return  gcps.maximize_likelihood(
                     widths,
                     mus    = [q_05,q_95],            # Initially assume the two means are 
                                                      # close to the extremities
                     sigmas = [q_95-q_05,q_95-q_05],  # Standard deviation also needs to be large
                     alphas = [0.5,0.5],              # Perfect ignorance - assume each of the two
                                                      # spans of data contains half the points 
                     N      = N,
                     limit  = tolerance,
                     K      = 2) 

# fit_line_fsc_width
#
# returns gradient, intercept, r_value, p_value, std_err
def fit_line_fsc_width(xs=[], ys=[], x_gcp=0):
    selector = [i for i in range(len(xs)) if xs[i]>x_gcp]
    return stats.linregress([xs[i] for i in selector],
                            [ys[i] for i in selector])
    

def plot_line_fsc_width(x_gcp=0, x_max=0, ax=None,ylim=None,gradient=1, intercept=0, r_value=0,n=100):
    x  = np.linspace(x_gcp,x_max,n)
    ax.plot(x,gradient*x + intercept,
            '-r',
            label=f'$r^2$={r_value:.3f}')
    ax.set_ylim(ylim)
    ax.legend()
    
if __name__=='__main__':
    rc('text', usetex=True)
    start = time()
    parser = argparse.ArgumentParser('Plot FSC Width. Remove doublets from GCP wells, and perform regression on Red.')
    parser.add_argument('--root',
                        default = r'\data\cytoflex\Melbourne',
                        help    = 'Path to top of FCS files.')
    parser.add_argument('--plate',
                        nargs   = '+',
                        default = 'all',
                        help    = 'List of plates to process (or "all").')
    
    parser.add_argument('--wells',
                        nargs='+',
                        choices = ['all',
                                   'controls',
                                   'gcps'] 
                                  + [f'{row}{column}' for row in 'ABCDEFGH' for column in range(1,13)],
                        default = 'controls',
                        help    = 'Identify wells to be processed.')
    parser.add_argument('--mapping',
                        default = 'mapping.csv',
                        help    = 'File to store mapping between plates, locations, and serial numbers.')
    parser.add_argument('--log',
                        default = 'log.txt',
                        help    = 'Path to Log file.')    
    parser.add_argument('--r_values',
                        default = 'r_values.csv',
                        help    = 'File to store r_values.')    
    parser.add_argument('-N','--N',
                        default = 25, 
                        type    = int, 
                        help    = 'Number of attempts for iteration')
    parser.add_argument('-t', '--tolerance',
                        default = 1.0e-6,
                        type    = float, 
                        help    = 'Iteration stops when ratio between likelihoods is this close to 1.')
    parser.add_argument('--properties',
                        default = r'\data\properties',
                        help    = 'Root for properties files')    
    parser.add_argument('--show',
                        default = False, 
                        action  = 'store_true',
                        help    = 'Indicates whether to display plots (they will be saved irregardless).')
    parser.add_argument('--seed',
                        default = None,
                        help = 'Seed for random number generator')
    
    args              = parser.parse_args()
    references        = standards.create_standards(args.properties)
    mappingBuilder    = MappingBuilder(args.mapping)
    regressionTracker = RegressionTracker(args.r_values)
    widthStats        = {}
    
    random.seed(args.seed)
    with Logger(args.log) as logger:
        for key,value in vars(args).items():
            logger.log(f'{key} = {value}')
            
        for plate,well,df,meta,location in fcs.fcs(args.root,
                                     plate = args.plate,
                                     wells = args.wells):
            cytsn  = meta['$CYTSN']
            
            logger.log (f'{ plate} {well} {location} {cytsn}')
            mappingBuilder.accumulate(plate,cytsn,location)
            fig  = plt.figure(figsize=(15,12))
            fig.suptitle(f'{plate} {well} {location} {cytsn}')        
            
            if is_gcp(well):
                axes                = fig.subplots(nrows=2,ncols=2)     
                df_gated_on_sigma   = fcs.gate_data(df,nsigma=2,nw=1) 
   
                _,alphas,mus,sigmas = fit_gmm_to_widths(df_gated_on_sigma['FSC-Width'].values,
                                                        N=args.N,
                                                        tolerance=args.tolerance)
                
                plot_fsc_ssc_width(df_gated_on_sigma,
                                   ax=axes[0][0],
                                   title=r'Filtered on $\sigma$')
                
                plot_fsc_width_histogram(df_gated_on_sigma,
                                         ax     = axes[0][1],
                                         mus    = mus,
                                         sigmas = sigmas,
                                         alphas = alphas)
                
                df_resampled_doublets = resample_widths(df_gated_on_sigma,
                                                        mu    = mus[0],
                                                        sigma = sigmas[0])
         
                widthStats[well]   = (alphas,mus,sigmas,
                                      np.mean(df_resampled_doublets['FSC-H']),
                                      np.mean(df_resampled_doublets['SSC-H']))
                
                plot_fsc_ssc_width(df_resampled_doublets,
                                   ax=axes[1][0],
                                   title='Resampled on FSC-Width')
                 
                intensities                      = np.log(df_resampled_doublets['Red-H']).values 
                mus,sigmas,segments              = create_segments(intensities)
                alphas,mus,sigmas,levels,r_value = fit_reds(segments    = segments,
                                                            intensities = intensities,
                                                            mus         = mus,
                                                            sigmas      = sigmas,
                                                            N           = args.N,
                                                            tolerance   = args.tolerance)
                plot_GMM_for_reds(intensities = intensities,
                                  alphas      = alphas,
                                  mus         = mus,
                                  sigmas      = sigmas,
                                  levels      = levels,
                                  r_value     = r_value,
                                  ax          = axes[1][1])
                
                regressionTracker.accumulate(plate,well,levels,r_value)
                
                plt.subplots_adjust(top    = 0.92,
                                    bottom = 0.08, 
                                    left   = 0.10,
                                    right  = 0.95, 
                                    hspace = 0.25,
                                    wspace = 0.35)
                
                          
            else:    # regular well
                axes                 = fig.subplots(nrows=2,ncols=3)     
                df_gated_on_sigma    = fcs.gate_data(df,nsigma=2,nw=1)
                _,mus_g12,sigmas_g12,x0_g12,y0_g12 = widthStats['G12'] 
                _,mus_h12,sigmas_h12,x0_h12,y0_h12 = widthStats['H12']                
                plot_fsc_ssc_width(df_gated_on_sigma,
                                   ax=axes[0][0],
                                   title=r'Filtered on $\sigma$')
                ax2 = axes[0][0].twinx();
                
                plot_fsc_width_histogram(df_gated_on_sigma,
                               ax     = axes[0][1],
                               mus    = [0.5*(mus_g12[i]+mus_h12[i]) for i in range(2)],
                               sigmas = [math.sqrt(0.5*(sigmas_g12[i]**2+sigmas_h12[i]**2))for i in range(2)] )
                
                sns.scatterplot(x  = df_gated_on_sigma['FSC-H'],
                                y  = df_gated_on_sigma['FSC-Width'],
                                s  = 1,
                                ax = axes[0][2])
                
                x_gcp     = 0.5*(x0_g12+x0_h12)
                gradient, intercept, r_value, _, _ = fit_line_fsc_width(
                                                        xs    = df_gated_on_sigma['FSC-H'].values,
                                                        ys    = df_gated_on_sigma['FSC-Width'].values,
                                                        x_gcp = x_gcp)
                plot_line_fsc_width(
                    ax        = axes[0][2].twinx(),
                    x_gcp     = x_gcp,
                    x_max     = max(df_gated_on_sigma['FSC-H']),
                    ylim      = axes[0][2].get_ylim(),
                    gradient  = gradient,
                    intercept = intercept)               
                
                fsc_h_s   = df_gated_on_sigma['FSC-H'].values
                ssc_h_s   = df_gated_on_sigma['SSC-H'].values
                fsc_w_s   = df_gated_on_sigma['FSC-Width'].values                
                selection = [i for i in range(len(fsc_h_s)) if fsc_h_s[i] < x_gcp or fsc_w_s[i]<gradient*fsc_h_s[i] + intercept]
                n,bins,_  = axes[1][0].hist(fsc_h_s[selection],bins=50)
                quantiles = [np.quantile(fsc_h_s,q/7) for q in range(1,7)]
            
                _,alphas,mus,sigmas=  gcps.maximize_likelihood(
                                            fsc_h_s,
                                            mus    = quantiles,       
                                            sigmas = [(quantiles[5]-quantiles[0])/10]*6,
                                            alphas = [1/6]*6,                                        
                                            N      = args.N,
                                            limit  = args.tolerance,
                                            K      = 6) 
                ax2      = axes[1][0].twinx()
                for k in range(6):
                    ax2.plot(bins,
                             [max(n)*alphas[k]*gcps.get_p(x,mu=mus[k],sigma=sigmas[k]) for x in bins],
                             label=fr'$\mu=${mus[k]:.3f}, $\sigma=${sigmas[k]:.3f}')
                _,ymax = ax2.get_ylim()
                ax2.set_ylim(0,ymax)
                ax2.legend(framealpha=0.5,title=f'$r^2=${r_value:.8f}')
                ax2.set_title('GMM for FSC-H')
                suppress_y_labels(ax2)                
               
                
               
                
            plt.savefig(
                fcs.get_image_name(
                    script = os.path.basename(__file__).split('.')[0],
                    plate  = plate,
                    well   = well)) 
            if not args.show:
                plt.close()
        
        mappingBuilder.save()
        regressionTracker.save()
        
        end              = time()
        hours, rem       = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.log(f'Elapsed time: {int(hours)}:{int(minutes)}:{int(seconds)} ')
        
    if args.show:
        plt.show()