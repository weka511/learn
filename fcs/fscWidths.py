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
import fcs
import gcps
import standards

# Tracker
#
# An abstract paret for various logging classes
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
        self.wells  = []
        self.r_values    = []
    def accumulate(self,plate,well,r_value):
        super().accumulate(plate)
        self.wells.append(well)
        self.r_values.append(r_value)
    def build(self):
        self.refs = pd.DataFrame({ 
            'Plate' : self.plates,
            'Well'  : self.wells,
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
                
# plot_fsc_ssc_width
#
# Plot FSC-H and SSC-H, with colour showing FSC-Width

def plot_fsc_ssc_width(df,ax=None,title=''):
    sns.scatterplot(x       = df['FSC-H'],
                    y       = df['SSC-H'],
                    hue     = df['FSC-Width'],
                    palette = sns.color_palette('icefire', as_cmap=True),
                    s       = 1,
                    ax      = ax)
    ax.set_title(title)

# plot_fsc_width
#
# Plot histogram for FSC-Width, accopmanied by plot of Gausian Mixture Model

def plot_fsc_width(df,ax=None,mus=[0,0],sigmas=[1,1],alphas = [0.5,0.5]):
    sns.histplot(df, x  = 'FSC-Width', ax = ax, label='FSC-Width')
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
    legend = ax2.legend()
    suppress_y_labels(ax2)
    # Make symbols in legend larger - see Bruno Morais contribution:
    # https://stackoverflow.com/questions/24706125/setting-a-fixed-size-for-points-in-legend
    for handle in legend.legendHandles:
        handle.set_sizes([6.0])    
    
    ax.legend(loc='lower right')
    ax.set_title('Gaussian Mixture Model for FSC-Width')

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
    print (f'Using standard {levels} for {barcode}, r_value={r_value}')  
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
   
    
if __name__=='__main__':
    rc('text', usetex=True)
    parser = argparse.ArgumentParser('Plot FSC Width. Remove doublets from GCP wells, and perform regression on Red.')
    parser.add_argument('--root',
                        default = r'\data\cytoflex\Melbourne',
                        help    = 'Path to top of FCS files.')
    parser.add_argument('--plate',
                        nargs   = '+',
                        default = 'all',
                        help    = 'List of plates to process (or "all").')
    parser.add_argument('--wells',
                        choices = ['all',
                                   'controls',
                                   'gcps'],
                        default = 'all',
                        help    = 'Identify wells to be processed.')
    parser.add_argument('--mapping',
                        default = 'mapping.csv',
                        help    = 'File to store mapping between plates, locations, and serial numbers.')
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
    args              = parser.parse_args()
    references        = standards.create_standards(args.properties)
    mappingBuilder    = MappingBuilder(args.mapping)
    regressionTracker = RegressionTracker(args.r_values)
    widthStats        = {}
    
    for plate,well,df,meta,location in fcs.fcs(args.root,
                                 plate = args.plate,
                                 wells = args.wells):
        cytsn  = meta['$CYTSN']
        
        print (f'{ plate} {well} {location} {cytsn}')
        mappingBuilder.accumulate(plate,cytsn,location)
        fig  = plt.figure(figsize=(15,12))
        fig.suptitle(f'{plate} {well} {location} {cytsn}')        
        #plt.tight_layout()
        if is_gcp(well):
            axes                = fig.subplots(nrows=2,ncols=2)     
            df_gated_on_sigma   = fcs.gate_data(df,nsigma=2,nw=1) 
            widths              = df_gated_on_sigma['FSC-Width'].values
            q_05                = np.quantile(widths,0.05)
            q_95                = np.quantile(widths,0.95)
            _,alphas,mus,sigmas = gcps.maximize_likelihood(
                                       widths,
                                       mus    = [q_05,q_95],            # Initially assume the two means are 
                                                                        # close to the extremities
                                       sigmas = [q_95-q_05,q_95-q_05],  # Standard deviation also needs to be large
                                       alphas = [0.5,0.5],              # Perfect ignorance - assume each of the two
                                                                        # spans of data contains half the points 
                                       N      = args.N,
                                       limit  = args.tolerance,
                                       K      = 2)            
            
            widthStats[well]   = (alphas,mus,sigmas)
   
            plot_fsc_ssc_width(df_gated_on_sigma,
                               ax=axes[0][0],
                               title=r'Filtered on $\sigma$')
            
            plot_fsc_width(df_gated_on_sigma,
                           ax     = axes[0][1],
                           mus    = mus,
                           sigmas = sigmas,
                           alphas = alphas)
            
            df_resampled_doublets = resample_widths(df_gated_on_sigma,
                                                    mu    = mus[0],
                                                    sigma = sigmas[0])
            
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
            
            regressionTracker.accumulate(plate,well,r_value)
            
            plt.subplots_adjust(top=0.92,
                                bottom=0.08, 
                                left=0.10,
                                right=0.95, 
                                hspace=0.25,
                                wspace=0.35)
            
                      
        else:    # regular well
            axes                = fig.subplots(nrows=2,ncols=3)     
            df_gated_on_sigma   = fcs.gate_data(df,nsigma=2,nw=1)
            df_reduced_doublets = df_gated_on_sigma[df_gated_on_sigma['FSC-Width']<1000]            
            plot_fsc_ssc_width(df_reduced_doublets,ax=axes[0][0],title=r'Filtered on $\sigma$')
            sns.histplot(df_reduced_doublets,
                         x  = 'FSC-H',
                         ax = axes[0][1])
            sns.histplot(df_reduced_doublets,
                         x  = 'SSC-H',
                         ax = axes[1][0])
            
            _,mus_g12,sigmas_g12 = widthStats['G12'] 
            _,mus_h12,sigmas_h12 = widthStats['H12']   
            plot_fsc_width(df_gated_on_sigma,
                           ax=axes[1][1],
                           mu=0.5*(mus_g12[0]+mus_h12[0]),
                           sigma=math.sqrt(0.5*(sigmas_g12[0]**2+sigmas_h12[0]**2)))
            
            sns.scatterplot(x  = df_gated_on_sigma['FSC-H'],
                            y  = df_gated_on_sigma['FSC-Width'],
                            s  = 1,
                            ax = axes[0][2])
            sns.scatterplot(x  = df_gated_on_sigma['SSC-H'],
                            y  = df_gated_on_sigma['FSC-Width'],
                            s  = 1,
                            ax = axes[1][2])            
            
  
        plt.savefig(
            fcs.get_image_name(
                script = os.path.basename(__file__).split('.')[0],
                plate  = plate,
                well   = well)) 
        if not args.show:
            plt.close()
    
    mappingBuilder.save()
    regressionTracker.save()
    if args.show:
        plt.show()