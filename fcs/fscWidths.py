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

import argparse
import fcsparser
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os
import seaborn as sns
import pandas as pd
import re
import fcs
import gcps

class MappingBuilder:
    def __init__(self):
        self.plates    = []
        self.cytsns    = []
        self.locations = []
        self.refs      = None
        
    def accumulate(self,plate,cytsn,location):
        if not plate in self.plates:
            self.plates.append(plate)
            self.cytsns.append(cytsn)
            self.locations.append(location)
            
    # Build mapping between Plate, serial number, and location 
    def build(self):
        self.refs = pd.DataFrame({ 
            'Plate'    : self.plates,
            'CYTSN'    : self.cytsns,
            'Location' : self.locations})
        self.refs.sort_values(by      = ['Plate'],
                              inplace = True)
    def save(self,path):
        self.refs.to_csv(path,index=False)  
 
def plot_fsc_ssc_width(df,ax=None):
    sns.scatterplot(x       = df['FSC-H'],
                    y       = df['SSC-H'],
                    hue     = df['FSC-Width'],
                    palette = sns.color_palette('icefire', as_cmap=True),
                    s       = 1,
                    ax      = ax)

#plot_fsc_width(df_gated_on_sigma,ax=None,mu=mus[0],sigma=sigmas[0])
def plot_fsc_width(df,ax=None,mu=0,sigma=1):
    sns.histplot(df,
                 x  = 'FSC-Width',
                 ax = ax)
    ax2 = ax.twinx()
    xs = df['FSC-Width'].values
    ax2.scatter(xs,[100 * gcps.get_p(w,mu,sigma) for w in xs],
                s=1,
                c='r',
                label=rf'$\mu$={mu:.0f},$\sigma$={sigma:.0f}')
    ax2.legend()
    
if __name__=='__main__':
    rc('text', usetex=True)
    parser = argparse.ArgumentParser('Plot FSC Width')
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
    parser.add_argument('-N','--N',
                        default = 25, 
                        type    = int, 
                        help    = 'Number of attempts for iteration')
    parser.add_argument('-t', '--tolerance',
                        default = 1.0e-6,
                        type    = float, 
                        help    = 'Iteration stops when ratio between likelihoods is this close to 1.')    
    parser.add_argument('--show',
                        default = False, 
                        action  = 'store_true',
                        help    = 'Indicates whether to display plots (they will be saved irregardless).')
    args           = parser.parse_args()
    
    re_gcp         = re.compile('[GH]12')
    mappingBuilder = MappingBuilder()
    widthStats     = {}
    
    for plate,well,df,meta,location in fcs.fcs(args.root,
                                 plate = args.plate,
                                 wells = args.wells):
        cytsn  = meta['$CYTSN']
        
        print (f'{ plate} {well} {location} {cytsn}')
        mappingBuilder.accumulate(plate,cytsn,location)
        fig                 = plt.figure(figsize=(15,10))
        axes                = fig.subplots(nrows=2,ncols=3) 
        fig.suptitle(f'{plate} {well} {location} {cytsn}')        
     
        df_gated_on_sigma   = fcs.gate_data(df,nsigma=2,nw=1)
        df_reduced_doublets = df_gated_on_sigma[df_gated_on_sigma['FSC-Width']<1000]
        if re_gcp.match(well) != None:         # GCP
            likelihoods,alphas,mus,sigmas =\
                gcps.maximize_likelihood(
                    df_gated_on_sigma['FSC-Width'].values,
                    mus    = [800,1200], #FIXME
                    sigmas = [200,200],  #FIXME
                    alphas = [0.5,0.5],  #FIXME
                    N      = args.N,
                    limit  = args.tolerance,
                    K      = 2)            
            print (alphas,mus,sigmas)
            widthStats[well] = (likelihoods,alphas,mus,sigmas)
   
            plot_fsc_ssc_width(df_reduced_doublets,ax=axes[0][0])
            
            sns.histplot(df_reduced_doublets,
                         x  = 'FSC-H',
                         ax = axes[0][1])
            sns.histplot(df_reduced_doublets,
                         x  = 'SSC-H',
                         ax = axes[1][0])
            plot_fsc_width(df_gated_on_sigma,ax=axes[1][1],mu=mus[0],sigma=sigmas[0])

            sns.scatterplot(x  = df_gated_on_sigma['FSC-H'],
                            y  = df_gated_on_sigma['FSC-Width'],
                            s  = 1,
                            ax = axes[0][2])
            sns.scatterplot(x  = df_gated_on_sigma['SSC-H'],
                            y  = df_gated_on_sigma['FSC-Width'],
                            s  = 1,
                            ax = axes[1][2])            
        else:
            plot_fsc_ssc_width(df_reduced_doublets,ax=axes[0][0])
            sns.histplot(df_reduced_doublets,
                         x  = 'FSC-H',
                         ax = axes[0][1])
            sns.histplot(df_reduced_doublets,
                         x  = 'SSC-H',
                         ax = axes[1][0])
            #sns.histplot(df_gated_on_sigma,
                         #x  = 'FSC-Width',
                         #ax = axes[1][1])
            
            _,_,mus_g12,sigmas_g12 = widthStats['G12'] 
            _,_,mus_h12,sigmas_h12 = widthStats['H12']   
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
    
    mappingBuilder.build()
    mappingBuilder.save(args.mapping)
    
    if args.show:
        plt.show()