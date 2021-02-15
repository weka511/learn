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
import fcs
import math
import matplotlib.pyplot as plt
from   matplotlib import rc
from   mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import random
import re
from scipy.stats import multivariate_normal
import sys
import emm

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
    parser.add_argument('-s', '--show',
                        default=False,
                        action='store_true',
                        help='Display graphs')
    parser.add_argument('--init',
                        choices=['randomize','sigma'],
                        default='sigma',
                        help='Initialize means by offsetting from true mean using stndard deviation (sigma) or at random (randomize)')
    parser.add_argument('--mult',
                        default = 0.25,
                        type=float,
                        help='If INIT is sigma, this is the offset from the mean')
    parser.add_argument('--separation',
                        default=1000000,
                        type=float,
                        help='If INIT is randomize, this is minimum squared separation')
    args     = parser.parse_args()
    show     = args.show or args.plate!='all'
    failures = []
    cm       = plt.cm.get_cmap('RdYlBu')
    
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
                        df1      = df[(0               < df['FSC-H']) & \
                                      (df['FSC-H']     < 1000000  )   & \
                                      (0               < df['SSC-H']) & \
                                      (df['SSC-H']     < 1000000)     & \
                                      (df['FSC-Width'] < 2000000)]
                        
                        well     = fcs.get_well_name(tbnm)
 
                        if well in args.well:
                            print (f'{plate} {well}')
  
                            plt.figure(figsize=(10,10))
                            plt.suptitle(f'{plate} {well}')
                                          
                            d      = 3
  
                            xs     = df1['FSC-H'].values
                            ys     = df1['SSC-H'].values
                            zs     = df1['FSC-Width'].values 
                            
                            mus       = []
                            init_text = []
                            if args.init=='sigma':
                                mu     = [np.mean(xs),np.mean(ys),np.mean(zs)]
                                sigma  = [np.std(xs),np.std(ys),np.std(zs)]
                                mus    = [[mu[i]+ direction*args.mult*sigma[i] for i in range(d)] for direction in [-1,+1]]
                                init_text = f'Sigma: mult={args.mult}'
                            else:
                                i      = random.choice(range(len(xs)))
                                j      = random.choice(range(len(xs)))
                                while sqdist([xs[i],ys[i],zs[i]],[xs[j],ys[j],zs[j]])<args.separation:
                                    i      = random.choice(range(len(xs)))
                                    j      = random.choice(range(len(xs)))                                
                                mus = [[xs[i],ys[i],zs[i]],[xs[j],ys[j],zs[j]]]
                                init_text = f'Random: separation={args.separation}'
  
                            alphas = [0.5,0.5]
                            Sigmas = [np.cov([xs,ys,zs],rowvar=True),
                                      np.cov([xs,ys,zs],rowvar=True)]
  
                            outcome,likelihoods,ws,alphas,mus,Sigmas = emm.maximize_likelihood(
                                                                           xs,ys,zs,
                                                                           mus=mus,
                                                                           Sigmas=Sigmas,
                                                                           alphas=alphas)                       
                            if outcome:
                                ax1    = plt.subplot(2,2,1, projection='3d')
                                sc1    = ax1.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1,c=[ws[0]], cmap=cm)
                                ax1.set_xlabel('FSC-H')
                                ax1.set_ylabel('SSC-H')
                                ax1.set_zlabel('FSC-Width')
                                ax1.set_title(init_text)
                                ax2    = plt.subplot(2,2,2, projection='3d')
                                sc2    = ax2.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1,c=[ws[1]], cmap=cm) 
                                ax2.set_xlabel('FSC-H')
                                ax2.set_ylabel('SSC-H')
                                ax2.set_zlabel('FSC-Width')                            
                                plt.colorbar(sc1)
                                ax3    = plt.subplot(2,2,3)
                                ax3.plot(range(len(likelihoods)),likelihoods)
                                ax3.set_ylabel('log(likelihood)')
                            else:
                                ax1    = plt.subplot(2,2,1, projection='3d')
                                sc1    = ax1.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1)
                                ax1.set_xlabel('FSC-H')
                                ax1.set_ylabel('SSC-H')
                                ax1.set_zlabel('FSC-Width')
                                ax1.set_title('Failed')
                                failures.append(f'{plate} {well}')
                            plt.savefig(
                                fcs.get_image_name(
                                    script = os.path.basename(__file__).split('.')[0],
                                    plate  = plate,
                                    well   = well)) 
                            if not show:
                                plt.close()                            
 
    if len(failures)>0:
        print ('Failures')
        for fail in failures:
            print (fail)
             
    if show:
        plt.show()  
        
           
