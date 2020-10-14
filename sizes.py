# Copyright (C) 2020 Greenweaves Software Limited

#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Notes on the EM Algorithm for Gaussian Mixtures: CS 274A, Probabilistic Learning 
# Padhraic Smyth 
# https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf

import  argparse
import fcsparser
import matplotlib.pyplot as plt
from   matplotlib import rc
from   mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import re
import sys
import time

import emm
import fcs

rc('text', usetex=True)
cmap     = plt.cm.get_cmap('RdYlBu')

d = 3

parser   = argparse.ArgumentParser('Group data into size clusters')

parser.add_argument('--root',
                    default = r'\data\cytoflex\Melbourne',
                    help    = 'Root for fcs files')
parser.add_argument('--plate',
                    default = 'all',
                    nargs   = '+',
                    help    = 'Name of plate(s) to be processed (omit for all)')
parser.add_argument('--well', 
                    default = [],
                    nargs   = '+',
                    help    = 'Names of wells to be processed (omit for all)')
parser.add_argument('--K',
                    default = [6],
                    type    = int,
                    nargs   = '+')

parser.add_argument( '--show',
                    default=False,
                    action='store_true',
                    help='Display graphs')

args   = parser.parse_args()

for root, dirs, files in os.walk(args.root):
    path  = root.split(os.sep)
    match = re.match('.*(((PAP)|(RTI))[A-Z]*[0-9]+[r]?)',path[-1])
    if match:
        plate = match.group(1)
        if args.plate=='all' or plate in args.plate:
            for file in files:
                if re.match('.*(([A-H][1-9][01]?)|([A-F]12)).fcs',file):
                    path     = os.path.join(root,file)
                    meta, df = fcsparser.parse(path, reformat_meta=True)
                    
                    date     = meta['$DATE']
                    tbnm     = meta['TBNM']
                    btim     = meta['$BTIM']
                    etim     = meta['$ETIM']
                    cyt      = meta['$CYT']
                    well     = fcs.get_well_name(tbnm)
                    
                    df1      = df[(200000          < df['FSC-H']) & \
                                  (df['FSC-H']     < 1000000  )   & \
                                  (75000           < df['SSC-H']) & \
                                  (df['SSC-H']     < 1000000)     & \
                                  (df['FSC-Width'] < 2000)]
                    
                    if well in args.well or len(args.well)==0:
 
                        start_time = time.time()                        
 
                        xs     = df1['FSC-H'].values
                        ys     = df1['SSC-H'].values
                        zs     = df1['FSC-Width'].values  
   
                        for K in args.K:
                            try:
                                mus    = emm.get_mus(xs,ys,zs,K=K,min_separation=10000000)
                                alphas = [1/K for _ in range(K)]
                                #Sigmas = [np.cov([xs,ys,zs],rowvar=True) for _ in range(K)]   
                                Sigma = [[0 for _ in range(d)] for _ in range(d)]
                                Sigma[0][0]= 1e10
                                Sigma[1][1]= 1e10
                                Sigma[2][2]= 1e4
                                
                                Sigmas = [Sigma for _ in range(K)]  
                                maximized,likelihoods,ws,alphas,mus,Sigmas = \
                                    emm.maximize_likelihood(
                                        xs,ys,zs,
                                        mus    = mus,
                                        Sigmas = Sigmas,
                                        alphas = alphas,
                                        K      = K)
                                plt.figure(figsize=(10,10))                       
                                for k in range(K):
                                    axk    = plt.subplot(3,3,k+1, projection='3d')
                                    axk.scatter(xs,ys,zs,s=1,c=ws[k],cmap=cmap)
                                elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                                plt.suptitle(f'{plate} {well} K={K}: elapsed {elapsed_time}')
                                plt.savefig(
                                    fcs.get_image_name(
                                        script = os.path.basename(__file__).split('.')[0],
                                        plate  = plate,
                                        well   = well,
                                        K      = K))                                
                                print (f'{plate} {well} K={K}: elapsed {elapsed_time}')
                                if not args.show:
                                    plt.close()
                            except:
                                print (f'{plate} {well} K={K}: failed: {sys.exc_info()[0]}') 
if args.show:
    plt.show()
