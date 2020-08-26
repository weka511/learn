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


import  re, argparse,sys,em,os,fcsparser,fcs,matplotlib.pyplot as plt,numpy as np
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D

rc('text', usetex=True)
cmap     = plt.cm.get_cmap('RdYlBu')
parser   = argparse.ArgumentParser('Fit Gaussian Mixture Model to GCP wells')

parser.add_argument('-r','--root',
                    default=r'\data\cytoflex\Melbourne',
                    help='Root for fcs files')
parser.add_argument('-p','--plate',
                    default='all',
                    nargs='+',
                    help='Name of plate to be processed (omit for all)')
parser.add_argument('-w','--well', 
                    default=[],
                    nargs='+',
                    help='Names of wells to be processed (omit for all)')
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
                    df1      = df[(0               < df['FSC-H']) & \
                                  (df['FSC-H']     < 1000000  )   & \
                                  (0               < df['SSC-H']) & \
                                  (df['SSC-H']     < 1000000)     & \
                                  (df['FSC-Width'] < 2000000)]
                    
                    if well in args.well or len(args.well)==0:
                        print (f'{plate} {well}') 
                        plt.figure(figsize=(10,10))
                        plt.suptitle(f'{plate} {well}')

                        xs     = df1['FSC-H'].values
                        ys     = df1['SSC-H'].values
                        zs     = df1['FSC-Width'].values  
                        ax1    = plt.subplot(3,3,1, projection='3d')
                        plt.colorbar(ax1.scatter(xs,ys,zs,s=1,c=zs,cmap=cmap)) 
                        K      = 6
                        mus    = em.get_mus(xs,ys,zs)
                        alphas = [1/K for _ in range(K)]
                        Sigmas = [np.cov([xs,ys,zs],rowvar=True) for _ in range(K)]
                    
                        maximized,likelihoods,ws,alphas,mus,Sigmas = \
                            em.maximize_likelihood(
                                xs,ys,zs,
                                mus=mus,
                                Sigmas=Sigmas,
                                alphas=alphas,
                                K=K)
                        for k in range(K):
                            axk    = plt.subplot(3,3,k+2, projection='3d')
                            plt.colorbar(axk.scatter(xs,ys,zs,s=1,c=ws[k],cmap=cmap)) 
plt.show()
