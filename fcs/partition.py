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


from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import fcsparser
import math
import numpy as np
from scipy import stats
import em
import fcs
import gcps
import standards

def cluster_gcp(plate,well,df,references):
    print (f'{well}')

    xs     = df['FSC-H'].values
    ys     = df['SSC-H'].values
    zs     = df['FSC-Width'].values       
    K      = 3
    maximized,likelihoods,ws,alphas,mus_xyw,Sigmas_xyw = gcps.filter_doublets(xs,ys,zs)     
    singlet     = gcps.get_selector(ws)
    intensities = np.log(df['Red-H'][singlet]).values
    n,bins      = np.histogram(intensities,bins=100)
    indices     = gcps.get_boundaries(n,K=K)
    indices.append(len(n))
    segments = [[r for r in intensities if bins[indices[k]]<r and r < bins[indices[k+1]]] 
                for k in range(K)]
    mus      = []
    sigmas   = []
    heights      = []
    for k in range(K):
        mu,sigma,_,y = gcps.get_gaussian(segments[k],n=max(n[i] for i in range(indices[k],indices[k+1])),bins=bins)
        mus.append(mu)
        sigmas.append(sigma)
        heights.append(y)
        
        alphas = [len(segments[k]) for k in range(K)]
        alpha_norm = sum(alphas)
        for k in range(K):
            alphas[k] /= alpha_norm
    likelihoods,alphas,mus,sigmas =\
        gcps.maximize_likelihood(
            intensities,
            mus    = mus,
            sigmas = sigmas,
            alphas = alphas,
            N      = args.N,
            limit  = args.tolerance,
            K      = K)    
    barcode,levels = standards.lookup(plate,references)
    print (f'{barcode},{levels}')
    _, _, r_value, _, _ = stats.linregress(levels,[math.exp(y) for y in mus])
    print (f'Using standard for {barcode}, r_value={r_value}')    
    return well,r_value,mus_xyw,Sigmas_xyw

def read_fcs(root,file):
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
    return well,df1
    
if __name__=='__main__':
    import argparse
    import sys
    import os
    import re
    from matplotlib import rc
    
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
                        default=['A12'],
                        nargs='+',
                        help='Names of regular wells to be processed (omit for all)') 
    parser.add_argument('--properties',
                        default=r'\data\properties',
                        help='Root for properties files')
    parser.add_argument('-N','--N',
                        default=25, 
                        type=int, 
                        help='Number of attempts for iteration')
    parser.add_argument('-t', '--tolerance',
                        default=1.0e-6,
                        type=float, 
                        help='Iteration stops when ratio between likelihoods is this close to 1.')    
    args = parser.parse_args()
    
    gcp_stats = []
    references = standards.create_standards(args.properties)    

    for root, dirs, files in os.walk(args.root):
        path  = root.split(os.sep)
        match = re.match('.*(((PAP)|(RTI))[A-Z]*[0-9]+[r]?)',path[-1])
        if not match: continue
        
        plate = match.group(1)
        if args.plate=='all' or plate in args.plate:
            for file in files:
                if not re.match('.*[GH]12.fcs',file): continue
                well,df=read_fcs(root,file)
                gcp_stats.append(cluster_gcp(plate,well,df,references))
        
            ii = np.argmax([r for _,r,_,_ in gcp_stats])
            print (f'{gcp_stats[ii]}')
            for file in files:
                if not re.match('.*[A-H][1]?[0-9].fcs',file): continue
                if re.match('.*[GH]12.fcs',file): continue
                well,df=read_fcs(root,file)
                if well in args.well:
                    print (f'{plate} {well}')                