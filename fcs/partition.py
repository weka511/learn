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



from scipy import stats
import fcsparser
import math
import numpy as np
from scipy.stats import multivariate_normal
import em
import fcs
import gcps
import standards

# cluster_gcp
#
# Determibe mean and standard deviation for a GCP well
#
# Parameters:
#      plate
#      well
#      df
#      references Reference levels, used to determine r_value
#      K          Number of red levels that we expect

def cluster_gcp(plate,well,df,references,K=3):
    xs                  = df['FSC-H'].values
    ys                  = df['SSC-H'].values
    zs                  = df['FSC-Width'].values       
    
    _,_,ws,_,mus,Sigmas = gcps.filter_doublets(xs,ys,zs)     
    singlet             = gcps.get_selector(ws)
    intensities         = np.log(df['Red-H'][singlet]).values
    n,bins              = np.histogram(intensities,bins=100)
    indices             = gcps.get_boundaries(n,K=K) + [len(n)]

    segments            = [[r for r in intensities if bins[indices[k]]<r and r < bins[indices[k+1]]] 
                           for k in range(K)]
    mus_intensity       = []
    sigmas_intensity    = []
    heights_intensity   = []
    for k in range(K):
        mu,sigma,_,y = gcps.get_gaussian(segments[k],n=max(n[i] for i in range(indices[k],indices[k+1])),bins=bins)
        mus_intensity.append(mu)
        sigmas_intensity.append(sigma)
        heights_intensity.append(y)
        
    alphas_intensity = [len(segments[k]) for k in range(K)]
    alpha_norm       = sum(alphas_intensity)
    for k in range(K):
        alphas_intensity[k] /= alpha_norm
            
    _,_,mus_intensity,_  = gcps.maximize_likelihood(intensities,
                                                    mus    = mus_intensity,
                                                    sigmas = sigmas_intensity,
                                                    alphas = alphas_intensity,
                                                    N      = args.N,
                                                    limit  = args.tolerance,
                                                    K      = K)
    
    _,levels            = standards.lookup(plate,references)
    _, _, r_value, _, _ = stats.linregress(levels,[math.exp(y) for y in mus_intensity])    
    
    return well,r_value,mus,Sigmas

# read_fcs
#
# Read data from FCS file and gate it
#
# Parameters:
#     root
#     file

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

# get_distance
#
# Determine distance of point from GCP
#
# Parameters:
#     pt
#     mu
#     Sigma
#     d           Dimensionality of space

def get_distance(pt,mu,Sigma,d=3):
    return math.sqrt(sum([(pt[i]-mu[i]) * Sigma[i][j] * (pt[j]-mu[j]) for i in range(d) for j in range(d)]))

if __name__=='__main__':
    import argparse
    import sys
    import os
    import re
    from matplotlib import rc
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
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
    parser.add_argument('-s', '--show',
                        default=False,
                        action='store_true',
                        help='Display graphs')    
    args       = parser.parse_args()
    
    gcp_stats  = []
    references = standards.create_standards(args.properties)
    cmap       = plt.cm.get_cmap('seismic')
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
        
            best_gcp,rsq,mu_gcp,Sigma_gcp = gcp_stats[np.argmax([r for _,r,_,_ in gcp_stats])]
            for file in files:
                if not re.match('.*[A-H][1]?[0-9].fcs',file): continue
                if re.match('.*[GH]12.fcs',file): continue
                well,df=read_fcs(root,file)
                if well in args.well:
                    fig = plt.figure(figsize=(10,10))
                    plt.suptitle (f'{plate} {well} GCP={best_gcp}, rsq={rsq:.6f}')
                    xs     = df['FSC-H'].values
                    ys     = df['SSC-H'].values
                    zs     = df['FSC-Width'].values
                    ax1    = plt.subplot(2,2,1, projection='3d')
                    ax1.set_xlabel('FSC-H')
                    ax1.set_ylabel('SSC-H')
                    ax1.set_zlabel('FSC-Width')
                    # Plotting with superimposed points is tricky - 
                    # see https://stackoverflow.com/questions/33151163/pyplot-3d-scatter-points-at-the-back-overlap-points-at-the-front
                    ax1.plot(xs,ys,zs,'.',c='b',markersize=1)
                    ax1.plot([mu_gcp[0][0]],[mu_gcp[0][1]],[mu_gcp[0][2]],'x',c='r',markersize=20)
                    
                    ax2    = plt.subplot(2,2,2, projection='3d')
                    ax2.set_xlabel('FSC-H')
                    ax2.set_ylabel('SSC-H')
                    ax2.set_zlabel('FSC-Width')

                    gcp_distances = [get_distance([x,y,z],mu_gcp[0],Sigma_gcp[0]) for (x,y,z) in zip(xs,ys,zs)]
                    sc2           = ax2.scatter(xs,ys,zs,s=1,c=gcp_distances,cmap=cmap)
                    plt.colorbar(sc2)
                    ax2.set_xlim(ax1.get_xlim())
                    ax2.set_ylim(ax1.get_ylim()) 
                    ax2.set_zlim(ax1.get_zlim())
                    
                    ax3 = plt.subplot(2,2,3)
                    ax3.scatter(range(len(gcp_distances)),sorted(gcp_distances),
                                s=1,
                                c=sorted(gcp_distances),
                                cmap=cmap)
                    fig.tight_layout()
                    fig.savefig(
                        fcs.get_image_name(
                            script = os.path.basename(__file__).split('.')[0],
                            plate  = plate,
                            well   = well))
                    if not args.show:
                        plt.close(fig)
    if args.show:
        plt.show()
                    