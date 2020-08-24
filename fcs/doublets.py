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

import fcsparser,fcs,gcps,numpy as np
from scipy.stats import multivariate_normal


if __name__=='__main__':
    import os, re, argparse,sys,matplotlib.pyplot as plt
    from matplotlib import rc
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
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
                            ax1 = plt.subplot(3,3,1, projection='3d') #FIXME
                           
                            ax1.scatter(df1['FSC-H'],df1['SSC-H'] ,df1['FSC-Width'],s=1,c=df1['FSC-Width'])                            
                            ax1.set_xlabel('FSC-H')
                            ax1.set_ylabel('SSC-H')
                            ax1.set_zlabel('FSC-Width')
                            
                            n_f,bins_f = np.histogram(df1['FSC-H'],bins =100)
                            ax2 = plt.subplot(3,3,2)
                            ax2.scatter(bins_f[1:],n_f,s=1)
                            ax2.set_xlabel('FSC-H')
                            
                            n_s,bins_s = np.histogram(df1['SSC-H'],bins =100)                        
                            ax3 = plt.subplot(3,3,3)
                            ax3.scatter(bins_s[1:],n_s,s=1)
                            ax3.set_xlabel('SSC-H')

                            n_w,bins_w = np.histogram(df1['FSC-Width'],bins =100)
                            ax4 = plt.subplot(3,3,4)
                            ax4.scatter(bins_w[1:],n_w,s=1)
                            ax4.set_xlabel('FSC-Width')
                            
                            K=1
                            mu = [np.mean(df1['FSC-H']),np.mean(df1['SSC-H']),np.mean(df1['FSC-Width'])]
                            sigma = [
                                [np.std(df1['FSC-H'])**2, 0,                       0],
                                [0,                       np.std(df1['SSC-H'])**2, 0],
                                [0,                       0,                       np.std(df1['FSC-Width'])**2]]
                            var = multivariate_normal(mean=mu, cov=sigma)
                            xs = df1['FSC-H'].values
                            ys = df1['SSC-H'].values
                            zs = df1['FSC-Width'].values
                            ps = [var.pdf([xs[i],ys[i],zs[i]])
                                  for i in range(len(df1['FSC-H']))]                            
                            ax5 = plt.subplot(3,3,5, projection='3d')
                           
                            ax5.scatter(df1['FSC-H'],df1['SSC-H'] ,df1['FSC-Width'],s=1,c=ps)                            
                            ax5.set_xlabel('FSC-H')
                            ax5.set_ylabel('SSC-H')
                            ax5.set_zlabel('FSC-Width')
                            
                            K=2
                            
                            plt.savefig(
                                gcps.get_image_name(
                                    script = os.path.basename(__file__).split('.')[0],
                                    plate  = plate,
                                    well   = well))                            
    
    if show:
        plt.show()  
        