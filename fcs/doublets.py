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
                             
                            n_f,bins_f = np.histogram(df1['FSC-H'],bins =100)
                              
                            n_s,bins_s = np.histogram(df1['SSC-H'],bins =100)                        
 
                            n_w,bins_w = np.histogram(df1['FSC-Width'],bins =100)
                            
                            K     = 1
                            xs    = df1['FSC-H'].values
                            ys    = df1['SSC-H'].values
                            zs    = df1['FSC-Width'].values
                            mu    = [np.mean(xs),np.mean(ys),np.mean(zs)]
                            sigma = np.cov([xs,ys,zs],rowvar=True)                                                    
                            var   = multivariate_normal(mean=mu, cov=sigma)
                            ps    = [var.pdf([xs[i],ys[i],zs[i]]) for i in range(len(df1['FSC-H']))]  
                            ax1   = plt.subplot(2,2,1, projection='3d')
                            ax1.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1,c=ps)                            
                            ax1.set_xlabel('FSC-H')
                            ax1.set_ylabel('SSC-H')
                            ax1.set_zlabel('FSC-Width')
                            
                            for i in range(3):
                                mu    = [np.average(xs,weights=ps),np.average(ys,weights=ps),np.average(zs,weights=ps)]
                                sigma = np.cov([xs,ys,zs],rowvar=True,aweights=ps)                                                    
                                var   = multivariate_normal(mean=mu, cov=sigma)                            
                                ps    = [var.pdf([xs[i],ys[i],zs[i]]) for i in range(len(df1['FSC-H']))] 
                                ax2   = plt.subplot(2,2,2+i, projection='3d')
                                ax2.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1,c=ps)                            
                                ax2.set_xlabel('FSC-H')
                                ax2.set_ylabel('SSC-H')
                                ax2.set_zlabel('FSC-Width')                            
                            
                            K=2
                            
                            plt.savefig(
                                gcps.get_image_name(
                                    script = os.path.basename(__file__).split('.')[0],
                                    plate  = plate,
                                    well   = well))                            
    
    if show:
        plt.show()  
        