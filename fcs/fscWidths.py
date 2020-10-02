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
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import seaborn as sns
import fcs

if __name__=='__main__':
    rc('text', usetex=True)
    parser = argparse.ArgumentParser('Plot FSC Width')
    parser.add_argument('--root',
                        default = r'\data\cytoflex\Melbourne')
    parser.add_argument('--plate',
                        nargs   = '+',
                        default = 'all')
    parser.add_argument('--wells',
                        choices = ['all',
                                   'controls',
                                   'gcps'],
                        default = 'all')
    parser.add_argument('--show',
                        default=False, 
                        action = 'store_true')
    args   = parser.parse_args()
    
    for plate,well,df in fcs.fcs(args.root,
                                 plate = args.plate,
                                 wells = args.wells):
        print (f'{ plate} {well}')    
    
        df_gated_on_sigma   = fcs.gate_data(df,nsigma=2,nw=1)
        df_reduced_doublets = df_gated_on_sigma[df_gated_on_sigma['FSC-Width']<1000]
        fig                 = plt.figure(figsize=(15,10))
        axes                = fig.subplots(nrows=2,ncols=3) 
        fig.suptitle(f'{ plate} {well}')
        sns.scatterplot(x       = df_reduced_doublets['FSC-H'],
                        y       = df_reduced_doublets['SSC-H'],
                        hue     = df_reduced_doublets['FSC-Width'],
                        palette = sns.color_palette('icefire', as_cmap=True),
                        s       = 1,
                        ax      = axes[0][0])
    
        sns.histplot(df_reduced_doublets,
                     x  = 'FSC-H',
                     ax = axes[0][1])
        sns.histplot(df_reduced_doublets,
                     x  = 'SSC-H',
                     ax = axes[1][0])
        sns.histplot(df_gated_on_sigma,
                     x  = 'FSC-Width',
                     ax = axes[1][1])
        
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
            
    if args.show:
        plt.show()