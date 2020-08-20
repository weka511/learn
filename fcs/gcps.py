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

import fcsparser, matplotlib.pyplot as plt,numpy as np,scipy.stats as stats

def get_well_name(tbnm):
    return tbnm[-3:]

def get_bounds(df,channel,nsigma=3):
    mean   = np.mean(df[channel])
    std    = np.std(df[channel])
    return (mean-nsigma*std,mean+nsigma*std)

def gate_data(df,nsigma=3,nw=2):
    fsc_min,fsc_max = get_bounds(df,'FSC-H',nsigma=nsigma)
    ssc_min,ssc_max = get_bounds(df,'SSC-H',nsigma=nsigma)
    _,fsc_width_max = get_bounds(df,'FSC-Width',nsigma=nw)
    
    return df[(fsc_min         < df['FSC-H']) & \
              (df['FSC-H']     < fsc_max)     & \
              (ssc_min         < df['SSC-H']) & \
              (df['SSC-H']     < ssc_max)     & \
              (df['FSC-Width'] < fsc_width_max)]

def purge_outliers(df,nsigma=3,nw=2,max_iterations=float('inf')):
    nr0,_ = df.shape
    df1   = gate_data(df,nsigma=nsigma)
    nr1,_ = df1.shape
    i     = 0
    while nr1<nr0 and i<max_iterations:
        nr0    = nr1
        df1    = gate_data(df1,nsigma=nsigma,nw=nw)
        nr1,_  = df1.shape
        i     += 1
    return df1

if __name__=='__main__':
    import os, re, argparse
    from matplotlib import rc
    rc('text', usetex=True)
    
    parser = argparse.ArgumentParser('Plot GCP wells')
    parser.add_argument('-r','--root',default=r'\data\cytoflex\Melbourne')
    parser.add_argument('-p','--plate',nargs='+',default='all')
    parser.add_argument('-w','--well',nargs='+',default=['G12','H12'])
    args   = parser.parse_args()
    
    for root, dirs, files in os.walk(args.root):
        path = root.split(os.sep)
        match = re.match('.*(((PAP)|(RTI))[A-Z]*[0-9]+)',path[-1])
        if match:
            plate=match.group(1)
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
                        df1      = purge_outliers(df)
                        well     = get_well_name(tbnm)
                        
                        if well in args.well:
                            plt.figure(figsize=(10,10))
                            plt.suptitle(f'{plate} {well}.png')
                            
                            ax1 = plt.subplot(2,2,1)
                            ax1.scatter(df1['FSC-H'],df1['SSC-H'],s=1,c='g')
                            ax1.set_xlabel('FSC-H')
                            ax1.set_ylabel('SSC-H')
                            
                            ax2 = plt.subplot(2,2,2)
                            n,bins,_ = ax2.hist(df1['Red-H'],facecolor='g',label='Observed',bins=100)
                            ax2.set_xlabel('Red-H')                        
                            
                            
                        
    if args.plate!='all':
        plt.show()    