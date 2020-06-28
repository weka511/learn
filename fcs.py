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

def purge_outliers(df,nsigma=3,nw=2):
    nr0,_ = df.shape
    df1   = gate_data(df,nsigma=nsigma)
    nr1,_ = df1.shape
    
    while nr1<nr0:
        nr0    = nr1
        df1    = gate_data(df1,nsigma=nsigma,nw=nw)
        nr1,_  = df1.shape
        
    return df1

def get_well_name(tbnm):
    return tbnm[-3:]

def plot_norm(ax=None,df=None,channel='SSC-H',bins=None):
    ax_twin = ax.twinx()
    h       = sorted(df[channel])
    hmean   = np.mean(h)
    hstd    = np.std(h)
    h       = [(0.5*(a+b) - 0) for (a,b) in zip(bins[:-1],bins[1:])]
    h_n     = [(0.5*(a+b) - hmean) / hstd for (a,b) in zip(bins[:-1],bins[1:])]
    pdf     = stats.norm.pdf( h_n )
    ax_twin.plot(h, pdf, c='r',lw=1)
    return pdf

if __name__=='__main__':
    import os, re, argparse
    
    parser = argparse.ArgumentParser('Plot GCP wells')
    parser.add_argument('-r','--root',default=r'\data\cytoflex\Melbourne')
    parser.add_argument('-p','--plate',nargs='+',default='all')
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
                        
                        plt.figure(figsize=(10,10))
                        plt.suptitle(f'{plate} {get_well_name(tbnm)}.png')
                        
                        ax1=plt.subplot(2,2,1)
                        n,bins,_ = ax1.hist(df1['FSC-H'],facecolor='g',bins=25)
                        ax1.set_xlabel('FSC-H')
                        nn               = sum(n)
                        pdf              = plot_norm(ax=ax1, df=df1, channel='FSC-H',bins=bins)
                        npdf             = sum(pdf)
                        freqs            = [(nn/npdf)*p for p in pdf]                 
                        nf               = sum(freqs)
                        statistic,pvalue = stats.chisquare(n,freqs)
                        ax1.set_title(f'{statistic:.2f},{pvalue:.2f}')
                        
                        ax2    =plt.subplot(2,2,2)
                        n,bins,_ = ax2.hist(df1['SSC-H'],facecolor='g')
                        ax2.set_xlabel('SSC-H')
                        
                        nn               = sum(n)
                        pdf              = plot_norm(ax=ax2, df=df1, channel='SSC-H',bins=bins)
                        npdf             = sum(pdf)
                        freqs            = [(nn/npdf)*p for p in pdf]                 
                        nf               = sum(freqs)
                        statistic,pvalue = stats.chisquare(n,freqs)
                        ax2.set_title(f'{statistic:.2f},{pvalue:.2f}')
                        
                        ax3=plt.subplot(2,2,3)
                        ax3.scatter(df1['FSC-H'],df1['SSC-H'],s=1,c='g')
                        ax3.set_xlabel('FSC-H')
                        ax3.set_ylabel('SSC-H')
                        
                        ax4=plt.subplot(2,2,4)
                        ax4.scatter(df1['FSC-H'],df1['FSC-Width'],s=1,c='g')
                        ax4.set_xlabel('FSC-H')
                        ax4.set_ylabel('FSC-Width')
                        
                        plt.savefig(os.path.join('figs',f'{plate}-{get_well_name(tbnm)}.png'))
                        if args.plate=='all':
                            plt.close()
                        
    if args.plate!='all':
        plt.show()