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

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import os 
import re
import scipy.stats as stats


#  get_bounds
#
# Used to clip data into band near to mean

def get_bounds(df,channel,nsigma=3):
    mean   = np.mean(df[channel])
    std    = np.std(df[channel])
    return (mean-nsigma*std,mean+nsigma*std)

# gate_data
#
# Used to clip data

def gate_data(df,nsigma=3,nw=2):
    fsc_min,fsc_max = get_bounds(df,'FSC-H',nsigma=nsigma)
    ssc_min,ssc_max = get_bounds(df,'SSC-H',nsigma=nsigma)
    _,fsc_width_max = get_bounds(df,'FSC-Width',nsigma=nw)
    
    return df[(fsc_min         < df['FSC-H']) & \
              (df['FSC-H']     < fsc_max)     & \
              (ssc_min         < df['SSC-H']) & \
              (df['SSC-H']     < ssc_max)     & \
              (df['FSC-Width'] < fsc_width_max)]

# purge_outliers
#

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

# get_well_name
#
# Extract well name from tube name

def get_well_name(tbnm):
    m = re.match('.?([A-H][1-9][012]?)',tbnm[-3:])
    if m:
        return m.group(1)

# get_image_name
#
# Function used to name image files
#
# Parameters:
#
#     script
#     plate
#     well
#     K
#     file_type

def get_image_name(script=None,plate=None,well=None,K=None,file_type='png'):
    return os.path.join('figs',
                        f'{script}-{plate}-{well}.{file_type}' if K==None else f'{script}-{plate}-{well}-{K}.{file_type}')

def chi_sq(n,bins=[],df=None, channel='FSC-H'):
    assert len(n)+1==len(bins)
    mu    = np.mean(df[channel])
    sigma = np.std(df[channel])
    cdf   = stats.norm.cdf( bins, loc=mu, scale=sigma ) 
    nn    = sum(n)
    freqs = [nn*(b-a) for (a,b) in zip(cdf[:-1],cdf[1:]) ]
    statistic,pvalue = stats.chisquare(n,freqs)
    return statistic,pvalue,freqs

def consolidate(n,bins,minimum=5):
    n1    = []
    bins1 = [bins[0]]
    assert len(n)+1== len(bins), f'{len(n)} {len(bins)}'
    total = 0
    for n0,b in zip(n,bins[1:]):
        total+=n0
        if total>=minimum:
            n1.append(total)
            total = 0
            bins1.append(b)
    if 0 < total and total<minimum:
        n1[-1]+=total
        bins1[-1]=bins[-1] 
    return n1,bins1

# fcs
#
# A generator: it allows us to iterate through plates and wells

def fcs(root,
        plate = 'all',
        wells = 'all',
        locations = [
            'Albuquerque',
            'London',
            'Melbourne',
            'NewDelhi']):
    def parse_wells():
        if isinstance(wells,str) or len(wells)==1:
            return (re.compile(
                          '.*([A-H][1]?[0-9]).fcs'  if wells=='all'       else \
                          '.*([A-H]12).fcs'         if wells== 'controls' else \
                          '.*([GH]12).fcs' ) ,       # GCPs
                    [])
        else:
            re_w = re.compile('.*([A-H][1]?[0-9])')
            for w in wells:
                if not re_w.match(w):
                    raise Exception(f'{w} is not a valid well')            
            return (re.compile('.*([A-H][1]?[0-9]).fcs'),wells)
        
 

    # get_well
    #
    # Used to extract well number from file name
    
    def get_well(file_name):
        match   = re_wells.match(file_name)
        matched = match and (len(well_list)==0 or match.group(1) in well_list)
        return match.group(1) if matched else None
    
    def get_location(path):
        for component in path:
            if component in locations:
                return component
    
    # gcps_first
    #
    # Used to sort wells so GCPs come first (as we may want some data from GCPs before we process other wells)
    def gcps_first(couple):
        _,well_name = couple
        if well_name == 'G12': return -2
        if well_name == 'H12': return -1
        row    = ord(well_name[0])-ord('A')
        column = int(well_name[1:])
        return 12 * row + column
    
    re_plate =  re.compile('.*(((PAP)|(RTI))[A-Z]*[0-9]+[r]?)')
    
    re_wells,well_list = parse_wells() 
    
    for root, dirs, files in os.walk(root):   
        path     = root.split(os.sep)
        match    = re_plate.match(path[-1])
        location = get_location(path)
        if match:
            this_plate = match.group(1)
            if plate == 'all' or 'all' in plate or this_plate in plate:
                for file,well in sorted([(file,get_well(file)) for file in files if get_well(file) != None],
                                     key = gcps_first):
                    path     = os.path.join(root,file)
                    meta, df = fcsparser.parse(path, reformat_meta=True)
                    yield this_plate,well,df,meta,location
                if this_plate in plate: return
                
if __name__=='__main__':

    rc('text', usetex=True)
    
    parser = argparse.ArgumentParser('Plot GCP wells')
    parser.add_argument('-r','--root',default=r'\data\cytoflex\Melbourne')
    parser.add_argument('-p','--plate',nargs='+',default='all')
    args   = parser.parse_args()

    for root, dirs, files in os.walk(args.root):
        path  = root.split(os.sep)
        match = re.match('.*(((PAP)|(RTI))[A-Z]*[0-9]+)',path[-1])
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
                        
                        df1      = purge_outliers(df)
                        
                        plt.figure(figsize=(10,10))
                        plt.suptitle(f'{plate} {get_well_name(tbnm)}.png')
                        
                        ax1=plt.subplot(2,2,1)
                        n,bins,_ = ax1.hist(df1['FSC-H'],facecolor='g',bins=25,label='Observed')
                        n,bins   = consolidate(n,bins)
                        ax1.set_xlabel('FSC-H')

                        statistic,pvalue,freqs = chi_sq(n,bins=bins,df=df1, channel='FSC-H')
                        ax1.plot([0.5*(a+b) for (a,b) in zip(bins[:-1],bins[1:])],freqs, c='r',lw=1,label='Gaussian')
                        ax1.set_title(r'$\chi^2=${0:.2f}, p={1:.2f}'.format(statistic,pvalue))
                        ax1.legend()
                        
                        ax2    =plt.subplot(2,2,2)
                        n,bins,_ = ax2.hist(df1['SSC-H'],facecolor='g',label='Observed')
                        n,bins   = consolidate(n,bins)
                        ax2.set_xlabel('SSC-H')
                        
                        statistic,pvalue,freqs = chi_sq(n,bins=bins, df=df1, channel='SSC-H')
                        ax2.plot([0.5*(a+b) for (a,b) in zip(bins[:-1],bins[1:])],freqs, c='r',lw=1,label='Gaussian')
                        ax2.set_title(r'$\chi^2=${0:.2f}, p={1:.2f}'.format(statistic,pvalue))
                        ax2.legend()
                        
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