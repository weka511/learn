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

import fcsparser
import matplotlib.pyplot as plt
import numpy as np

import fcs

def normalize(df,key):
    mu    = np.mean(df[key])
    sigma = np.std(df[key])
    norm  = [(x-mu)/sigma for x in df[key]]
    return norm,mu,sigma

def PCA(data):
    d       = len(data)
    m       = len(data[0])
    Cov     = np.zeros([d,d])
    for i in range(d):
        for j in range(i,d):
            Cov[i][j] = sum([x*y for (x,y) in zip(data[i],data[j]) ]) / m
            if not i==j:
                Cov[j][i]=Cov[i][j]
    U,_,_ = np.linalg.svd(Cov)
    return np.matmul(U.transpose(),data)
    
if __name__=='__main__':
    meta,df = fcsparser.parse(r'C:\data\cytoflex\Melbourne\PAP15100054\01-Tube-A12.fcs')
    df1     = fcs.gate_data(df)
    data    = np.zeros([3,df1.shape[0]])
    fsc_h,mu_fsc_h,sigma_fsc_h = normalize(df1,'FSC-H')
    ssc_h,mu_ssc_h,sigma_ssc_h = normalize(df1,'SSC-H')
    fsc_w,mu_fsc_w,sigma_fsc_w = normalize(df1,'FSC-Width')
    data[0] = [x for x in fsc_h]
    data[1] = [x for x in ssc_h]
    data[2] = [x for x in fsc_w]
    dd      = PCA(data)
    plt.scatter(dd[0],dd[1],s=1)
    plt.show()