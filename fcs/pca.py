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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import fcs

def normalize(df,key):
    mu    = np.mean(df[key])
    sigma = np.std(df[key])
    norm  = [(x-mu)/sigma for x in df[key]]
    return norm,mu,sigma

def covariance(data):
    d       = len(data)
    Cov     = np.zeros([d,d])
    for i in range(d):
        for j in range(i,d):
            Cov[i][j] = sum([x*y for (x,y) in zip(data[i],data[j]) ]) / len(data[0])
            if i<j:
                Cov[j][i]=Cov[i][j]
    return Cov

def PCA(data):
    U,_,_ = np.linalg.svd(covariance(data))
    return np.matmul(U.transpose(),data)
    
if __name__=='__main__':
    meta,df   = fcsparser.parse(r'C:\data\cytoflex\Melbourne\PAP15100054\01-Tube-A12.fcs')
    df1       = fcs.gate_data(df,nsigma=2)
    data      = np.zeros([3,df1.shape[0]])
    fsc_h,_,_ = normalize(df1,'FSC-H')
    ssc_h,_,_ = normalize(df1,'SSC-H')
    fsc_w,_,_ = normalize(df1,'FSC-Width')
    data[0]   = [x for x in fsc_h]
    data[1]   = [y for y in ssc_h]
    data[2]   = [z for z in fsc_w]
    dd        = PCA(data)
    plt.figure(figsize=(10,10))
    ax1       = plt.subplot(1,1,1, projection='3d')
    ax1.scatter(dd[0],dd[1],dd[2],s=1)
    plt.show()