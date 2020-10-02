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
from matplotlib import rc
import numpy as np
import seaborn
import fcs

def normalize(vector):
    mu    = np.mean(vector)
    sigma = np.std(vector)
    norm  = [(x-mu)/sigma for x in vector]
    return norm,mu,sigma

def create_covariance(data):
    d       = len(data)
    product = np.zeros([d,d])
    for i in range(d):
        for j in range(i,d):
            product[i][j]     = sum([x*y for (x,y) in zip(data[i],data[j]) ]) / len(data[0])
            if i<j:
                product[j][i] = product[i][j]
    return product

def PCA(data):
    U,_,_ = np.linalg.svd(create_covariance(data))
    Ut    = U.transpose()
    return np.matmul(Ut,data),Ut
   
def create_data(df,fields):
    product = np.zeros([len(fields),df.shape[0]])
    for i in range(len(fields)):
        channel_data,_,_ = normalize(df[fields[i]])
        product[i] = [x for x in channel_data]
    return product

if __name__=='__main__':
    rc('text', usetex=True)    
    _,df      = fcsparser.parse(r'C:\data\cytoflex\Melbourne\PAP15100054\01-Tube-A12.fcs')
    raw       = create_data(fcs.gate_data(df,nsigma=2), ['FSC-H','SSC-H','FSC-Width'])
    reduced,_ = PCA(raw)
    
    plt.figure(figsize=(10,20))
    plt.title('PCA')
    cm  = plt.cm.get_cmap('plasma')
    ax1 = plt.subplot(1,2,1, projection='3d')
    sc1 = ax1.scatter(reduced[0],reduced[1],reduced[2],
                s=1,c=reduced[2],cmap=cm)
    cbar = plt.colorbar(sc1,shrink=0.25)
    cbar.set_label('$3^{rd}$ component',rotation=270)
    ax2 = plt.subplot(1,2,2)
    sc2 = ax2.scatter(reduced[0],reduced[1],
                      s=1,c=reduced[2],cmap=cm)
    ax2.set_title('Principal components')
    plt.tight_layout()
    plt.show()