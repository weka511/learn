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

import fcsparser,fcs,gcps,numpy as np,math
from scipy.stats import multivariate_normal

def sqdist(p1,p2,d=3):
    return sum ([(p1[i]-p2[i])**2 for i in range(d)])

def e_step(xs,ys,zs,mus=[],Sigmas=[],alphas=[],K=2):
    var = [multivariate_normal(mean=mus[k], cov=Sigmas[k]) for k in range(K)]
    ps  = [[var[k].pdf([xs[i],ys[i],zs[i]]) for i in range(len(xs))] for k in range(K)] 
    ws  = [[ps[k][i] for i in range(len(xs))] for k in range(K)] 
    Zs  = [sum([ws[k][i] for k in range(K)]) for i in range(len(xs))]
    return [[ws[k][i]/Zs[i] for i in range(len(xs))] for k in range(K)],ps

def m_step(xs,ys,zs,ws,K=2):
    N       = [sum([ws[k][i] for i in range(len(xs))] ) for k in range(K)]
    alphas  = [n/sum(N) for n in N]
    mus    = [[np.average(xs,weights=ws[k]),np.average(ys,weights=ws[k]),np.average(zs,weights=ws[k])] for k in range(K)]
    Sigmas = [np.cov([xs,ys,zs],rowvar=True,aweights=ws[k]) for k in range(K)]      
    return (alphas,mus,Sigmas)

def get_log_likelihood(ps,mus=[],Sigmas=[],alphas=[],K=2):
    return sum([math.log(sum([alphas[k]*ps[k][i] for k in range(K)])) for i in range(len(xs))])

def maximize_likelihood(xs,ys,zs,mus=[],Sigmas=[],alphas=[],K=2,N=25,limit=1.0e-6):
    likelihoods=[]
    for i in range(N):
        ws,ps = e_step(xs,ys,zs,mus=mus,Sigmas=Sigmas,alphas=alphas)
        alphas,mus,Sigmas = m_step(xs,ys,zs,ws)
        likelihoods.append(get_log_likelihood(ps,mus=mus,Sigmas=Sigmas,alphas=alphas))
    return likelihoods,ws

if __name__=='__main__':
    import os, re, argparse,sys,matplotlib.pyplot as plt
    from matplotlib import rc
    from mpl_toolkits.mplot3d import Axes3D
    #from matplotlib import cm
    cm = plt.cm.get_cmap('RdYlBu')
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
                             
                            #n_f,bins_f = np.histogram(df1['FSC-H'],bins =100)                            
                            #n_s,bins_s = np.histogram(df1['SSC-H'],bins =100)                        
                            #n_w,bins_w = np.histogram(df1['FSC-Width'],bins =100)                   
 
                            K      = 2
                            d      = 3
                            mult   = 0.25
                            xs     = df1['FSC-H'].values
                            ys     = df1['SSC-H'].values
                            zs     = df1['FSC-Width'].values
                            mu     = [np.mean(xs),np.mean(ys),np.mean(zs)]
                            sigma  = [np.std(xs),np.std(ys),np.std(zs)]
                            mus    = [[mu[i]+ direction*mult*sigma[i] for i in range(d)] for direction in [-1,+1]]
                            clust  = [0 if sqdist(p,mus[0]) < sqdist(p,mus[1]) else 1 for p in zip(xs,ys,zs)]
                            alphas = [[1,0] if a==0 else [0,1] for a in clust]
                            Sigmas = [np.cov([xs,ys,zs],rowvar=True,aweights=[a[0] for a in alphas]),
                                      np.cov([xs,ys,zs],rowvar=True,aweights=[a[1] for a in alphas])]
                            
                            #ax1    = plt.subplot(2,2,1, projection='3d')
                            #ax1.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1,c=[a[0] for a in alphas],alpha=0.5)                            
                            #ax1.scatter(mus[0][0],mus[0][1],mus[0][2],c='r',alpha=0.5) 
                            #ax1.scatter(mus[1][0],mus[1][1],mus[1][2],c='b',alpha=0.5)
                            
                            likelihoods,ws=maximize_likelihood(xs,ys,zs,mus=mus,Sigmas=Sigmas,alphas=alphas)
                            
                            #ws,ps = e_step(xs,ys,zs,mus=mus,Sigmas=Sigmas,alphas=alphas)
                            ax1    = plt.subplot(2,2,1, projection='3d')
                            sc1=ax1.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1,c=[ws[0]], cmap=cm)
                            ax2    = plt.subplot(2,2,2, projection='3d')
                            sc2 = ax2.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1,c=[ws[1]], cmap=cm) 
                            plt.colorbar(sc1)
                            
                            #alphas,mus,Sigmas = m_step(xs,ys,zs,ws)
                            
                            #print (get_log_likelihood(ps,mus=mus,Sigmas=Sigmas,alphas=alphas))
                            #ws,ps = e_step(xs,ys,zs,mus=mus,Sigmas=Sigmas,alphas=alphas)
                            #ax3    = plt.subplot(2,2,3, projection='3d')
                            #ax3.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1,c=[ws[0]])
                            #ax4    = plt.subplot(2,2,4, projection='3d')
                            #ax4.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1,c=[ws[1]])
                            #alphas,mus,Sigmas = m_step(xs,ys,zs,ws)
                            
                            #print (get_log_likelihood(ps,mus=mus,Sigmas=Sigmas,alphas=alphas))                            
                            
                            plt.savefig(
                                gcps.get_image_name(
                                    script = os.path.basename(__file__).split('.')[0],
                                    plate  = plate,
                                    well   = well))                            
    
    if show:
        plt.show()  
        
    #K     = 1
    #xs    = df1['FSC-H'].values
    #ys    = df1['SSC-H'].values
    #zs    = df1['FSC-Width'].values
    #mu    = [np.mean(xs),np.mean(ys),np.mean(zs)]
    #sigma = np.cov([xs,ys,zs],rowvar=True)                                                    
    #var   = multivariate_normal(mean=mu, cov=sigma)
    #ps    = [var.pdf([xs[i],ys[i],zs[i]]) for i in range(len(df1['FSC-H']))]  
    #ax1   = plt.subplot(2,2,1, projection='3d')
    #ax1.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1,c=ps)                            
    #ax1.set_xlabel('FSC-H')
    #ax1.set_ylabel('SSC-H')
    #ax1.set_zlabel('FSC-Width')
    
    #for i in range(3):
        #mu    = [np.average(xs,weights=ps),np.average(ys,weights=ps),np.average(zs,weights=ps)]
        #sigma = np.cov([xs,ys,zs],rowvar=True,aweights=ps)                                                    
        #var   = multivariate_normal(mean=mu, cov=sigma)                            
        #ps    = [var.pdf([xs[i],ys[i],zs[i]]) for i in range(len(df1['FSC-H']))] 
        #ax2   = plt.subplot(2,2,2+i, projection='3d')
        #ax2.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1,c=ps)                            
        #ax2.set_xlabel('FSC-H')
        #ax2.set_ylabel('SSC-H')
        #ax2.set_zlabel('FSC-Width')                            
