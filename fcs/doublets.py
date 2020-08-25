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





def maximize_likelihood(xs,ys,zs,mus=[],Sigmas=[],alphas=[],K=2,N=25,limit=1.0e-6):
    def has_converged():
        return len(likelihoods)>1 and abs(likelihoods[-1]/likelihoods[-2]-1)<limit  
    def get_log_likelihood(ps):
        return sum([math.log(sum([alphas[k]*ps[k][i] for k in range(K)])) for i in range(len(xs))])
    def e_step():
        var = [multivariate_normal(mean=mus[k], cov=Sigmas[k]) for k in range(K)]
        ps  = [[var[k].pdf([xs[i],ys[i],zs[i]]) for i in range(len(xs))] for k in range(K)] 
        ws  = [[ps[k][i] * alphas[k] for i in range(len(xs))] for k in range(K)] 
        Zs  = [sum([ws[k][i] for k in range(K)]) for i in range(len(xs))]
        return [[ws[k][i]/Zs[i] for i in range(len(xs))] for k in range(K)],ps
    
    def m_step(ws):
        N       = [sum([ws[k][i] for i in range(len(xs))] ) for k in range(K)]
        alphas  = [n/sum(N) for n in N]
        mus    = [[np.average(xs,weights=ws[k]),np.average(ys,weights=ws[k]),np.average(zs,weights=ws[k])] for k in range(K)]
        Sigmas = [np.cov([xs,ys,zs],rowvar=True,aweights=ws[k]) for k in range(K)]      
        return (alphas,mus,Sigmas)    
    
    likelihoods=[]
    try:
        while len(likelihoods)<N and not has_converged():
            ws,ps = e_step()
            alphas,mus,Sigmas = m_step(ws)
            likelihoods.append(get_log_likelihood(ps))
        return True,likelihoods,ws,alphas,mus,Sigmas
    except(ValueError):
        return False, likelihoods,ws,alphas,mus,Sigmas
    
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
    failures = []
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
                            alphas=[0.5,0.5]
                            outcome,likelihoods,ws,alphas,mus,Sigmas=maximize_likelihood(xs,ys,zs,
                                                                                         mus=mus,
                                                                                         Sigmas=Sigmas,
                                                                                         alphas=alphas)                       
                            if outcome:
                                ax1    = plt.subplot(2,2,1, projection='3d')
                                sc1    = ax1.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1,c=[ws[0]], cmap=cm)
                                ax1.set_xlabel('FSC-H')
                                ax1.set_ylabel('SSC-H')
                                ax1.set_zlabel('FSC-Width')
                                ax2    = plt.subplot(2,2,2, projection='3d')
                                sc2    = ax2.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1,c=[ws[1]], cmap=cm) 
                                ax2.set_xlabel('FSC-H')
                                ax2.set_ylabel('SSC-H')
                                ax2.set_zlabel('FSC-Width')                            
                                plt.colorbar(sc1)
                                ax3    = plt.subplot(2,2,3)
                                ax3.plot(range(len(likelihoods)),likelihoods)
                                ax3.set_ylabel('log(likelihood)')
                            else:
                                ax1    = plt.subplot(2,2,1, projection='3d')
                                sc1    = ax1.scatter(df1['FSC-H'], df1['SSC-H'], df1['FSC-Width'],s=1)
                                ax1.set_xlabel('FSC-H')
                                ax1.set_ylabel('SSC-H')
                                ax1.set_zlabel('FSC-Width')
                                ax1.set_title('Failed')
                                failures.append(f'{plate} {well}')
                            plt.savefig(
                                gcps.get_image_name(
                                    script = os.path.basename(__file__).split('.')[0],
                                    plate  = plate,
                                    well   = well))                            
 
    if len(failures)>0:
        print ('Failures')
        for fail in failures:
            print (fail)
             
    if show:
        plt.show()  
        
           
