#!/usr/bin/python
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

import numpy as np
import time
import pandas as pd
from tqdm import tqdm

plt.close('all')

import sys
sys.path.append('../')
from CASTRO5G import MultipathLocationEstimator

Npath=20
Nsims=100

Ndim=2#3
#random locations in a 40m square
d0=np.random.rand(Nsims,Ndim)*100-50
d=np.random.rand(Nsims,Npath,Ndim)*100-50

#angles from locations
AoD0=np.arctan2(d0[:,1],d0[:,0])
AoD=np.arctan2(d[:,:,1],d[:,:,0])
AoA=np.arctan2( d[:,:,1]-d0[:,None,1], d[:,:,0]-d0[:,None,0])

#delays based on distance
c=3e8
ToA0=np.linalg.norm(d0,axis=1)/c
ToA=(np.linalg.norm(d,axis=2)+np.linalg.norm(d-d0[:,None,:],axis=2))/c

#typical channel multipath estimation outputs
#AoD = np.mod(AoD,2*np.pi)
Tserr=2.5e-9
Nanterr=1024
AoD = np.mod(AoD+np.random.rand(Nsims,Npath)*2*np.pi/Nanterr,2*np.pi)
AoA0=np.random.rand(Nsims)*2*np.pi #receiver angular measurement offset
DAoA = np.mod(AoA-AoA0[:,None]+np.random.rand(Nsims,Npath)*2*np.pi/Nanterr,2*np.pi)
clock_error=(40/c)*np.random.rand(Nsims) #delay estimation error
del_error=(Tserr)*np.random.randn(Nsims,Npath) #delay estimation error
TDoA = ToA-ToA0[:,None]+clock_error[:,None]+del_error

loc=MultipathLocationEstimator.MultipathLocationEstimator(nPoint=100,orientationMethod='lm')

allPathsData = pd.DataFrame(index=pd.MultiIndex.from_product([np.arange(Nsims),np.arange(Npath)]),
                            data={
                                "AoD" : AoD.reshape(-1),
                                "DAoA" : DAoA.reshape(-1),
                                "TDoA" : TDoA.reshape(-1)
                                })

t_start_b = time.time()
AoA0_b=np.zeros(Nsims)
tauE_b=np.zeros(Nsims)
d0_b=np.zeros((Nsims,Ndim))
d_b=np.zeros((Nsims,Npath,Ndim))
for nsim in tqdm(range(Nsims),desc="brute"):
    (d0_b[nsim,:],tauE_b[nsim],d_b[nsim,:,:],AoA0_b[nsim],_)= loc.computeAllLocationsFromPaths(allPathsData.loc[nsim,:] ,orientationMethod='brute', orientationMethodArgs={'groupMethod':'3path','nPoint':100})
error_brute=np.linalg.norm(d0-d0_b,axis=1)
t_run_b = time.time() - t_start_b
plt.figure(1)
plt.semilogx(np.sort(error_brute).T,np.linspace(0,1,error_brute.size),'b')

t_start_r = time.time()
AoA0_r=np.zeros(Nsims)
tauE_r=np.zeros(Nsims)
d0_r=np.zeros((Nsims,Ndim))
d_r=np.zeros((Nsims,Npath,Ndim))
for nsim in tqdm(range(Nsims),desc="lm 3P"):
    (d0_r[nsim,:],tauE_r[nsim],d_r[nsim,:,:],AoA0_r[nsim],_)= loc.computeAllLocationsFromPaths(allPathsData.loc[nsim,:] ,orientationMethod='lm', orientationMethodArgs={'groupMethod':'3path'})
error_root=np.linalg.norm(d0-d0_r,axis=1)
t_run_r = time.time() - t_start_r
plt.semilogx(np.sort(error_root).T,np.linspace(0,1,error_root.size),'-.r')


t_start_r2 = time.time()
AoA0_r2=np.zeros(Nsims)
tauE_r2=np.zeros(Nsims)
d0_r2=np.zeros((Nsims,Ndim))
d_r2=np.zeros((Nsims,Npath,Ndim))
for nsim in tqdm(range(Nsims),desc="lm D1"):
    (d0_r2[nsim,:],tauE_r2[nsim],d_r2[nsim,:,:],AoA0_r2[nsim],_)= loc.computeAllLocationsFromPaths(allPathsData.loc[nsim,:] ,orientationMethod='lm', orientationMethodArgs={'groupMethod':'drop1'})
error_root2=np.linalg.norm(d0-d0_r2,axis=1)
t_run_r2 = time.time() - t_start_r2
plt.semilogx(np.sort(error_root2).T,np.linspace(0,1,error_root2.size),'--sk')

t_start_h= time.time()
AoA0_h=np.zeros(Nsims)
tauE_h=np.zeros(Nsims)
d0_h=np.zeros((Nsims,Ndim))
d_h=np.zeros((Nsims,Npath,Ndim))
AoA0_coarse=np.round(AoA0*256/np.pi/2)*np.pi*2/256
for nsim in tqdm(range(Nsims),desc="lm D1h"):
    (d0_h[nsim,:],tauE_h[nsim],d_h[nsim,:,:],AoA0_h[nsim],_)= loc.computeAllLocationsFromPaths(allPathsData.loc[nsim,:] ,orientationMethod='lm', orientationMethodArgs={'groupMethod':'drop1','initRotation':AoA0_coarse[nsim]})
error_rooth=np.linalg.norm(d0-d0_h,axis=1)
t_run_h = time.time() - t_start_h
plt.semilogx(np.sort(error_rooth).T,np.linspace(0,1,error_rooth.size),':xk')


t_start_3p= time.time()
d0_3p=np.zeros((Nsims,Npath-2,Ndim))
d_3p=np.zeros((Nsims,Npath-2,3,Ndim))
tauE_3p=np.zeros((Nsims,Npath-2))
for nsim in tqdm(range(Nsims),desc='lin every 3 paths mean'):
    for gr in range(Npath-2):
        (d0_3p[nsim,gr,:],tauE_3p[nsim,gr],d_3p[nsim,gr,:,:])= loc.computeAllPaths(allPathsData.loc[nsim][gr:gr+3],rotation=AoA0[nsim])
d0_3p=np.mean(d0_3p,axis=1)
error_3p=np.linalg.norm(d0-d0_3p,axis=1)
t_run_3p = time.time() - t_start_3p
plt.semilogx(np.sort(error_3p).T,np.linspace(0,1,error_3p.size),':or')


t_start_l= time.time()
tauE_l=np.zeros(Nsims)
d0_l=np.zeros((Nsims,Ndim))
d_l=np.zeros((Nsims,Npath,Ndim))
for nsim in tqdm(range(Nsims),desc="lin all"):
    (d0_l[nsim,:],tauE_l[nsim],d_l[nsim,:,:])= loc.computeAllPaths(allPathsData.loc[nsim],rotation=AoA0[nsim])
error_l=np.linalg.norm(d0-d0_l,axis=1)
t_run_l = time.time() - t_start_l
plt.semilogx(np.sort(error_l).T,np.linspace(0,1,error_l.size),':xg')


error_dumb=np.linalg.norm(d0-d[:,0,:],axis=1)
plt.semilogx(np.sort(error_dumb).T,np.linspace(0,1,error_dumb.size),':k')

plt.xlabel('Location error(m)')
plt.ylabel('C.D.F.')
plt.legend(['brute $\\textnormal{AoA}_o$ 3path','lm $\\textnormal{AoA}i_o$ 3path','lm $\\textnormal{AoA}_o$ linear','lm $\\textnormal{AoA}_o$ linhint','$\\textnormal{AoA}_o$ known, 3path','$\\textnormal{AoA}_o$ known, linear','randomguess'])
plt.savefig('../Figures/cdflocgeosim.svg')

plt.figure(2)

# plt.plot(np.vstack((d0[:,0],d0_b[:,0])),np.vstack((d0[:,1],d0_b[:,1])),':xr',label='brute')
# plt.plot(np.vstack((d0[:,0],d0_r[:,0])),np.vstack((d0[:,1],d0_r[:,1])),':+k',label='lm')
plt.plot(np.vstack((d0[:,0],d0_r2[:,0])),np.vstack((d0[:,1],d0_r2[:,1])),':+g',label='lm $\\textnormal{AoA}_o$ linear')
plt.plot(np.vstack((d0[:,0],d0_l[:,0])),np.vstack((d0[:,1],d0_l[:,1])),':xr',label='known $\\textnormal{AoA}_o$ linear')
plt.plot(d0[:,0].T,d0[:,1].T,'ob',label='locations')
handles, labels = plt.gca().get_legend_handles_labels()
# labels will be the keys of the dict, handles will be values
temp = {k:v for k,v in zip(labels, handles)}
plt.legend(temp.values(), temp.keys(), loc='best')
plt.savefig('../Figures/errormap.svg')

plt.figure(3)
plt.semilogx(np.sort(AoA0-AoA0_b).T,np.linspace(0,1,AoA0.size),'r')
plt.semilogx(np.sort(AoA0-AoA0_r).T,np.linspace(0,1,AoA0.size),'-.b')
plt.semilogx(np.sort(AoA0-AoA0_r2).T,np.linspace(0,1,AoA0.size),':xg')
plt.xlabel('$\\textnormal{AoA}_o-\hat{\\textnormal{AoA}}_o$ (rad)')
plt.ylabel('C.D.F.')
plt.legend(['brute $\\textnormal{AoA}_o$ 3path','lm $\\textnormal{AoA}_o$ 3path','lm $\\textnormal{AoA}_o$ linear'])
plt.savefig('../Figures/cdfpsi0err.svg')

plt.figure(4)
plt.loglog(np.abs(np.mod(AoA0.T,2*np.pi)-np.mod(AoA0_b.T,2*np.pi)),error_brute.T,'xr')
plt.loglog(np.abs(np.mod(AoA0.T,2*np.pi)-np.mod(AoA0_r.T,2*np.pi)),error_root.T,'ob')
plt.loglog(np.abs(np.mod(AoA0.T,2*np.pi)-np.mod(AoA0_r2.T,2*np.pi)),error_root2.T,'sg')
plt.xlabel('$\\textnormal{AoA}_o-\hat{\\textnormal{AoA}}_o$ (rad)')
plt.ylabel('Location error (m)')
plt.legend(['brute $\\textnormal{AoA}_o$ 3path','lm $\\textnormal{AoA}o$ 3path','lm $\\textnormal{AoA}_o$ linear'])
plt.savefig('../Figures/corrpsi0-Poserr.svg')
