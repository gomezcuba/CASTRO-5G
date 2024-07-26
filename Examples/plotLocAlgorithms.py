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

bScenario3D = True
bMultipathErrors = True

loc=MultipathLocationEstimator.MultipathLocationEstimator(nPoint=100,orientationMethod='lm',disableTQDM= True)

Npath=20
Nsims=20

Ndim= 3 if bScenario3D else 2
#random locations in a 40m square
Dmax0=np.array([100,100,1.5])
Dmin0=np.array([-100,-100,0])
Dmax=np.array([100,100,15])
Dmin=np.array([-100,-100,-5])
d0=np.random.rand(Nsims,Ndim)*(Dmax0-Dmin0)+Dmin0
d=np.random.rand(Nsims,Npath,Ndim)*(Dmax-Dmin)+Dmin

#delays based on distance
c=3e8
ToA0=np.linalg.norm(d0,axis=1)/c
ToA=(np.linalg.norm(d,axis=2)+np.linalg.norm(d-d0[:,None,:],axis=2))/c
TDoA = ToA-ToA0[:,None]

#angles from locations
DoD=d/np.linalg.norm(d,axis=2,keepdims=True)
DoA=(d-d0[:,None,:])/np.linalg.norm( d-d0[:,None,:] ,axis=2,keepdims=True)

if bScenario3D:
    AoD0,ZoD0=loc.angVector(d0)
    AoD,ZoD=loc.angVector(DoD)
    RoT0=np.random.rand(Nsims,3)*np.array([2,1,2])*np.pi #receiver angular measurement offset
    R0=np.array([ loc.rMatrix(*x) for x in RoT0])
    DDoA=DoA@R0 #transpose of R0.T @ DoA.transpose([0,2,1])
    DAoA,DZoA=loc.angVector(DDoA)
else:    
    AoD0=loc.angVector(d0)
    AoD=loc.angVector(DoD)
    AoA=loc.angVector(DoA)
    RoT0=np.random.rand(Nsims)*2*np.pi #receiver angular measurement offset
    R0=np.array([ loc.rMatrix(x) for x in RoT0])
    DDoA=DoA@R0 #transpose of R0.T @ DoA.transpose([0,2,1])
    DAoA=loc.angVector(DDoA)
    # DAoA=AoA-RoT0[:,None]
    # print(np.isclose(DAoA,np.mod(AoA-RoT0[:,None]+np.pi,2*np.pi)-np.pi))
    

if bMultipathErrors:
    #dictionary channel multipath estimation outputs modelled as Additive Uniform Noise
    maxSynchErr=40/c # 40m at lightspeed in seconds
    Tserr=2.5e-9 # Nyquist sampling period for B=400MHz
    Nanterr=1024 # Overcomplete beamforming codebook with 1024 beams
    clock_error=maxSynchErr*np.random.rand(Nsims) #delay estimation error
    TDoA_error=(Tserr)*np.random.randn(Nsims,Npath) #delay estimation error
    TDoA = TDoA + clock_error[:,None]+TDoA_error
    AoD = np.mod(AoD+np.random.rand(Nsims,Npath)*2*np.pi/Nanterr,2*np.pi)
    DAoA = np.mod(DAoA+np.random.rand(Nsims,Npath)*2*np.pi/Nanterr,2*np.pi)
    if bScenario3D:
        ZoD = np.mod(ZoD+np.random.rand(Nsims,Npath)*2*np.pi/Nanterr,2*np.pi)
        DZoA = np.mod(DZoA+np.random.rand(Nsims,Npath)*2*np.pi/Nanterr,2*np.pi)

allPathsData = pd.DataFrame(index=pd.MultiIndex.from_product([np.arange(Nsims),np.arange(Npath)]),
                            data={
                                "AoD" : AoD.reshape(-1),
                                "DAoA" : DAoA.reshape(-1),
                                "TDoA" : TDoA.reshape(-1)
                                })
if bScenario3D:
    allPathsData['ZoD'] = ZoD.reshape(-1)
    allPathsData['DZoA'] = DZoA.reshape(-1)
    
RoT0_coarse=np.round(RoT0*32/np.pi/2)*np.pi*2/32
#TODO until we find faster algorithms for brute force 3D orientation, BF with large size and LM without hint should not be used
tableMethods=[
    # ("BF 3P","brute",{'groupMethod':'4path' if bScenario3D else '3path','nPoint':(10,10,10) if bScenario3D else 100},'-','x','r'),
    # ("BF D1","brute",{'groupMethod':'drop1','nPoint':(20,10,20) if bScenario3D else 100},'-.','s','r'),
    # ("BF D1","brute",{'groupMethod':'ortho3','nPoint':(20,10,20) if bScenario3D else 100},'--','s','r'),
    # ("LM 3P","lm",{'groupMethod':'4path' if bScenario3D else '3path'},'-','*','g'),
    # ("LM D1","lm",{'groupMethod':'drop1'},'-.','o','g'),
    # ("LMh 3P","lm",{'groupMethod':'4path' if bScenario3D else '3path','initRotation':None},'-','+','b'),
    ("LMh D1","lm",{'groupMethod':'drop1','initRotation':None},'-.','d','b'),
    # ("LMh O3","lm",{'groupMethod':'ortho3','initRotation':None},'--','+','b'),
    ("lin oracle","linear",RoT0,'-','^','k'),
    ("lin gyro","linear",RoT0_coarse,'-.','v','k')
    ]
Nmeth=len(tableMethods)
RoT0_est=np.zeros((Nmeth,Nsims,3 if bScenario3D else 1))
tauE_est=np.zeros((Nmeth,Nsims))
d0_est=np.zeros((Nmeth,Nsims,Ndim))
d_est=np.zeros((Nmeth,Nsims,Npath,Ndim))
run_times=np.zeros((Nmeth,Nsims))
for m in range(Nmeth):
    legStr,meth,arg,_,_,_=tableMethods[m]    
    for nsim in tqdm(range(Nsims),desc=legStr):
        t_start=time.time()
        if meth=='linear':
            RoT0_est[m,nsim,:]=arg[nsim]
            (d0_est[m,nsim,:],tauE_est[m,nsim],d_est[m,nsim,:,:])= loc.computeAllPaths(allPathsData.loc[nsim],rotation=RoT0_est[m,nsim,:])        
        else:
            if 'initRotation' in arg.keys():
                arg['initRotation']=RoT0_coarse[nsim]
            (d0_est[m,nsim,:],tauE_est[m,nsim],d_est[m,nsim,:,:],RoT0_est[m,nsim,:],_)= loc.computeAllLocationsFromPaths(allPathsData.loc[nsim,:] ,orientationMethod=meth, orientationMethodArgs=arg)
        run_times[m,nsim]=time.time()-t_start

error_loc=np.linalg.norm(d0_est-d0,axis=-1)
error_map=np.linalg.norm(d_est-d,axis=-1)
# error_clock=np.abs(tauE_est-clock_error)
error_Rot0=np.linalg.norm( np.abs(np.mod(RoT0.reshape(Nsims,-1),2*np.pi)-np.mod(RoT0_est,2*np.pi)) ,axis=-1)

plt.figure(1)
for m in range(Nmeth):
    legStr,meth,arg,lin,mrk,clr=tableMethods[m]    
    plt.semilogx(np.sort(error_loc[m,:]),np.linspace(0,1,Nsims),linestyle=lin,color=clr,label=legStr)
drandom=np.random.rand(Nsims,Ndim)*100-50
error_dumb=np.linalg.norm(d0-drandom,axis=1)
plt.semilogx(np.sort(error_dumb).T,np.linspace(0,1,error_dumb.size),':k',label="Random location")

plt.xlabel('Location error(m)')
plt.ylabel('C.D.F.')
plt.legend()
plt.savefig('../Figures/cdflocgeosim.svg')

fig=plt.figure(2)
if bScenario3D:
    ax=fig.add_subplot(111, projection='3d')    
    for m in range(Nmeth):
        legStr,meth,arg,lin,mrk,clr=tableMethods[m]  
        for nsim in range(Nsims):        
            ax.plot3D([d0[nsim,0],d0_est[m,nsim,0]],[d0[nsim,1],d0_est[m,nsim,1]],[d0[nsim,2],d0_est[m,nsim,2]],linestyle=':',color=clr,marker=mrk,label=legStr)    
    for nsim in range(Nsims):        
        ax.plot([d0[nsim,0],d0[nsim,0]],[d0[nsim,1],d0[nsim,1]],[d0[nsim,2],0],':k',label='locations')
    ax.plot(d0[:,0],d0[:,1],d0[:,2],'ok',label='locations')    
else:
    for m in range(Nmeth):
        legStr,meth,arg,lin,mrk,clr=tableMethods[m]  
        plt.plot(np.vstack((d0[:,0],d0_est[m,:,0])),np.vstack((d0[:,1],d0_est[m,:,1])),linestyle=':',color=clr,marker=mrk,label=legStr)
    plt.plot(d0[:,0].T,d0[:,1].T,'ok',label='locations')
handles, labels = plt.gca().get_legend_handles_labels()
# # labels will be the keys of the dict, handles will be values
temp = {k:v for k,v in zip(labels, handles)}
plt.legend(temp.values(), temp.keys(), loc='best')
plt.savefig('../Figures/errormap.svg')

plt.figure(3)
for m in range(Nmeth):
    legStr,meth,arg,lin,mrk,clr=tableMethods[m] 
    plt.semilogx(np.sort(error_Rot0[m,:]).T,np.linspace(0,1,Nsims),linestyle=lin,color=clr,label=legStr)
plt.xlabel('$\\textnormal{AoA}_o-\\hat{\\textnormal{AoA}}_o$ (rad)')
plt.ylabel('C.D.F.')
plt.legend()
plt.savefig('../Figures/cdfpsi0err.svg')

plt.figure(4)
for m in range(Nmeth):
    legStr,meth,arg,lin,mrk,clr=tableMethods[m] 
    plt.loglog(error_Rot0[m,:],error_loc[m,:],linestyle='',marker=mrk,color=clr,label=legStr)
    # plt.loglog(np.abs(np.mod(RoT0.T,2*np.pi)-np.mod(RoT0_b.T,2*np.pi)),error_loc[m,:],'*r')
plt.xlabel('$\\textnormal{AoA}_o-\\hat{\\textnormal{AoA}}_o$ (rad)')
plt.ylabel('Location error (m)')
plt.legend()
plt.savefig('../Figures/corrpsi0-Poserr.svg')