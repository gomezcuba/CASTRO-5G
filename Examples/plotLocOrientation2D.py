#!/usr/bin/python
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm

plt.close('all')

fig_ctr=0
import sys
sys.path.append('../')
from CASTRO5G import MultipathLocationEstimator

Npath=10
x_true=np.random.rand(Npath)*40-20
y_true=np.random.rand(Npath)*40-20
x0_true=np.random.rand(1)*40-10
y0_true=np.random.rand(1)*40-10

AoA0_true=np.random.rand(1)*2*np.pi
AoD0_true=np.mod(np.arctan(y0_true/x0_true)+np.pi*(x0_true<0) , 2*np.pi)

AoD_true=np.mod( np.arctan(y_true/x_true)+np.pi*(x_true<0), 2*np.pi)
AoA_true=np.mod( np.pi-( np.arctan((y_true-y0_true)/(x0_true-x_true))+np.pi*((x0_true-x_true)<0) ), 2*np.pi)

c=3e8
ToA0_true=y0_true/np.sin(AoD0_true)/c

ToA_true=(np.abs(y_true/np.sin(AoD_true))+np.abs((y_true-y0_true)/np.sin(np.pi-AoA_true)))/c

#AoD = np.mod(AoD_true,2*np.pi)
AoD = np.mod(AoD_true,2*np.pi)
DAoA = np.mod(AoA_true-AoA0_true,2*np.pi)
TDoA = ToA_true-ToA0_true

paths=pd.DataFrame({'DAoA':DAoA,'AoD':AoD,'TDoA':TDoA})

fig_ctr+=1
fig=plt.figure(fig_ctr)
plt.plot([0,x0_true[0]],[0,y0_true[0]],':g')
plt.plot(x_true,y_true,'or')
scaleAngRuler=.4*np.max(np.abs(np.concatenate([y_true,y0_true,x_true,x0_true],0)))

plt.plot([x0_true[0],x0_true[0]+1.2*scaleAngRuler*.05*np.shape(AoD_true)[0]*np.cos(AoA0_true[0])],[y0_true[0],y0_true[0]+1.2*scaleAngRuler*.05*np.shape(AoD_true)[0]*np.sin(AoA0_true[0])],'k')
for p in range(np.shape(AoD_true)[0]):
    plt.plot([0,x_true[p],x0_true[0]],[0,y_true[p],y0_true[0]],':k')
    t=np.linspace(0,1,21)
    plt.plot(0+scaleAngRuler*((p+1)/Npath)*np.cos(AoD[p]*t),0+scaleAngRuler*((p+1)/Npath)*np.sin(AoD[p]*t),'k')
    plt.plot(x0_true+scaleAngRuler*((p+1)/Npath)*np.cos(DAoA[p]*t+AoA0_true),y0_true+scaleAngRuler*((p+1)/Npath)*np.sin(DAoA[p]*t+AoA0_true),'k')

plt.plot(0,0,'sb')
plt.plot(x0_true,y0_true,'^g')
plt.title("All angles of a multipath channel with correct ToA0, random AoA0")

loc=MultipathLocationEstimator.MultipathLocationEstimator()
Nsearch=200
AoA0_search=np.linspace(0,2*np.pi,Nsearch)

d0_3path=np.zeros((Nsearch,Npath-2,2))
tauE_3path=np.zeros((Nsearch,Npath-2))

for ct in tqdm(range(Nsearch),desc='searchin rotation using 3 path groups'):
    for gr in range(Npath-2):
        (d0_3path[ct,gr,:],tauE_3path[ct,gr],_)= loc.computeAllPaths(paths[gr:gr+3],rotation=AoA0_search[ct])

fig_ctr+=1
fig=plt.figure(fig_ctr)

plt.plot([0,x0_true[0]],[0,y0_true[0]],':g', label='_nolegend_')
plt.plot(d0_3path[:,:,0],d0_3path[:,:,1],':', label='_nolegend_')
plt.plot(0,0,'sb',markersize=10)
plt.plot(x0_true,y0_true,'^g',markersize=10)
plt.axis([-50,50,-50,50])
plt.xlabel('$d_{ox}$ (m)')
plt.ylabel('$d_{oy}$ (m)')
plt.legend(['Transmitter','Receiver'])
plt.savefig('../Figures/graphsol%d.svg'%(Npath))

fig_ctr+=1
fig=plt.figure(fig_ctr)

MSD=np.mean(np.abs(d0_3path-np.mean(d0_3path,axis=1,keepdims=True))**2,axis=(1,2))
plt.plot(AoA0_search,MSD)
plt.axis([0,2*np.pi,0,np.percentile(MSD,80)])
ctm=np.argmin(MSD)
plt.plot(AoA0_true[0],MSD[ctm],'sg')
plt.plot(AoA0_search[ctm],MSD[ctm],'xr')
plt.xlabel("AoA0 search")
plt.ylabel("MSD 3-path")
plt.savefig('../Figures/graphcost3P%d.svg'%(Npath))

fig_ctr+=1
fig=plt.figure(fig_ctr)
ax = plt.axes(projection='3d')
for gr in range(Npath-2):
    ax.plot3D(d0_3path[:,gr,0],d0_3path[:,gr,1],ToA0_true*c+c*tauE_3path[:,gr], ':', label='_nolegend_')
ax.plot3D([0],[0],[0],'sb',markersize=10)
ax.plot3D(x0_true,y0_true,ToA0_true*c*np.ones_like(y0_true),'^g',markersize=10)
ax.set_xlim(-50,50)
ax.set_ylim(-50,50)
ax.set_zlim(0,np.max(ToA0_true*c+3e8*tauE_3path[tauE_3path>0]))
ax.set_xlabel('$d_{ox}$ (m)')
ax.set_ylabel('$d_{oy}$ (m)')
ax.set_zlabel('$\\ell_e$ (m)')
plt.legend(['Transmitter','Receiver'])
plt.savefig('../Figures/graph3Dsol%d.svg'%(Npath))

d0_drop1=np.zeros((Nsearch,Npath,2))
tauE_drop1=np.zeros((Nsearch,Npath))
for ct in  tqdm(range(Nsearch),desc='searchin rotation using drop-1 groups'):
    for gr in range(Npath):
        (d0_drop1[ct,gr,:],tauE_drop1[ct,gr],_)=loc.computeAllPaths(paths[np.arange(Npath)!=gr],rotation=AoA0_search[ct])


fig_ctr+=1
fig=plt.figure(fig_ctr)
plt.plot(d0_drop1[:,:,0],d0_drop1[:,:,1],':', label='_nolegend_')
plt.plot(0,0,'sb',markersize=10)
plt.plot(x0_true,y0_true,'^g',markersize=10)
plt.axis([-50,50,-50,50])
plt.xlabel('$d_{ox}$ (m)')
plt.ylabel('$d_{oy}$ (m)')
plt.legend(['Transmitter','Receiver'])
plt.savefig('../Figures/graphsoldrop1%d.svg'%(Npath))

fig_ctr+=1
fig=plt.figure(fig_ctr)

MSD=np.mean(np.abs(d0_drop1-np.mean(d0_drop1,axis=1,keepdims=True))**2,axis=(1,2))
plt.plot(AoA0_search,MSD)
plt.axis([0,2*np.pi,0,np.percentile(MSD,80)])
plt.plot(AoA0_true[0],MSD[ctm],'sg')
plt.plot(AoA0_search[ctm],MSD[ctm],'xr')
plt.xlabel("AoA0 search")
plt.ylabel("MSD drop1")
plt.savefig('../Figures/graphcostD1%d.svg'%(Npath))

fig_ctr+=1
fig=plt.figure(fig_ctr)
ax = plt.axes(projection='3d')
for gr in range(Npath):
    ax.plot3D(d0_drop1[:,gr,0],d0_drop1[:,gr,1],ToA0_true*c+c*tauE_drop1[:,gr], ':', label='_nolegend_')
ax.plot3D([0],[0],[0],'sb',markersize=10)
ax.plot3D(x0_true,y0_true,ToA0_true*c*np.ones_like(y0_true),'^g',markersize=10)
ax.set_xlim(-50,50)
ax.set_ylim(-50,50)
ax.set_zlim(0,np.max(ToA0_true*c+c*tauE_drop1[tauE_drop1>0]))
ax.set_xlabel('$d_{ox}$ (m)')
ax.set_ylabel('$d_{oy}$ (m)')
ax.set_zlabel('$\\ell_e$ (m)')
plt.legend(['Transmitter','Receiver'])
plt.savefig('../Figures/graph3Dsoldrop1%d.svg'%(Npath))

(d0_brute,tauE_brute,d_brute,AoA0_brute,covAoA0_brute)= loc.computeAllLocationsFromPaths(paths,orientationMethod='brute', orientationMethodArgs={'groupMethod':'3path','nPoint':100})
print(np.mod(AoA0_brute,2*np.pi),AoA0_true[0])
    
fig_ctr+=1
fig=plt.figure(fig_ctr)
plt.plot(0,0,'sb')
plt.plot(x0_true,y0_true,'^g')
plt.plot([0,x0_true[0]],[0,y0_true[0]],':g')
plt.plot(d0_brute[0],d0_brute[1],'^c')
plt.plot([0,d0_brute[0]],[0,d0_brute[1]],':c')
plt.plot(x_true,y_true,'or')
plt.plot(d_brute[:,0],d_brute[:,1],'oy')
scaleAngRuler=0.4*np.max(np.abs(np.concatenate([y_true,y0_true,x_true,x0_true],0)))
plt.plot([x0_true,x0_true+1.2*scaleAngRuler*.05*np.shape(AoD_true)[0]*np.cos(AoA0_true)],[y0_true,y0_true+1.2*scaleAngRuler*.05*np.shape(AoD_true)[0]*np.sin(AoA0_true)],'k')
plt.plot([d0_brute[0],d0_brute[0]+1.2*scaleAngRuler*.05*np.shape(AoD_true)[0]*np.cos(AoA0_brute)],[d0_brute[1],d0_brute[1]+1.2*scaleAngRuler*.05*np.shape(AoD_true)[0]*np.sin(AoA0_brute)],'m')

for p in range(np.shape(AoD_true)[0]):
    plt.plot([0,x_true[p],x0_true[0]],[0,y_true[p],y0_true[0]],':k')
    t=np.linspace(0,1,21)
    plt.plot(0+scaleAngRuler*((p+1)/Npath)*np.cos(AoD[p]*t),0+scaleAngRuler*((p+1)/Npath)*np.sin(AoD[p]*t),'k')
    plt.plot(x0_true+scaleAngRuler*((p+1)/Npath)*np.cos(DAoA[p]*t+AoA0_true),y0_true+scaleAngRuler*((p+1)/Npath)*np.sin(DAoA[p]*t+AoA0_true),'k')
    plt.plot([0,d_brute[p,0],d0_brute[0]],[0,d_brute[p,1],d0_brute[1]],':m')
    plt.plot(d0_brute[0]+scaleAngRuler*((p+1)/Npath)*np.cos(DAoA[p]*t+AoA0_brute),d0_brute[1]+scaleAngRuler*((p+1)/Npath)*np.sin(DAoA[p]*t+AoA0_brute),'m')

plt.title("All estimations of position for the full set of multipaths, after AoA0 is estimated with brute")
plt.savefig('../Figures/locfromDAoAAoDbrute.svg')

loc.orientationMethod='lm'
(d0_root,tauE_root,d_root,AoA0_root,covAoA0_root)= loc.computeAllLocationsFromPaths(paths,orientationMethod='lm', orientationMethodArgs={'groupMethod':'3path'})
print(np.mod(AoA0_root,np.pi*2),AoA0_true[0])

fig_ctr+=1
fig=plt.figure(fig_ctr)
plt.plot(0,0,'sb')
plt.plot(x0_true,y0_true,'^g')
plt.plot([0,x0_true[0]],[0,y0_true[0]],':g')
plt.plot(d0_root[0],d0_root[1],'^c')
plt.plot([0,d0_root[0]],[0,d0_root[1]],':c')
plt.plot(x_true,y_true,'or')
plt.plot(d_root[:,0],d_root[:,1],'oy')
scaleAngRuler=0.4*np.max(np.abs(np.concatenate([y_true,y0_true,x_true,x0_true],0)))
plt.plot([x0_true,x0_true+1.2*scaleAngRuler*.05*np.shape(AoD_true)[0]*np.cos(AoA0_true)],[y0_true,y0_true+1.2*scaleAngRuler*.05*np.shape(AoD_true)[0]*np.sin(AoA0_true)],'k')
plt.plot([d0_root[0],d0_root[0]+1.2*scaleAngRuler*.05*np.shape(AoD_true)[0]*np.cos(AoA0_root)],[d0_root[1],d0_root[1]+1.2*scaleAngRuler*.05*np.shape(AoD_true)[0]*np.sin(AoA0_root)],'m')
for p in range(np.shape(AoD_true)[0]):
    plt.plot([0,x_true[p],x0_true[0]],[0,y_true[p],y0_true[0]],':k')
    t=np.linspace(0,1,21)
    plt.plot(0+scaleAngRuler*((p+1)/Npath)*np.cos(AoD[p]*t),0+scaleAngRuler*((p+1)/Npath)*np.sin(AoD[p]*t),'k')
    plt.plot(x0_true+scaleAngRuler*((p+1)/Npath)*np.cos(DAoA[p]*t+AoA0_true),y0_true+scaleAngRuler*((p+1)/Npath)*np.sin(DAoA[p]*t+AoA0_true),'k')
    plt.plot([0,d_root[p,0],d0_root[0]],[0,d_root[p,1],d0_root[1]],':m')
    plt.plot(d0_root[0]+scaleAngRuler*((p+1)/Npath)*np.cos(DAoA[p]*t+AoA0_root),d0_root[1]+scaleAngRuler*((p+1)/Npath)*np.sin(DAoA[p]*t+AoA0_root),'m')

plt.title("All estimations of position for the full set of multipaths, after AoA0 is estimated with root method")
plt.savefig('../Figures/locfromDAoAAoDroot.svg')

error_brute=np.sqrt(np.abs(x0_true-d0_brute[0])**2+np.abs(y0_true-d0_brute[1]))
error_root=np.sqrt(np.abs(x0_true-d0_root[0])**2+np.abs(y0_true-d0_root[1]))

print(error_brute,error_root)