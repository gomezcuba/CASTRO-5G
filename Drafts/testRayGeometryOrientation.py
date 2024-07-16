#!/usr/bin/python
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm

plt.close('all')

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

plt.figure(1)
plt.plot([0,x0_true[0]],[0,y0_true[0]],':g')
plt.plot(x_true,y_true,'or')
scaleguide=np.max(np.abs(np.concatenate([y_true,y0_true,x_true,x0_true],0)))

plt.plot([x0_true[0],x0_true[0]+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.cos(AoA0_true[0])],[y0_true[0],y0_true[0]+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.sin(AoA0_true[0])],'k')
for p in range(np.shape(AoD_true)[0]):
    plt.plot([0,x_true[p],x0_true[0]],[0,y_true[p],y0_true[0]],':k')
    t=np.linspace(0,1,21)
    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k')
    plt.plot(x0_true+scaleguide*.05*(p+1)*np.cos(DAoA[p]*t+AoA0_true),y0_true+scaleguide*.05*(p+1)*np.sin(DAoA[p]*t+AoA0_true),'k')

plt.plot(0,0,'sb')
plt.plot(x0_true,y0_true,'^g')
plt.title("All angles of a multipath channel with correct ToA0, random AoA0")

loc=MultipathLocationEstimator.MultipathLocationEstimator()
AoA0_search=np.linspace(0,2*np.pi,1000)

d0_3path=np.zeros((1000,Npath-2,2))
tauE_3path=np.zeros((1000,Npath-2))

for ct in tqdm(range(AoA0_search.size),desc=f'searchin rotation using 3 path groups'):
    for gr in range(Npath-2):
        (d0_3path[ct,gr,:],tauE_3path[ct,gr],_)= loc.computeAllPaths(paths[gr:gr+3],rotation=AoA0_search[ct])

plt.figure(1)

plt.plot([0,x0_true[0]],[0,y0_true[0]],':g', label='_nolegend_')
plt.plot(d0_3path[:,:,0],d0_3path[:,:,1],':', label='_nolegend_')
plt.plot(0,0,'sb',markersize=10)
plt.plot(x0_true,y0_true,'^g',markersize=10)
plt.axis([-50,50,-50,50])
plt.xlabel('$d_{ox}$ (m)')
plt.ylabel('$d_{oy}$ (m)')
plt.legend(['Transmitter','Receiver'])
plt.savefig('../Figures/graphsol%d.svg'%(Npath))

plt.figure(2)

MSD=np.mean(np.abs(d0_3path-np.mean(d0_3path,axis=1,keepdims=True))**2,axis=(1,2))
plt.plot(AoA0_search,MSD)
plt.axis([0,2*np.pi,0,np.percentile(MSD,80)])

plt.figure(3)
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

d0_drop1=np.zeros((1000,Npath,2))
tauE_drop1=np.zeros((1000,Npath))
for ct in  tqdm(range(AoA0_search.size),desc=f'searchin rotation using drop-1 groups'):
    for gr in range(Npath):
        (d0_drop1[ct,gr,:],tauE_drop1[ct,gr],_)=loc.computeAllPaths(paths[np.arange(Npath)!=gr],rotation=AoA0_search[ct])


plt.figure(4)
plt.plot(d0_drop1[:,:,0],d0_drop1[:,:,1],':', label='_nolegend_')
plt.plot(0,0,'sb',markersize=10)
plt.plot(x0_true,y0_true,'^g',markersize=10)
plt.axis([-50,50,-50,50])
plt.xlabel('$d_{ox}$ (m)')
plt.ylabel('$d_{oy}$ (m)')
plt.legend(['Transmitter','Receiver'])
plt.savefig('../Figures/graphsoldrop1%d.svg'%(Npath))

plt.figure(5)
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


plt.figure(6)

MSD=np.mean(np.abs(d0_drop1-np.mean(d0_drop1,axis=1,keepdims=True))**2,axis=(1,2))
plt.plot(AoA0_search,MSD)
plt.axis([0,2*np.pi,0,np.percentile(MSD,80)])

(d0_brute,tauE_brute,d_brute,AoA0_brute,covAoA0_brute)= loc.computeAllLocationsFromPaths(paths,orientationMethod='brute', orientationMethodArgs={'groupMethod':'3path','nPoint':100})
print(np.mod(AoA0_brute,2*np.pi),AoA0_true[0])
    
plt.figure(7)
plt.plot(0,0,'sb')
plt.plot(x0_true,y0_true,'^g')
plt.plot([0,x0_true[0]],[0,y0_true[0]],':g')
plt.plot(d0_brute[0],d0_brute[1],'^c')
plt.plot([0,d0_brute[0]],[0,d0_brute[1]],':c')
plt.plot(x_true,y_true,'or')
plt.plot(d_brute[:,0],d_brute[:,1],'oy')
scaleguide=np.max(np.abs(np.concatenate([y_true,y0_true,x_true,x0_true],0)))
plt.plot([x0_true,x0_true+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.cos(AoA0_true)],[y0_true,y0_true+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.sin(AoA0_true)],'k')
plt.plot([d0_brute[0],d0_brute[0]+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.cos(AoA0_brute)],[d0_brute[1],d0_brute[1]+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.sin(AoA0_brute)],'m')

for p in range(np.shape(AoD_true)[0]):
    plt.plot([0,x_true[p],x0_true[0]],[0,y_true[p],y0_true[0]],':k')
    t=np.linspace(0,1,21)
    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k')
    plt.plot(x0_true+scaleguide*.05*(p+1)*np.cos(DAoA[p]*t+AoA0_true),y0_true+scaleguide*.05*(p+1)*np.sin(DAoA[p]*t+AoA0_true),'k')
    plt.plot([0,d_brute[p,0],d0_brute[0]],[0,d_brute[p,1],d0_brute[1]],':m')
    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'m')
    plt.plot(d0_brute[0]+scaleguide*.05*(p+1)*np.cos(DAoA[p]*t+AoA0_brute),d0_brute[1]+scaleguide*.05*(p+1)*np.sin(DAoA[p]*t+AoA0_brute),'m')

plt.title("All estimations of position for the full set of multipaths, after AoA0 is estimated with bisec")

loc.orientationMethod='lm'
(d0_root,tauE_root,d_root,AoA0_root,covAoA0_root)= loc.computeAllLocationsFromPaths(paths,orientationMethod='lm', orientationMethodArgs={'groupMethod':'3path'})
print(np.mod(AoA0_root,np.pi*2),AoA0_true[0])

plt.figure(7)
plt.plot(0,0,'sb')
plt.plot(x0_true,y0_true,'^g')
plt.plot([0,x0_true[0]],[0,y0_true[0]],':g')
plt.plot(d0_root[0],d0_root[1],'^c')
plt.plot([0,d0_root[0]],[0,d0_root[1]],':c')
plt.plot(x_true,y_true,'or')
plt.plot(d_root[:,0],d_root[:,1],'oy')
scaleguide=np.max(np.abs(np.concatenate([y_true,y0_true,x_true,x0_true],0)))
plt.plot([x0_true,x0_true+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.cos(AoA0_true)],[y0_true,y0_true+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.sin(AoA0_true)],'k')
plt.plot([d0_root[0],d0_root[0]+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.cos(AoA0_root)],[d0_root[1],d0_root[1]+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.sin(AoA0_root)],'m')
for p in range(np.shape(AoD_true)[0]):
    plt.plot([0,x_true[p],x0_true[0]],[0,y_true[p],y0_true[0]],':k')
    t=np.linspace(0,1,21)
    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k')
    plt.plot(x0_true+scaleguide*.05*(p+1)*np.cos(DAoA[p]*t+AoA0_true),y0_true+scaleguide*.05*(p+1)*np.sin(DAoA[p]*t+AoA0_true),'k')
    plt.plot([0,d_root[p,0],d0_root[0]],[0,d_root[p,1],d0_root[1]],':m')
#    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'m')
    plt.plot(d0_root[0]+scaleguide*.05*(p+1)*np.cos(DAoA[p]*t+AoA0_root),d0_root[1]+scaleguide*.05*(p+1)*np.sin(DAoA[p]*t+AoA0_root),'m')

plt.title("All estimations of position for the full set of multipaths, after AoA0 is estimated with root method")
plt.savefig('../Figures/locfromAoAAoD.svg')

# error_brute=np.sqrt(np.abs(x0_true-d0_brute[0])**2+np.abs(y0_true-d0_brute[1]))
# error_root=np.sqrt(np.abs(x0_true-d0_root[0])**2+np.abs(y0_true-d0_root[1]))

# print(error_brute,error_root)

# #experimental code to find the correct AoA_true as a function of TDoA , AoD_true, x0 and y0; used in mmwave sim to modify channel model
# #AoD_dif=AoD_true-AoD0_true
# #l0_true=np.sqrt(x0_true**2+y0_true**2)
# #l_true=l0_true+(ToA_true-ToA0_true)*c
# #C=l_true/np.tan(AoD_dif)-l0_true/np.sin(AoD_dif)
# #L=l_true**2
# #Z=l0_true**2
# #S=(C*l0_true+np.sqrt( Z*(C**2) - (C**2+L)*(Z-L) ))/(C**2+L)
# #S2=(C*l0_true-np.sqrt( Z*(C**2) - (C**2+L)*(Z-L) ))/(C**2+L)
# #AoA_dif=np.zeros((4,AoD_true.size))
# #AoA_dif[0,:]=np.arcsin(S)
# #AoA_dif[1,:]=np.arcsin(S2)
# #AoA_dif[2,:]=np.pi-np.arcsin(S)
# #AoA_dif[3,:]=np.pi-np.arcsin(S2)
# #
# #x=(y0_true+x0_true*np.tan(AoA_dif-AoD0_true))/(np.tan(AoD_true)+np.tan(AoA_dif-AoD0_true))
# #y=x*np.tan(AoD_true)
# #
# #dist=np.sqrt(x**2+y**2)+np.sqrt((x-x0_true)**2+(y-y0_true)**2)
# #
# #AoA_dif_final=AoA_dif[np.argmin(np.abs(dist-l_true),0),range(l_true.size)]
