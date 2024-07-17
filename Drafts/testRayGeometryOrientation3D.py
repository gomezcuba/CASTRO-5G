#!/usr/bin/python
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd
from tqdm import tqdm

plt.close('all')

import sys
sys.path.append('../')
from CASTRO5G import MultipathLocationEstimator
loc=MultipathLocationEstimator.MultipathLocationEstimator()

fig_ctr=0
Npath=10
x_true=np.random.rand(Npath)*40-20
y_true=np.random.rand(Npath)*40-20
z_true=np.random.rand(Npath)*5
d_true=np.column_stack([x_true,y_true,z_true])
x0_true=np.random.rand(1)*40-10
y0_true=np.random.rand(1)*40-10
z0_true=np.random.rand(1)*1+1
d0_true=np.concatenate([x0_true,y0_true,z0_true])

AoD0_true,ZoD0_true=loc.angVector(d0_true)

AoA0_true=np.random.rand(1)*2*np.pi
ZoA0_true=np.random.rand(1)*np.pi
SoA0_true=np.random.rand(1)*2*np.pi
rotation0_true=np.concatenate([AoA0_true,ZoA0_true,SoA0_true])#yaw=aoa0, pitch = 90-zenit, roll=spin
print(rotation0_true)
DoA0_true=loc.uVector(AoA0_true,ZoA0_true)
print(DoA0_true)
R0_true=loc.rMatrix(*rotation0_true)
print(R0_true)
DDoA0_true=R0_true.T@DoA0_true
print(DDoA0_true)

DoD_true=d_true.T/np.linalg.norm(d_true,axis=1)
AoD_true,ZoD_true=loc.angVector(DoD_true)
DoA_true=(d_true-d0_true).T/np.linalg.norm(d_true-d0_true,axis=1)
AoA_true,ZoA_true=loc.angVector(DoA_true)
#for debuging
# DoA_true=np.concatenate([[[0],[1],[0]],DoA_true],axis=1)
DDoA_true=R0_true.T@DoA_true
# print(DDoA_true[:,0:1])
DAoA_true,DZoA_true=loc.angVector(DDoA_true)

c=3e8
l0=np.linalg.norm(d0_true)
ToA0_true=l0/c
li=np.linalg.norm(d_true,axis=1)+np.linalg.norm(d_true-d0_true,axis=1) 
ToA_true=li/c

TDoA_true = ToA_true-ToA0_true

paths=pd.DataFrame({
    'AoD':AoD_true,
    'ZoD':ZoD_true,
    'DAoA':DAoA_true,
    'DZoA':DZoA_true,
    'TDoA':TDoA_true
    })

# fig_ctr+=1
# fig=plt.figure(fig_ctr)
# ax = fig.add_subplot(111, projection='3d')
# scaleguide=np.max(np.abs(np.concatenate([d_true,d0_true[None,:]])))

# # plt.plot([x0_true,x0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.cos(phi0_true)],[y0_true,y0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.sin(phi0_true)],'k')
# for n in range(Npath):
#     ax.plot3D([0,d_true[n,0],d0_true[0]],[0,d_true[n,1],d0_true[1]],[0,d_true[n,2],d0_true[2]],':k')
#     t=np.linspace(0,1,21)
#     plt.plot(0+scaleguide*.05*(n+1)*np.cos(AoD_true[n]*t),0+scaleguide*.05*(n+1)*np.sin(AoD_true[n]*t),0,'k')
#     plt.plot(0+scaleguide*.05*(n+1)*np.cos(AoD_true[n])*np.cos(t*(np.pi/2-ZoD_true[n])),0+scaleguide*.05*(n+1)*np.sin(AoD_true[n])*np.cos(t*(np.pi/2-ZoD_true[n])),0+scaleguide*.05*(n+1)*np.sin(t*(np.pi/2-ZoD_true[n])),'k')
    
#     plt.plot(d0_true[0]+scaleguide*.05*(n+1)*np.cos(AoA_true[n]*t),d0_true[1]+scaleguide*.05*(n+1)*np.sin(AoA_true[n]*t),d0_true[2],'k')
#     plt.plot(d0_true[0]+scaleguide*.05*(n+1)*np.cos(AoA_true[n])*np.cos(t*(np.pi/2-ZoA_true[n])),d0_true[1]+scaleguide*.05*(n+1)*np.sin(AoA_true[n])*np.cos(t*(np.pi/2-ZoA_true[n])),d0_true[2]+scaleguide*.05*(n+1)*np.sin(t*(np.pi/2-ZoA_true[n])),'k')

# ax.plot3D([0,d0_true[0]],[0,d0_true[1]],[0,d0_true[2]],':g')
# ax.plot3D(d_true[:,0],d_true[:,1],d_true[:,2],'or')
# ax.plot3D(0,0,'sb')
# ax.plot3D(d0_true[0],d0_true[1],d0_true[2],'^g')
# plt.title("All angles of a multipath channel with correct tau0, random phi0")

# searchDim=100
# AoA0_search=np.linspace(0,2*np.pi,searchDim)

# d0_4path_1D=np.zeros((searchDim,Npath-3,3))
# tauE_4path_1D=np.zeros((searchDim,Npath-3))

# for ct in tqdm(range(searchDim),desc=f'searchin 1D rotation in {searchDim} points'):
#     for gr in range(Npath-3):
#         (d0_4path_1D[ct,gr,:],tauE_4path_1D[ct,gr],_)= loc.computeAllPaths(paths[gr:gr+4],rotation=(AoA0_search[ct],rotation0_true[1],rotation0_true[2]))


# fig_ctr+=1
# fig=plt.figure(fig_ctr)
# ax = fig.add_subplot(111, projection='3d')
# for gr in range(3):
#     ax.plot(d0_4path_1D[:,gr,0],d0_4path_1D[:,gr,1],d0_4path_1D[:,gr,2],':', label='_nolegend_')
# ax.plot3D(0,0,'sb')
# ax.plot3D(d0_true[0],d0_true[1],d0_true[2],'^g')
# ax.plot3D([0,d0_true[0]],[0,d0_true[1]],[0,d0_true[2]],':g')
# ax.set_zlim(0,10)
# ax.set_xlim(-40,40)
# ax.set_ylim(-40,40)
# ax.set_xlabel('$d_{ox}$ (m)')
# ax.set_ylabel('$d_{oy}$ (m)')
# ax.set_zlabel('$d_{oz}$ (m)')
# plt.legend(['Transmitter','Receiver'])
# plt.savefig('../Figures/graphsol3DS1%d.svg'%(Npath))
# fig_ctr+=1
# fig=plt.figure(fig_ctr)
# MSD=np.sum(np.abs(d0_4path_1D-np.mean(d0_4path_1D,axis=1,keepdims=True))**2,axis=(1,2))
# plt.semilogy(AoA0_search,MSD)
# ctm=np.argmin(MSD)
# plt.plot(AoA0_true[0],MSD[ctm],'sg')
# plt.plot(AoA0_search[ctm],MSD[ctm],'xr')
# plt.xlabel("AoA0 search")
# plt.ylabel("MSD 4-path")
# plt.savefig('../Figures/graphcost3DS1%d.svg'%(Npath))


# searchDim=(100,100)
# AoA0_search=np.linspace(0,2*np.pi,searchDim[0])
# ZoA0_search=np.linspace(0,np.pi,searchDim[1])

# d0_4path_2D=np.zeros(searchDim+(Npath-3,3))
# tauE_4path_2D=np.zeros(searchDim+(Npath-3,))

# for ct in tqdm(range(np.prod(searchDim)),desc=f'searchin 2D rotation in {searchDim} points'):
#     c1,c2=np.unravel_index(ct,searchDim)
#     for gr in range(Npath-3):
#         (d0_4path_2D[c1,c2,gr,:],tauE_4path_2D[c1,c2,gr],_)= loc.computeAllPaths(paths[gr:gr+4],rotation=(AoA0_search[c1],ZoA0_search[c2],rotation0_true[2]))

# fig_ctr+=1
# fig=plt.figure(fig_ctr)
# ax = fig.add_subplot(111, projection='3d')
# for gr in range(3):
#     ax.plot(d0_4path_2D[:,:,gr,0].reshape(-1),d0_4path_2D[:,:,gr,1].reshape(-1),d0_4path_2D[:,:,gr,2].reshape(-1),':', label='_nolegend_')
# ax.plot3D(0,0,'sb')
# ax.plot3D(d0_true[0],d0_true[1],d0_true[2],'^g')
# ax.plot3D([0,d0_true[0]],[0,d0_true[1]],[0,d0_true[2]],':g')
# ax.set_zlim(0,10)
# ax.set_xlim(-40,40)
# ax.set_ylim(-40,40)
# ax.set_xlabel('$d_{ox}$ (m)')
# ax.set_ylabel('$d_{oy}$ (m)')
# ax.set_zlabel('$d_{oz}$ (m)')
# plt.legend(['Transmitter','Receiver'])
# plt.savefig('../Figures/graphsol3DS2%d.svg'%(Npath))
# fig_ctr+=1
# fig=plt.figure(fig_ctr)
# MSD=np.sum(np.abs(d0_4path_2D-np.mean(d0_4path_2D,axis=2,keepdims=True))**2,axis=(2,3))
# X,Y=np.meshgrid(AoA0_search,ZoA0_search)
# plt.pcolor(X.T,Y.T,np.log10(MSD),cmap=cm.coolwarm,linewidth=0)
# c1,c2=np.unravel_index(np.argmin(MSD),searchDim)
# plt.plot(AoA0_true[0],ZoA0_true[0],'sg')
# plt.plot(AoA0_search[c1],ZoA0_search[c2],'xr')
# plt.xlabel("AoA0 search")
# plt.ylabel("ZoA0 search")
# plt.colorbar(label="log MSD (log-m)")
# plt.savefig('../Figures/graphcost3DS2%d.svg'%(Npath))


# searchDim=(25,25,25)
# AoA0_search=np.linspace(0,2*np.pi,searchDim[0])
# ZoA0_search=np.linspace(0,np.pi,searchDim[1])
# SoA0_search=np.linspace(0,2*np.pi,searchDim[2])

# d0_4path_3D=np.zeros(searchDim+(Npath-3,3))
# tauE_4path_3D=np.zeros(searchDim+(Npath-3,))

# for ct in tqdm(range(np.prod(searchDim)),desc=f'searchin 3D rotation in {searchDim} points'):
#     c1,c2,c3=np.unravel_index(ct,searchDim)
#     for gr in range(Npath-3):
#         (d0_4path_3D[c1,c2,c3,gr,:],tauE_4path_3D[c1,c2,c3,gr],_)= loc.computeAllPaths(paths[gr:gr+4],rotation=(AoA0_search[c1],ZoA0_search[c2],SoA0_search[c3]))

# fig_ctr+=1
# fig=plt.figure(fig_ctr)
# ax = fig.add_subplot(111, projection='3d')
# for gr in range(3):
#     ax.plot(d0_4path_3D[:,:,:,gr,0].reshape(-1),d0_4path_3D[:,:,:,gr,1].reshape(-1),d0_4path_3D[:,:,:,gr,2].reshape(-1),':', label='_nolegend_')
# ax.plot3D(0,0,'sb')
# ax.plot3D(d0_true[0],d0_true[1],d0_true[2],'^g')
# ax.plot3D([0,d0_true[0]],[0,d0_true[1]],[0,d0_true[2]],':g')
# ax.set_zlim(0,10)
# ax.set_xlim(-40,40)
# ax.set_ylim(-40,40)
# ax.set_xlabel('$d_{ox}$ (m)')
# ax.set_ylabel('$d_{oy}$ (m)')
# ax.set_zlabel('$d_{oz}$ (m)')
# plt.legend(['Transmitter','Receiver'])
# plt.savefig('../Figures/graphsol3DS3%d.svg'%(Npath))
# fig_ctr+=1
# fig=plt.figure(fig_ctr)
# MSD=np.sum(np.abs(d0_4path_3D-np.mean(d0_4path_3D,axis=3,keepdims=True))**2,axis=(3,4))
# c1,c2,c3=np.unravel_index(np.argmin(MSD),searchDim)
# fig=plt.subplot(221)
# X,Y=np.meshgrid(AoA0_search,ZoA0_search)
# plt.pcolor(X.T,Y.T,np.log10(np.min(MSD,axis=2)),cmap=cm.coolwarm,linewidth=0)
# plt.plot(AoA0_true[0],ZoA0_true[0],'sg')
# plt.plot(AoA0_search[c1],ZoA0_search[c2],'xr')
# plt.xlabel("AoA0 search")
# plt.ylabel("ZoA0 search")
# fig=plt.subplot(222)
# X,Y=np.meshgrid(AoA0_search,SoA0_search)
# plt.pcolor(X.T,Y.T,np.log10(np.min(MSD,axis=1)),cmap=cm.coolwarm,linewidth=0)
# plt.plot(AoA0_true[0],SoA0_true[0],'sg')
# plt.plot(AoA0_search[c1],SoA0_search[c3],'xr')
# plt.xlabel("AoA0 search")
# plt.ylabel("SoA0 search")
# fig=plt.subplot(223)
# X,Y=np.meshgrid(ZoA0_search,SoA0_search)
# plt.pcolor(X.T,Y.T,np.log10(np.min(MSD,axis=0)),cmap=cm.coolwarm,linewidth=0)
# plt.plot(ZoA0_true[0],SoA0_true[0],'sg')
# plt.plot(ZoA0_search[c2],SoA0_search[c3],'xr')
# plt.xlabel("ZoA0 search")
# plt.ylabel("SoA0 search")
# plt.savefig('../Figures/graphcost3DS3%d.svg'%(Npath))

# d0_drop1_3D=np.zeros(searchDim+(Npath,3))
# tauE_drop1_3D=np.zeros(searchDim+(Npath,))

# for ct in tqdm(range(np.prod(searchDim)),desc=f'searchin 3D rotation in {searchDim} points'):
#     c1,c2,c3=np.unravel_index(ct,searchDim)
#     for gr in range(Npath):
#         (d0_drop1_3D[c1,c2,c3,gr,:],tauE_drop1_3D[c1,c2,c3,gr],_)= loc.computeAllPaths(paths[np.arange(Npath)!=gr],rotation=(AoA0_search[c1],ZoA0_search[c2],SoA0_search[c3]))

# fig_ctr+=1
# fig=plt.figure(fig_ctr)
# ax = fig.add_subplot(111, projection='3d')
# for gr in range(3):
#     ax.plot(d0_drop1_3D[:,:,:,gr,0].reshape(-1),d0_drop1_3D[:,:,:,gr,1].reshape(-1),d0_drop1_3D[:,:,:,gr,2].reshape(-1),':', label='_nolegend_')
# ax.plot3D(0,0,'sb')
# ax.plot3D(d0_true[0],d0_true[1],d0_true[2],'^g')
# ax.plot3D([0,d0_true[0]],[0,d0_true[1]],[0,d0_true[2]],':g')
# ax.set_zlim(0,10)
# ax.set_xlim(-40,40)
# ax.set_ylim(-40,40)
# ax.set_xlabel('$d_{ox}$ (m)')
# ax.set_ylabel('$d_{oy}$ (m)')
# ax.set_zlabel('$d_{oz}$ (m)')
# plt.legend(['Transmitter','Receiver'])
# plt.savefig('../Figures/graphsol3DS3d1%d.svg'%(Npath))
# fig_ctr+=1
# fig=plt.figure(fig_ctr)
# MSD=np.sum(np.abs(d0_drop1_3D-np.mean(d0_drop1_3D,axis=3,keepdims=True))**2,axis=(3,4))
# c1,c2,c3=np.unravel_index(np.argmin(MSD),searchDim)
# fig=plt.subplot(221)
# X,Y=np.meshgrid(AoA0_search,ZoA0_search)
# plt.pcolor(X.T,Y.T,np.log10(np.min(MSD,axis=2)),cmap=cm.coolwarm,linewidth=0)
# plt.plot(AoA0_true[0],ZoA0_true[0],'sg')
# plt.plot(AoA0_search[c1],ZoA0_search[c2],'xr')
# plt.xlabel("AoA0 search")
# plt.ylabel("ZoA0 search")
# fig=plt.subplot(222)
# X,Y=np.meshgrid(AoA0_search,SoA0_search)
# plt.pcolor(X.T,Y.T,np.log10(np.min(MSD,axis=1)),cmap=cm.coolwarm,linewidth=0)
# plt.plot(AoA0_true[0],SoA0_true[0],'sg')
# plt.plot(AoA0_search[c1],SoA0_search[c3],'xr')
# plt.xlabel("AoA0 search")
# plt.ylabel("SoA0 search")
# fig=plt.subplot(223)
# X,Y=np.meshgrid(ZoA0_search,SoA0_search)
# plt.pcolor(X.T,Y.T,np.log10(np.min(MSD,axis=0)),cmap=cm.coolwarm,linewidth=0)
# plt.plot(ZoA0_true[0],SoA0_true[0],'sg')
# plt.plot(ZoA0_search[c2],SoA0_search[c3],'xr')
# plt.xlabel("ZoA0 search")
# plt.ylabel("SoA0 search")
# plt.savefig('../Figures/graphcost3DS3d1%d.svg'%(Npath))

(d0_brute,tauE_brute,d_brute,rotation0_brute,covR0_brute)= loc.computeAllLocationsFromPaths(paths,orientationMethod='brute', orientationMethodArgs={'groupMethod':'4path','nPoint':(25,25,25)})
print(rotation0_true,rotation0_brute)
print(0,tauE_brute)
print(d0_true,d0_brute)
    
# plt.figure(6)
# plt.plot(0,0,'sb')
# plt.plot(x0_true,y0_true,'^g')
# plt.plot([0,x0_true[0]],[0,y0_true[0]],':g')
# plt.plot(d0_brute[0],d0_brute[1],'^c')
# plt.plot([0,d0_brute[0]],[0,d0_brute[1]],':c')
# plt.plot(x_true,y_true,'or')
# plt.plot(d_brute[:,0],d_brute[:,1],'oy')
# scaleguide=np.max(np.abs(np.concatenate([y_true,y0_true,x_true,x0_true],0)))
# plt.plot([x0_true,x0_true+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.cos(AoA0_true)],[y0_true,y0_true+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.sin(AoA0_true)],'k')
# plt.plot([d0_brute[0],d0_brute[0]+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.cos(AoA0_brute)],[d0_brute[1],d0_brute[1]+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.sin(AoA0_brute)],'m')

# for p in range(np.shape(AoD_true)[0]):
#     plt.plot([0,x_true[p],x0_true[0]],[0,y_true[p],y0_true[0]],':k')
#     t=np.linspace(0,1,21)
#     plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k')
#     plt.plot(x0_true+scaleguide*.05*(p+1)*np.cos(DAoA[p]*t+AoA0_true),y0_true+scaleguide*.05*(p+1)*np.sin(DAoA[p]*t+AoA0_true),'k')
#     plt.plot([0,d_brute[p,0],d0_brute[0]],[0,d_brute[p,1],d0_brute[1]],':m')
#     plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'m')
#     plt.plot(d0_brute[0]+scaleguide*.05*(p+1)*np.cos(DAoA[p]*t+AoA0_brute),d0_brute[1]+scaleguide*.05*(p+1)*np.sin(DAoA[p]*t+AoA0_brute),'m')

# plt.title("All estimations of position for the full set of multipaths, after AoA0 is estimated with bisec")

# loc.orientationMethod='lm'
# (d0_root,tauE_root,d_root,AoA0_root,covAoA0_root)= loc.computeAllLocationsFromPaths(paths,orientationMethod='lm', orientationMethodArgs={'groupMethod':'3path'})
# print(np.mod(AoA0_root,np.pi*2),AoA0_true[0])

# plt.figure(7)
# plt.plot(0,0,'sb')
# plt.plot(x0_true,y0_true,'^g')
# plt.plot([0,x0_true[0]],[0,y0_true[0]],':g')
# plt.plot(d0_root[0],d0_root[1],'^c')
# plt.plot([0,d0_root[0]],[0,d0_root[1]],':c')
# plt.plot(x_true,y_true,'or')
# plt.plot(d_root[:,0],d_root[:,1],'oy')
# scaleguide=np.max(np.abs(np.concatenate([y_true,y0_true,x_true,x0_true],0)))
# plt.plot([x0_true,x0_true+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.cos(AoA0_true)],[y0_true,y0_true+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.sin(AoA0_true)],'k')
# plt.plot([d0_root[0],d0_root[0]+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.cos(AoA0_root)],[d0_root[1],d0_root[1]+1.2*scaleguide*.05*np.shape(AoD_true)[0]*np.sin(AoA0_root)],'m')
# for p in range(np.shape(AoD_true)[0]):
#     plt.plot([0,x_true[p],x0_true[0]],[0,y_true[p],y0_true[0]],':k')
#     t=np.linspace(0,1,21)
#     plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k')
#     plt.plot(x0_true+scaleguide*.05*(p+1)*np.cos(DAoA[p]*t+AoA0_true),y0_true+scaleguide*.05*(p+1)*np.sin(DAoA[p]*t+AoA0_true),'k')
#     plt.plot([0,d_root[p,0],d0_root[0]],[0,d_root[p,1],d0_root[1]],':m')
# #    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'m')
#     plt.plot(d0_root[0]+scaleguide*.05*(p+1)*np.cos(DAoA[p]*t+AoA0_root),d0_root[1]+scaleguide*.05*(p+1)*np.sin(DAoA[p]*t+AoA0_root),'m')

# plt.title("All estimations of position for the full set of multipaths, after AoA0 is estimated with root method")
# plt.savefig('../Figures/locfromAoAAoD.svg')

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
