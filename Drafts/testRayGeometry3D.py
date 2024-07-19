#!/usr/bin/python
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import time

import sys
sys.path.append('../')
from CASTRO5G import MultipathLocationEstimator

####################################################
# Generate true values
# Npath=1
# d0=np.array([10,0,1.5])
# d=np.array([5,5,1.5]).reshape(1,-1)
Npath=4#minimum 4 paths for linear location with clock error
d0=np.concatenate([np.random.rand(2)*40-10,[1.5]])
d=np.random.rand(Npath,3)*40-20
dinv = d-d0

AoD0=np.arctan2(d0[1],d0[0])
ZoD0=np.arctan2(np.linalg.norm( d0[0:2], axis=0 ) , d0[2] )
AoA0=np.random.rand(1)*2*np.pi
ZoA0=np.random.rand(1)*np.pi
SoA0=np.random.rand(1)*2*np.pi

AoD=np.arctan2(d[:,1],d[:,0])
ZoD=np.arctan2(np.linalg.norm( d[:,0:2], axis=1 ) , d[:,2] )
AoA=np.arctan2(dinv[:,1],dinv[:,0])
ZoA=np.arctan2(np.linalg.norm( dinv[:,0:2], axis=1 ) , dinv[:,2])

c=3e8
l0=np.linalg.norm(d0)
ToA0 = l0/c
liD = np.linalg.norm(d, axis=1)
liA = np.linalg.norm(dinv, axis=1)
li = liD + liA
ToA = li/c
tauE = np.random.uniform(-1,1,1)*40e-9
TDoA = ToA - ToA0 + tauE



fig=plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot3D([0,d0[0]],[0,d0[1]],[0,d0[2]],':g')
ax.plot3D(d[:,0],d[:,1],d[:,2],'or')
scaleguide=np.max(np.abs(np.concatenate([d,d0[None,:]])))

# plt.plot([x0_true,x0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.cos(phi0_true)],[y0_true,y0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.sin(phi0_true)],'k')
for n in range(Npath):
    ax.plot3D([0,d[n,0],d0[0]],[0,d[n,1],d0[1]],[0,d[n,2],d0[2]],':k')
    t=np.linspace(0,1,21)
    plt.plot(0+scaleguide*.05*(n+1)*np.cos(AoD[n]*t),0+scaleguide*.05*(n+1)*np.sin(AoD[n]*t),0,'k')
    plt.plot(0+scaleguide*.05*(n+1)*np.cos(AoD[n])*np.cos(t*(np.pi/2-ZoD[n])),0+scaleguide*.05*(n+1)*np.sin(AoD[n])*np.cos(t*(np.pi/2-ZoD[n])),0+scaleguide*.05*(n+1)*np.sin(t*(np.pi/2-ZoD[n])),'k')
    
    plt.plot(d0[0]+scaleguide*.05*(n+1)*np.cos(AoA[n]*t),d0[1]+scaleguide*.05*(n+1)*np.sin(AoD[n]*t),d0[2],'k')
    plt.plot(d0[0]+scaleguide*.05*(n+1)*np.cos(AoA[n])*np.cos(t*(np.pi/2-ZoA[n])),d0[1]+scaleguide*.05*(n+1)*np.sin(AoA[n])*np.cos(t*(np.pi/2-ZoA[n])),d0[2]+scaleguide*.05*(n+1)*np.sin(t*(np.pi/2-ZoA[n])),'k')

ax.plot3D(0,0,'sb')
ax.plot3D(d0[0],d0[1],d0[2],'^g')
plt.title("All angles of a multipath channel with correct tau0, random phi0")

def uVectorT(A,Z):
    return( np.column_stack([np.cos(A)*np.sin(Z),np.sin(A)*np.sin(Z),np.cos(Z)]) )

DoD = uVectorT(AoD,ZoD)
DoA = uVectorT(AoA,ZoA)
Ui = np.stack([DoD,-DoA],axis=-1) # [npath,vDimensions,D->A]
M = -np.ones((Npath,4))
for n in range(Npath):
    print(f'Distance match {np.isclose(ToA[n]*c,np.sum( np.linalg.lstsq(Ui[n,:,:], d0, rcond=None)[0]))}' )
    dagUi = np.linalg.lstsq(Ui[n,:,:], np.eye(3), rcond=None)[0]
    vDagUi = np.sum(dagUi,axis=0)
    # M[n,0:3] = vDagUi
    C12=np.dot(Ui[n,:,0],Ui[n,:,1])
    A=np.array([[1,-C12],[-C12,1]])/(1-C12**2)
    print(f'Fast inverse 1 match {np.all(np.isclose( dagUi, A@Ui[n,:,:].T ))}' )
    Vi=np.array([
        Ui[n,:,0]-C12*Ui[n,:,1],
        Ui[n,:,1]-C12*Ui[n,:,0],
        ])/(1-C12**2)    
    print(f'Fast inverse 2 match {np.all(np.isclose( dagUi, Vi ))}' )
    print(f'Fast inverse-sum match {np.all(np.isclose( vDagUi, (DoD[n,:]-DoA[n,:])/(1+C12) ))}')
    M[n,0:3] = (DoD[n,:]-DoA[n,:])/(1+C12)

def testSpeedLSTSQ(U):
    Npath=U.shape[0]
    M = -np.ones((Npath,3))
    for n in range(Npath):
        dagUi = np.linalg.lstsq(Ui[n,:,:], np.eye(3), rcond=None)[0]
        vDagUi = np.sum(dagUi,axis=0)
        M[n,:] = vDagUi
    return(M)
%timeit testSpeedLSTSQ(Ui)
def testSpeedExpression(DoD,DoA):
    C12= np.sum(-DoA*DoD,axis=1,keepdims=True)
    M=(DoD-DoA)/(1+C12)
    return(M)
%timeit testSpeedExpression(DoD,DoA)

DoD = uVectorT(AoD,ZoD)
DoA = uVectorT(AoA,ZoA) 
C12= np.sum(-DoD*DoA,axis=1,keepdims=True)
M=np.column_stack([(DoD-DoA)/(1+C12),-np.ones((Npath,1))])
d0_est = np.linalg.lstsq(M, TDoA*c, rcond=None)[0]
print(f'Location match { np.isclose(d0 , d0_est[0:3])}' )
Vi=(DoD+C12*DoA)/(1-C12**2)
liD_est=Vi@d0_est[0:3]
print(f'Ref dist match {np.all(np.isclose(liD, liD_est))}' )
d_est=liD_est[:,None]*DoD
print(f'Reflectors match {np.all(np.isclose(d, d_est))}' )

loc=MultipathLocationEstimator.MultipathLocationEstimator()
DAoA=AoA-AoA0
paths=pd.DataFrame({'DAoA':DAoA,'AoD':AoD,'TDoA':TDoA})
d0_old,ToAE_old,d_old=loc.computeAllPathsV1wrap(paths,rotation=AoA0)
d0_est,ToAE_est,d_est=loc.computeAllPaths(paths,rotation=AoA0)

print(f'2D Functions match {np.all(np.isclose(d0_old, d0_est))}  {np.all(np.isclose(ToAE_old, ToAE_est))}  {np.all(np.isclose(d_old, d_est))}' )

%timeit loc.computeAllPathsV1(AoD,DAoA,TDoA,AoA0)
%timeit loc.computeAllPaths(paths,rotation=AoA0)

rotation0_1D=np.concatenate([AoA0,[np.pi/2],[0]])
R0_1D=loc.rMatrix(*rotation0_1D)
DDoA_1D=( R0_1D.T@DoA.T ).T
DAoA_1D,ZoA_1D=loc.angVector(DDoA_1D)
print(f'3D AoA 1D-rotation is equivalent to additive {np.all(np.isclose(np.mod(DAoA,2*np.pi), np.mod(DAoA_1D,2*np.pi)))}, {np.all(np.isclose(np.mod(ZoA,2*np.pi), np.mod(ZoA_1D,2*np.pi)))}' )
paths_1D=pd.DataFrame({'DAoA':DAoA,'AoD':AoD,'TDoA':TDoA,'ZoD':ZoD,'DZoA':ZoA})
d0_est,ToAE_est,d_est=loc.computeAllPaths(paths_1D,rotation=rotation0_1D)
print(f'3D Location 1D-rotation-H match {np.all(np.isclose(d0, d0_est))}  {np.all(np.isclose(tauE, ToAE_est))}  {np.all(np.isclose(d, d_est))}' )

rotation0_1DZ=np.concatenate([[0],ZoA0,[0]])
DZoA_additive=ZoA-ZoA0
R0_1DZ=loc.rMatrix(*rotation0_1DZ)
DDoA_1DZ=( R0_1DZ.T@DoA.T ).T
DAoA_1DZ,DZoA_1DZ=loc.angVector(DDoA_1DZ)
#the trick here is that a ZoA displacement above the Z axis flips AoA by pi
print(f'3D ZoA 1D-rotation is NOT equivalent to additive {np.all(np.isclose(np.mod(AoA,2*np.pi), np.mod(DAoA_1DZ,2*np.pi)))}, {np.all(np.isclose(np.mod(DZoA_additive,2*np.pi), np.mod(DZoA_1DZ,2*np.pi)))}' )

paths_1DZ=pd.DataFrame({'DAoA':DAoA_1DZ,'AoD':AoD,'TDoA':TDoA,'ZoD':ZoD,'DZoA':DZoA_1DZ})
d0_est,ToAE_est,d_est=loc.computeAllPaths(paths_1DZ,rotation=rotation0_1DZ)
print(f'3D Location 1D-rotation-V match {np.all(np.isclose(d0, d0_est))}  {np.all(np.isclose(tauE, ToAE_est))}  {np.all(np.isclose(d, d_est))}' )

rotation0_2D=np.concatenate([AoA0,ZoA0,[0]])
R0_2D=loc.rMatrix(*rotation0_2D)
DDoA_2D=( R0_2D.T@DoA.T ).T
DAoA_2D,DZoA_2D=loc.angVector(DDoA_2D)
paths_2D=pd.DataFrame({'DAoA':DAoA_2D,'AoD':AoD,'TDoA':TDoA,'ZoD':ZoD,'DZoA':DZoA_2D})
d0_est,ToAE_est,d_est=loc.computeAllPaths(paths_2D,rotation=rotation0_2D)
print(f'3D Location 2D-rotation match {np.all(np.isclose(d0, d0_est))}  {np.all(np.isclose(tauE, ToAE_est))}  {np.all(np.isclose(d, d_est))}' )

rotation0_3D=np.concatenate([AoA0,ZoA0,SoA0])
R0_3D=loc.rMatrix(*rotation0_3D)
DDoA_3D=( R0_3D.T@DoA.T ).T
DAoA_3D,DZoA_3D=loc.angVector(DDoA_3D)
paths_3D=pd.DataFrame({'DAoA':DAoA_3D,'AoD':AoD,'TDoA':TDoA,'ZoD':ZoD,'DZoA':DZoA_3D})
d0_est,ToAE_est,d_est=loc.computeAllPaths(paths_3D,rotation=rotation0_3D)
print(f'3D Location 3D-rotation match {np.all(np.isclose(d0, d0_est))}  {np.all(np.isclose(tauE, ToAE_est))}  {np.all(np.isclose(d, d_est))}' )

