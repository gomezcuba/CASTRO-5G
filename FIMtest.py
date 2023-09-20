#!/usr/bin/python


from CASTRO5G import threeGPPMultipathGenerator as mp3g
import MIMOPilotChannel as pil
from CASTRO5G import OMPCachedRunner as oc
import numpy as np
import pandas as pd

Nd=16
Na=16
Nt=128
Nxp=3
Nrft=1
Nrfr=2
K=128
Ts=300/Nt#2.5
Ds=Ts*Nt
sigma2=.01

omprunner = oc.OMPCachedRunner()
pilgen = pil.MIMOPilotChannel("UPhase")

(w,v)=pilgen.generatePilots((K,Nxp,Nrfr,Na,Nd,Nrft),"UPhase")

chgen = mp3g.ThreeGPPMultipathChannelModel()
model = mp3g.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=False)
plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
tau,powC,AOA,AOD,ZOA,ZOD = clusters.T.to_numpy()
nClusters=tau.size
los, PLfree, SF = plinfo
tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths.T.to_numpy()
Npath = pow_sp.size

paths=pd.DataFrame({
        'coef': np.sqrt(pow_sp)*np.exp(-2j*np.pi*np.random.rand((Npath))),
        'AoD': AOD_sp*np.pi/180,
        'AoA': AOA_sp*np.pi/180,
        'delay': tau_sp,
        })

p2=paths[0:2]

def beta(w,ang):
    Nant=w.shape[-1]
    return(w@np.exp(-1j*np.pi*np.arange(Nant)[:,np.newaxis]*np.sin(ang)))
    
def d_beta(w,ang):
    Nant=w.shape[-1]
    return(w@(-1j*np.pi*np.arange(Nant)[:,np.newaxis]*np.cos(ang)*np.exp(-1j*np.pi*np.arange(Nant)[:,np.newaxis]*np.sin(ang))))

#def diff_AOD(p,w,v):
#    return( p.coef[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] *
#           np.exp(-2j*np.pi* np.arange(K)[...,np.newaxis,np.newaxis,np.newaxis] * p.delay[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]) *
#           beta(w,p.AoA[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]) * 
#           d_beta( np.transpose(v,(0,1,3,2)) ,p.AoD[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])
#           )
#def diff_AOA(p,w,v):
#    return( p.coef[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] *
#           np.exp(-2j*np.pi* np.arange(K)[...,np.newaxis,np.newaxis,np.newaxis] * p.delay[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]) *
#           d_beta(w,p.AoA[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]) * 
#           beta( np.transpose(v,(0,1,3,2)) ,p.AoD[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])
#           )
    
def diffGen(p,w,v,dAxis='tau'):
    if dAxis=='tau':
        tau_term=-2j*np.pi* np.arange(K)[...,np.newaxis,np.newaxis,np.newaxis] *np.exp(-2j*np.pi* np.arange(K)[...,np.newaxis,np.newaxis,np.newaxis] * p.delay[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]/Ts)
    else:
        tau_term=np.exp(-2j*np.pi* np.arange(K)[...,np.newaxis,np.newaxis,np.newaxis] * p.delay[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]/Ts)
    if dAxis=='theta':
        theta_term=d_beta( np.transpose(v,(0,1,3,2)) ,p.AoD[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])
    else:
        theta_term=beta( np.transpose(v,(0,1,3,2)) ,p.AoD[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])
    if dAxis=='phi':
        phi_term=beta(w,p.AoA[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])
    else:
        phi_term=d_beta(w,p.AoA[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])
    return(p.coef[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] * tau_term * theta_term * phi_term)
        
print(diffGen(paths,w,v,'theta').shape)
print(diffGen(paths,w,v,'phi').shape)
print(diffGen(paths,w,v,'tau').shape)

npth=5
partialD=np.concatenate((
        diffGen(paths[0:npth],w,v,'tau').reshape(npth,-1),
        diffGen(paths[0:npth],w,v,'theta').reshape(npth,-1),
        diffGen(paths[0:npth],w,v,'phi').reshape(npth,-1)
        ),axis=0)

sigma2=1e-2
J=2*np.real(partialD@partialD.conj().T)*sigma2*K*Nd*Na*Nxp

pathMSE=np.trace(np.linalg.inv(J))
print("Path param MSE: %f"%pathMSE)

pos0=np.array([30,0])#we will not actually test with a consistent case, so the MSE will be bad
posPaths=np.random.rand(npth,2)
def diffTau(p0,pP):
    c=3e-1#in m/ns
    return (p0/np.linalg.norm(p0-pP,axis=1)[:,np.newaxis]/c)

def diffPhi(p0,pP):
    g=(p0[1]-pP[:,1])/(p0[0]-pP[:,0])
    dgx= 1/(p0[0]-pP[:,0])
    dgy= -1/((p0[0]-pP[:,0])**2)
    return np.concatenate([dgx[:,np.newaxis],dgy[:,np.newaxis]],axis=1) * 1/(1+g[:,np.newaxis]**2)

print(diffTau(pos0,posPaths).shape)
print(diffPhi(pos0,posPaths).shape)

#diffTheta / d p0 turns out to be zero
partialD_notheta=np.concatenate((
        diffGen(paths[0:npth],w,v,'tau').reshape(npth,-1),
        diffGen(paths[0:npth],w,v,'phi').reshape(npth,-1)
        ),axis=0)

J_notheta=2*np.real(partialD_notheta@partialD_notheta.conj().T) *K*Nd*Na*Nxp

partialP=np.concatenate((
        diffTau(pos0,posPaths),
        diffPhi(pos0,posPaths)
        ),axis=0)

J2=partialP.conj().T @ J_notheta @ partialP
posMSE=np.trace(np.linalg.inv(J2))
print("Path position MSE: %f"%posMSE)