#!/usr/bin/python

import threeGPPMultipathGenerator as mp3g
import multipathChannel as ch
import OMPCachedRunner as oc
import MIMOPilotChannel as pil
import csProblemGenerator as prb
#import testRLmp as rl

import matplotlib.pyplot as plt
import numpy as np

import time
from progress.bar import Bar
import pandas as pd

Nd=4
Na=4
Nt=8
Nxp=3
Nrft=1
Nrfr=2
K=16
Ts=300/Nt#2.5
Ds=Ts*Nt
sigma2=.01

omprunner = oc.OMPCachedRunner()
pilgen = pil.MIMOPilotChannel("UPhase")

(w,v)=pilgen.generatePilots((K,Nxp,Nrfr,Na,Nd,Nrft),"UPhase")

chgen = mp3g.ThreeGPPMultipathChannelModel()
chgen.bLargeBandwidthOption=True

mpch = chgen.create_channel((0,0,10),(30,0,1.5))
Npath = len(mpch.channelPaths)

paths=pd.DataFrame({
        'coef': [x.complexAmplitude[0] for x in mpch.channelPaths],
        'AoD': [x.azimutOfDeparture[0] for x in mpch.channelPaths],
        'AoA': [x.azimutOfArrival[0] for x in mpch.channelPaths],
        'delay': [x.excessDelay[0] for x in mpch.channelPaths],
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
        tau_term=-2j*np.pi* np.arange(K)[...,np.newaxis,np.newaxis,np.newaxis] *np.exp(-2j*np.pi* np.arange(K)[...,np.newaxis,np.newaxis,np.newaxis] * p.delay[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])
    else:
        tau_term=np.exp(-2j*np.pi* np.arange(K)[...,np.newaxis,np.newaxis,np.newaxis] * p.delay[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])
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

npth=40
partialD=np.concatenate((
        diffGen(paths[0:npth],w,v,'tau').reshape(npth,-1),
        diffGen(paths[0:npth],w,v,'theta').reshape(npth,-1),
        diffGen(paths[0:npth],w,v,'phi').reshape(npth,-1)
        ),axis=0)

J=np.real(partialD@partialD.conj().T)
sigma2=1e-2

MSE=np.trace(np.linalg.inv(J))*sigma2/K/Nd/Na/Nxp
print("MSE: %f"%MSE)