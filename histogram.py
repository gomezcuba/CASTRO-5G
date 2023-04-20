# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:48:43 2022

@author: user
"""

import threeGPPMultipathGenerator as pg

import matplotlib.pyplot as plt
import numpy as np
import math as mt
plt.close('all')
model = pg.ThreeGPPMultipathChannelModel(sce="UMi")
model.bLargeBandwidthOption=True

txPos = (0,0,25)
lista = []
lista_tau = []
rxPos = (np.random.uniform(0, 100),np.random.uniform(0, 100),1.5)

aPos = np.array(txPos)
bPos = np.array(rxPos)
vLOS = bPos-aPos

losphiAoD=np.mod( np.arctan( vLOS[1] / vLOS[0] )+np.pi*(vLOS[0]<0), 2*np.pi )
losphiAoA=np.mod(np.pi+losphiAoD, 2*np.pi ) # revise
vaux = (np.linalg.norm(vLOS[0:2]), vLOS[2] )
losthetaAoD=np.pi/2-np.arctan( vaux[1] / vaux[0] )
losthetaAoA=np.pi-losthetaAoD # revise

#3GPP model is in degrees but numpy uses radians
losphiAoD=(180.0/np.pi)*losphiAoD #angle of departure 
losthetaAoD=(180.0/np.pi)*losthetaAoD 
losphiAoA=(180.0/np.pi)*losphiAoA #angle of aperture
losthetaAoA=(180.0/np.pi)*losthetaAoA
angles = [losphiAoD,losphiAoA,losthetaAoD,losthetaAoA]

d2D = np.linalg.norm(bPos[0:-1]-aPos[0:-1])

macro = model.create_macro(txPos,rxPos)

listaTau = []
for i in range(1000):
    clusters, subpaths =model.create_small_param(angles,macro,d2D)
    nClusters,tau,powC,AOA,AOD,ZOA,ZOD =clusters
    listaTau.append(tau[1:])
    

tau_doubled = np.concatenate(listaTau)
plt.hist(tau_doubled, bins=20, density=True)
aux_x = np.linspace(0,np.max(tau_doubled),101)
lambda_tau = 1/(macro.ds*model.scenarioParamsLOS.rt)
plt.plot(aux_x,lambda_tau*np.exp(-aux_x*lambda_tau),'r:', label = "PDF")
plt.legend()
plt.title("Histogram of delays")
plt.xlabel("Time (s)")
plt.show()
print("delay spread", macro.ds)
print("lambda", lambda_tau)