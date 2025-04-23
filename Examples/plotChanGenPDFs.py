# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:48:43 2022

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['text.usetex'] = True

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as pg

plt.close('all')
model = pg.ThreeGPPMultipathChannelModel(scenario="UMi")
model.bLargeBandwidthOption=True

txPos = np.array((0,0,25))
rxPos = (np.random.uniform(0, 100),np.random.uniform(0, 100),1.5)
vLOS = rxPos-txPos
d2D = np.linalg.norm(vLOS[0:2])
d3D = np.linalg.norm(vLOS)

losAoD=np.mod( np.arctan( vLOS[1] / vLOS[0] )+np.pi*(vLOS[0]<0), 2*np.pi )
losAoA=np.mod(np.pi+losAoD, 2*np.pi ) # revise
losZoD=np.pi/2-np.arctan( vLOS[2] / d2D )
losZoA=np.pi-losZoD # revise

#3GPP model is in degrees but numpy uses radians
losAoD=(180.0/np.pi)*losAoD #angle of departure 
losZoD=(180.0/np.pi)*losZoD 
losAoA=(180.0/np.pi)*losAoA #angle of arrival
losZoA=(180.0/np.pi)*losZoA
angles = (losAoD,losAoA,losZoD,losZoA)

hut=rxPos[2]
los=False
macro = model.create_macro((txPos[0],txPos[1],rxPos[0],rxPos[1],los))
sfdB,ds,asa,asd,zsa,zsd_lslog,K =macro            
zsd_mu = model.scenarioParams.NLOS.funZSD_mu(d2D,hut)#unlike other statistics, ZSD changes with hut and d2D             
zsd = min( np.power(10.0,zsd_mu + zsd_lslog ), 52.0)
zod_offset_mu = model.scenarioParams.NLOS.funZoDoffset(d2D,hut)    
czsd = (3/8)*(10**zsd_mu)#intra-cluster ZSD    
smallStatistics = (los,ds,asa,asd,zsa,zsd,K,czsd,zod_offset_mu)        

Nchannels = 1000
Ncluster = model.scenarioParams.NLOS.N
Cphi=model.CphiNLOStable[Ncluster]
Cteta=model.CtetaNLOStable[Ncluster]
listaTau = [] #sometimes fewer clusters are generated
listaAoD = [] #sometimes fewer clusters are generated
listaAoA = [] #sometimes fewer clusters are generated
listaZoD = [] #sometimes fewer clusters are generated
listaZoA = [] #sometimes fewer clusters are generated

for i in range(Nchannels):
    clusters, subpaths =model.create_small_param(angles,smallStatistics)
    tau,powC,AOA,AOD,ZOA,ZOD =clusters.T.to_numpy()
    listaTau.append(tau[1:])#first cluster always has delay 0
    
    AODprima = 2*(asd/1.4)*np.sqrt(-np.log(powC/np.max(powC)))/Cphi
    listaAoD.append(np.mod( AOD - losAoD - AODprima*(2*(AOD>losAoD)-1) +180 ,360) -180 )#in NLOS case the first cluster is not centered in LOS angle    
    AOAprima = 2*(asa/1.4)*np.sqrt(-np.log(powC/np.max(powC)))/Cphi
    listaAoA.append(np.mod( AOA - losAoA - AOAprima*(2*(AOA>losAoA)-1) +180 ,360) -180 )#in NLOS case the first cluster is not centered in LOS angle    
    ZODprima = -((zsd*np.log(powC/np.max(powC)))/Cteta)
    indepZOD = ZOD - zod_offset_mu - losZoD
    listaZoD.append( indepZOD-ZODprima*np.sign(indepZOD) )#in NLOS case the first cluster is not centered in LOS angle    
    ZOAprima = -((zsa*np.log(powC/np.max(powC)))/Cteta)
    listaZoA.append(np.mod( ZOA - losZoA - ZOAprima*(2*(ZOA>losZoA)-1) +180 ,360) -180 )#in NLOS case the first cluster is not centered in LOS angle    
plt.figure(1)
tau_all = np.concatenate(listaTau)
plt.hist(tau_all, bins=20, density=True,label='Histogram')
aux_x = np.linspace(0,np.max(tau_all),101)
lambda_tau = 1/(macro.ds*model.scenarioParams.NLOS.rt)
plt.plot(aux_x,lambda_tau*np.exp(-aux_x*lambda_tau),'r:', label = "PDF")
plt.legend()
plt.title("Histogram of cluster delays vs p.d.f.")
plt.xlabel("Time (s)")
plt.show()
print("delay spread", macro.ds)
print("lambda", lambda_tau)

plt.figure(2)
aod_all=np.concatenate(listaAoD)
plt.hist(aod_all, bins=20, density=True,label='Histogram')
aux_x = np.linspace(np.min(aod_all),np.max(aod_all),101)
pdfAoD = np.exp(-(aux_x / (np.sqrt(2)*(asd/7)) ) ** 2)/(np.sqrt(2*np.pi)*asd/7)
plt.plot(aux_x,pdfAoD,'r:', label = "PDF")
plt.legend()
plt.title("Histogram of cluster AoD random part vs p.d.f.")
plt.xlabel("AoD - LOS-AoD - $\mu_{AoD}$(cluster) (º)")
plt.show()

plt.figure(3)
aoa_all=np.concatenate(listaAoA)
plt.hist(aoa_all, bins=20, density=True,label='Histogram')
aux_x = np.linspace(np.min(aoa_all),np.max(aoa_all),101)
pdfAoA = np.exp(-(aux_x / (np.sqrt(2)*(asa/7)) ) ** 2)/(np.sqrt(2*np.pi)*asa/7)
plt.plot(aux_x,pdfAoA,'r:', label = "PDF")
plt.legend()
plt.title("Histogram of cluster AoA random part vs p.d.f.")
plt.xlabel("AoA - LOS-AoA - $\mu_{AoA}$(cluster) (º)")
plt.show()

plt.figure(4)
zod_all=np.concatenate(listaZoD)
plt.hist(zod_all, bins=20, density=True,label='Histogram')
aux_x = np.linspace(np.min(zod_all),np.max(zod_all),101)
pdfZoD = np.exp(-(aux_x / (np.sqrt(2)*(zsd/7)) ) ** 2)/(np.sqrt(2*np.pi)*zsd/7)
plt.plot(aux_x,pdfZoD,'r:', label = "PDF")
plt.legend()
plt.title("Histogram of cluster ZoD random part vs p.d.f.")
plt.xlabel("ZoD - LOS-ZoD - $\mu_{ZoD}$(cluster) (º)")
plt.show()


plt.figure(5)
zoa_all=np.concatenate(listaZoA)
plt.hist(zoa_all, bins=20, density=True,label='Histogram')
aux_x = np.linspace(np.min(zoa_all),np.max(zoa_all),101)
pdfZoA = np.exp(-(aux_x / (np.sqrt(2)*(zsa/7)) ) ** 2)/(np.sqrt(2*np.pi)*zsa/7)
plt.plot(aux_x,pdfZoA,'r:', label = "PDF")
plt.legend()
plt.title("Histogram of cluster ZoA random part vs p.d.f.")
plt.xlabel("ZoA - LOS-ZoA - $\mu_{ZoA}$(cluster) (º)")
plt.show()
