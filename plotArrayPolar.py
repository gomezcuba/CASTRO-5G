#!/usr/bin/python
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
# equivalent to rcParams['animation.html'] = 'html5'
rc('animation', html='html5')
from matplotlib import cm
import os

import threeGPPMultipathGenerator as pg
import multipathChannel as mc

model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=True)
plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
tau,powC,AOA,AOD,ZOA,ZOD = clusters.T.to_numpy()
los, PLfree, SF = plinfo
nClusters = tau.size
if los:
    M=max(subpaths.loc[0,:].index)
    (tau_los,pow_los,losAoA,losAoD,losZoA,losZoD,losXPR,losphase00,losphase01,losphase10,losphase11)=subpaths.loc[(0,M),:]
    tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp,XPR_sp,phase00,phase01,phase10,phase11 =  subpaths.drop((0,M)).T.to_numpy()
else:    
    M=max(subpaths.loc[0,:].index)+1
    tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp,XPR_sp,phase00,phase01,phase10,phase11 =  subpaths.T.to_numpy()
tau_sp=tau_sp.reshape(nClusters,-1)
pow_sp=pow_sp.reshape(nClusters,-1)
AOA_sp=AOA_sp.reshape(nClusters,-1)
AOD_sp=AOD_sp.reshape(nClusters,-1)
ZOA_sp=ZOA_sp.reshape(nClusters,-1)
ZOD_sp=ZOD_sp.reshape(nClusters,-1)
phase00=phase00.reshape(nClusters,-1)
plt.close('all')
fig_ctr=0

#2D polar plots of AoA

#plot of rx AoAs and channel gains
fig_ctr+=1
fig = plt.figure(fig_ctr)

Nsp=AOA_sp.shape[1]

if los:
    plt.polar(losAoA*np.pi/180*np.ones((2,1)),np.vstack([[-40],10*np.log10(pow_los)]),':',color=cm.jet(0))
    plt.scatter(losAoA*np.pi/180*np.ones((2,1)),np.vstack([[-40],10*np.log10(pow_los)]),color=cm.jet(0),marker='<')
for n in range(nClusters):   
    pathAmplitudesdBtrunc25_sp = np.maximum(10*np.log10( pow_sp[n,:] ),-45)
    plt.polar(AOA_sp[n,:]*np.pi/180*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_sp]),':',color=cm.jet(n/(nClusters-1)) )
    plt.scatter(AOA_sp[n,:]*np.pi/180,pathAmplitudesdBtrunc25_sp,color=cm.jet(n/(nClusters-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'])

# compute the response of the antenna array with Nant antennas
Nant = 16
Npointsplot=1001
# compute the "beamforming vector". This vector is multiplied by the "response" when we want to receive from the desired angle
angles_plot = np.linspace(0,2*np.pi,Npointsplot)
BeamformingVectors =mc.fULA(angles_plot,Nant)

AntennaResponse1Path =mc.fULA(AOA_sp[0,0] *np.pi/180 ,Nant)

#plot of receive array response of first path
fig_ctr+=1
fig = plt.figure(fig_ctr)
arrayGain1Path=(BeamformingVectors.transpose([0,2,1]).conj()@ AntennaResponse1Path ).reshape(-1)
arrayGain1PathdBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(arrayGain1Path)**2),-25)

plt.polar(angles_plot,arrayGain1PathdBtrunc25)
plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])


#plot of receive array response of ALL paths in SEPARATE LINES, WITHOUT the effect of power
fig_ctr+=1
fig = plt.figure(fig_ctr)
if los:
    AntennaResponseLOS =mc.fULA(np.array(losAoA) *np.pi/180 ,Nant)
    arrayGainLOS = (AntennaResponseLOS.T.conj()@BeamformingVectors).reshape((Npointsplot))
    arrayGainLOSdBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(arrayGainLOS)**2),-25)
    plt.polar(angles_plot,arrayGainLOSdBtrunc25,color=cm.jet(0))
for n in range(nClusters):       
    AntennaResponseCluster =mc.fULA(AOA_sp[n,:] *np.pi/180 ,Nant)
    arrayGainCluster=(AntennaResponseCluster.transpose([0,2,1]).conj()@BeamformingVectors[:,None,:,:]).reshape((Npointsplot,Nsp))
    arrayGainClusterdBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(arrayGainCluster)**2),-25)
    plt.polar(angles_plot,arrayGainClusterdBtrunc25,color=cm.jet(n/(nClusters-1)))
plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])


#plot of receive array response of ALL paths in SEPARATE LINES, WITH the effect of power
fig_ctr+=1
fig = plt.figure(fig_ctr)
if los:
    AntennaResponseLOS =mc.fULA(np.array(losAoA) *np.pi/180 ,Nant)
    arrayGainLOS = (AntennaResponseLOS.T.conj()@BeamformingVectors).reshape((Npointsplot))
    chanGainLOSdBtrunc25 = np.maximum(10*np.log10(pow_los*Nant*np.abs(arrayGainLOS)**2),-25)
    plt.polar(angles_plot,chanGainLOSdBtrunc25,color=cm.jet(0))
for n in range(nClusters):       
    AntennaResponseCluster =mc.fULA(AOA_sp[n,:] *np.pi/180 ,Nant)
    arrayGainCluster=(AntennaResponseCluster.transpose([0,2,1]).conj()@BeamformingVectors[:,None,:,:]).reshape((Npointsplot,Nsp))
    chanGainClusterdBtrunc25 = np.maximum(10*np.log10(pow_sp[n,:]*Nant*np.abs(arrayGainCluster)**2),-25)
    plt.polar(angles_plot,chanGainClusterdBtrunc25,color=cm.jet(n/(nClusters-1)))
plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])


#plot of receive array response of ALL paths COMBINED
fig_ctr+=1
fig = plt.figure(fig_ctr)

AoAs = AOA_sp.reshape(-1)*np.pi/180#radians
Npath=np.size(AoAs)
AntennaResponseAll =mc.fULA( AoAs ,Nant)
arrayGainAllPaths=(AntennaResponseAll.transpose([0,2,1]).conj()@BeamformingVectors[:,None,:,:]).reshape((Npointsplot,Npath))
chanCoef_sp=np.sqrt(pow_sp)*np.exp(-1j*phase00)
arrayResponseCombined = np.sum( arrayGainAllPaths*chanCoef_sp.reshape(-1) , axis=1)
if los:
    AntennaResponseLOS =mc.fULA(np.array(losAoA) *np.pi/180 ,Nant)
    arrayGainLOS = (AntennaResponseLOS.T.conj()@BeamformingVectors).reshape((Npointsplot))
    arrayResponseCombined = np.sqrt(pow_los)*arrayGainLOS + arrayResponseCombined

arrayResCondBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(arrayResponseCombined)**2),-25)

plt.polar(angles_plot,arrayResCondBtrunc25)
plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])

