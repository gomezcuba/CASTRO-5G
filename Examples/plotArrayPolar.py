#!/usr/bin/python
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as pg
from CASTRO5G import multipathChannel as mc

model = pg.ThreeGPPMultipathChannelModel(scenario="UMi",bLargeBandwidthOption=False)
plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
nClusters = clusters.shape[0]
nNLOSsp=subpaths.loc[1,:].shape[0]
plt.close('all')
fig_ctr=0

#2D polar plots of AoA

#plot of rx AoAs and channel gains

fig_ctr+=1
fig = plt.figure(fig_ctr)
for n in range(nClusters):   
    pathAmplitudesdBtrunc25_sp = np.maximum(10*np.log10( subpaths.loc[n,:].P ),-45)
    Nsp=subpaths.loc[n,:].shape[0]
    plt.polar(np.tile(subpaths.loc[n,:].AoA*np.pi/180,(2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_sp]),':',color=cm.jet(n/(nClusters-1)))
    plt.scatter(subpaths.loc[n,:].AoA*np.pi/180,pathAmplitudesdBtrunc25_sp,color=cm.jet(n/(nClusters-1)),marker=(3,0,-90+clusters.loc[n].AoA))
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'])
plt.title("AoAs and normalized Power for all clusters and subpaths")

# compute the response of the antenna array with Nant antennas
Nant = 16
Npointsplot=1001
# compute the "beamforming vector". This vector is multiplied by the "response" when we want to receive from the desired angle
angles_plot = np.linspace(0,2*np.pi,Npointsplot)
BeamformingVectors =mc.fULA(angles_plot,Nant)

#plot of receive array and chanel gain of top N paths
fig_ctr+=1
fig = plt.figure(fig_ctr)
# plt.subplot(2,1,1, projection='polar')
Ntop = 3
topNpaths = subpaths.sort_values(by=['P'],ascending=False).index[0:Ntop]
for p in range(Ntop):
    n,m = topNpaths[p]
    
    AntennaResponse1Path =mc.fULA(subpaths.loc[n,m].AoA *np.pi/180 ,Nant)
    
    arrayGain1Path=(BeamformingVectors.conj()@ AntennaResponse1Path )
    arrayGain1PathdBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(arrayGain1Path)**2),-25)

    plt.polar(angles_plot,arrayGain1PathdBtrunc25,':',label="ULA G_{%d,%d}"%(n,m),color=cm.jet(p/(Ntop-1)))
    pathAmplitudesdBtrunc25_sp = np.maximum(10*np.log10( subpaths.loc[n,m].P ),-45)
    plt.polar(np.tile(subpaths.loc[n,m].AoA *np.pi/180,(2,1)),[-40,pathAmplitudesdBtrunc25_sp],'-',color=cm.jet(p/(Ntop-1)))
    plt.scatter(subpaths.loc[n,m].AoA*np.pi/180,pathAmplitudesdBtrunc25_sp,color=cm.jet(p/(Ntop-1)),marker=(3,0,-90+subpaths.loc[n,m].AoA))

plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])
plt.legend()
plt.title("%d ULA angular response and channel gain for %d strongest subpaths"%(Nant,Ntop))

#plot of receive array response of top N paths COMBINEDfig_ctr+=1
fig_ctr+=1
fig = plt.figure(fig_ctr)
# plt.subplot(2,1,2, projection='polar')
AntennaResponseNPaths = mc.fULA(subpaths.loc[topNpaths].AoA.to_numpy() *np.pi/180 ,Nant)
arrayGainNPaths=BeamformingVectors.conj() @ AntennaResponseNPaths.T
chanCoefNPaths=np.sqrt( subpaths.loc[topNpaths].P ) * np.exp( 1j* subpaths.loc[topNpaths].phase00 )
arrayResponseCombined = np.sum( arrayGainNPaths*chanCoefNPaths.to_numpy() , axis=1)
arrayResCondBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(arrayResponseCombined)**2),-25)

plt.polar(angles_plot,arrayResCondBtrunc25)
plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])
plt.title("%d ULA angular response times channel gain summed over %d strongest subpaths"%(Nant,Ntop))

#plot of receive array response of ALL paths in SEPARATE LINES, WITH the effect of power
fig_ctr+=1
fig = plt.figure(fig_ctr)
for n in range(nClusters):       
    AntennaResponseCluster =mc.fULA(subpaths.loc[n,:].AoA.to_numpy() *np.pi/180 ,Nant)
    arrayGainCluster=BeamformingVectors.conj() @ AntennaResponseCluster.T
    chanGainClusterdBtrunc25 = np.maximum(10*np.log10(subpaths.loc[n,:].P.to_numpy()*Nant*np.abs(arrayGainCluster)**2),-25)
    plt.polar(angles_plot,chanGainClusterdBtrunc25,color=cm.jet(n/(nClusters-1)),label='Cluster %d'%(n))
plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])
plt.title("%d ULA angular response times channel gain for all subpaths"%(Nant))


#plot of receive array response of ALL paths COMBINED
fig_ctr+=1
fig = plt.figure(fig_ctr)

AntennaResponseAll =mc.fULA( subpaths.AoA.to_numpy() *np.pi/180 ,Nant)
arrayGainAllPaths= BeamformingVectors.conj() @ AntennaResponseAll.T
chanCoefAllPaths=np.sqrt( subpaths.P )*np.exp(-1j* subpaths.phase00)
arrayResponseCombined = np.sum( arrayGainAllPaths*chanCoefAllPaths.to_numpy() , axis=1)

arrayResCondBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(arrayResponseCombined)**2),-25)

plt.polar(angles_plot,arrayResCondBtrunc25)
plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])
plt.title("%d ULA angular response times channel gain summed for all subpaths"%(Nant))
plt.savefig("../Figures/radiationAoA.eps")


