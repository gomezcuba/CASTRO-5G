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

model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=False)
plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
tau,powC,AOA,AOD,ZOA,ZOD = clusters.T.to_numpy()
los, PLfree, SF = plinfo
nClusters = tau.size
if los:
    M=max(subpaths.loc[0,:].index)
    (tau_los,pow_los,losAoA,losAoD,losZoA,losZoD)=subpaths.loc[(0,M),:]
    tau_sp,powC_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths.drop((0,M)).T.to_numpy()
else:    
    M=max(subpaths.loc[0,:].index)+1
    tau_sp,powC_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths.T.to_numpy()
tau_sp=tau_sp.reshape(nClusters,-1)
powC_sp=powC_sp.reshape(nClusters,-1)
AOA_sp=AOA_sp.reshape(nClusters,-1)
AOD_sp=AOD_sp.reshape(nClusters,-1)
ZOA_sp=ZOA_sp.reshape(nClusters,-1)
ZOD_sp=ZOD_sp.reshape(nClusters,-1)
plt.close('all')
fig_ctr=0

#2D polar plots of AoA
AoAs = AOA_sp.reshape(-1)*np.pi/180#radians
Npath=np.size(AoAs)
pathAmplitudes = np.sqrt( powC_sp.reshape(-1) )*np.exp(2j*np.pi*np.random.rand(Npath))

#plot of rx AoAs and channel gains
fig_ctr+=1
fig = plt.figure(fig_ctr)
pathAmplitudesdBtrunc25 = np.maximum(10*np.log10(np.abs(pathAmplitudes)**2),-45)

# plt.polar(AoAs*np.ones((2,1)),np.vstack([-40*np.ones((1,Npath)),pathAmplitudesdBtrunc25]),':')
# plt.scatter(AoAs,pathAmplitudesdBtrunc25,color='k',marker='x')
Nsp=AOA_sp.shape[1]

if los:
    plt.polar(losAoA*np.pi/180*np.ones((2,1)),np.vstack([[-40],10*np.log10(pow_los)]),':',color=cm.jet(0))
    plt.scatter(losAoA*np.pi/180*np.ones((2,1)),np.vstack([[-40],10*np.log10(pow_los)]),color=cm.jet(0),marker='<')
for n in range(nClusters):   
    pathAmplitudes_sp = np.sqrt( powC_sp[n,:] )*np.exp(2j*np.pi*np.random.rand(Nsp))
    pathAmplitudesdBtrunc25_sp = np.maximum(10*np.log10(np.abs(pathAmplitudes_sp)**2),-45)
    plt.polar(AOA_sp[n,:]*np.pi/180*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_sp]),':',color=cm.jet(n/(nClusters-1)) )
    plt.scatter(AOA_sp[n,:]*np.pi/180,pathAmplitudesdBtrunc25_sp,color=cm.jet(n/(nClusters-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'])

# compute the response of the antenna array with Nant antennas
Nant = 16
AntennaResponses =mc.fULA(AoAs,Nant)
Npointsplot=1001
# compute the "beamforming vector". This vector is multiplied by the "response" when we want to receive from the desired angle
angles_plot = np.linspace(0,2*np.pi,Npointsplot)
BeamformingVectors =mc.fULA(angles_plot,Nant)

# #plot of receive array response of first path
# fig_ctr+=1
# fig = plt.figure(fig_ctr)
# arrayGain1Path=(BeamformingVectors.transpose([0,2,1]).conj()@AntennaResponses[0,:,:]).reshape(-1)
# arrayGain1PathdBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(arrayGain1Path)**2),-25)

# plt.polar(angles_plot,arrayGain1PathdBtrunc25)
# plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])

# Nsp=AOA_sp.shape[1]


# #plot of receive array response of ALL paths in SEPARATE LINES, WITHOUT the effect of power
# fig_ctr+=1
# fig = plt.figure(fig_ctr)
# arrayGainAllPaths=(AntennaResponses.transpose([0,2,1]).conj()@BeamformingVectors[:,None,:,:]).reshape((Npointsplot,Npath))

# arrayGainAllPathsdBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(arrayGainAllPaths)**2),-25)

# plt.polar(angles_plot,arrayGainAllPathsdBtrunc25)
# plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])


# #plot of receive array response of ALL paths in SEPARATE LINES, WITH the effect of power
# fig_ctr+=1
# fig = plt.figure(fig_ctr)

# channelArrayGainAllPaths =  arrayGainAllPaths*pathAmplitudes 
# channelArrayGainAllPathsdBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(channelArrayGainAllPaths)**2),-25)

# plt.polar(angles_plot,channelArrayGainAllPathsdBtrunc25)
# plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])

# #plot of receive array response of 5 STRONGEST PATHS ONLY, in SEPARATE LINES, WITH the effect of power
# fig_ctr+=1
# fig = plt.figure(fig_ctr)
# Nbig = 5 # can also use something like np.sum(np.abs(pathAmplitudes)**2>1e-2) to calculate the number of paths greater than 0.01
# sortIndices = np.argsort(np.abs(pathAmplitudes)**2)

# channelArrayGainBigPaths =  arrayGainAllPaths[:,sortIndices[-Nbig:]]*pathAmplitudes [sortIndices[-Nbig:]]
# channelArrayGainBigPathsdBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(channelArrayGainBigPaths)**2),-35)

# plt.polar(angles_plot,channelArrayGainBigPathsdBtrunc25)
# plt.polar(AoAs[sortIndices[-Nbig:]]*np.ones((2,1)),np.vstack([-35*np.ones((1,Nbig)),10*np.log10(np.abs(pathAmplitudes [sortIndices[-Nbig:]])**2)]),':')
# plt.scatter(AoAs[sortIndices[-Nbig:]],10*np.log10(np.abs(pathAmplitudes [sortIndices[-Nbig:]])**2),color='k',marker='x')
# plt.yticks(ticks=[-30,-20,-10,0],labels=['-20dB','-30dB','-10dB','0dB'])


# #plot of receive array response of ALL paths COMBINED
# fig_ctr+=1
# fig = plt.figure(fig_ctr)

# arrayResponseCombined = np.sum( arrayGainAllPaths*pathAmplitudes , axis=1)
# arrayResCondBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(arrayResponseCombined)**2),-25)

# plt.polar(angles_plot,arrayResCondBtrunc25)
# plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])
plt.show()