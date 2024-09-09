#!/usr/bin/python
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as pg
from CASTRO5G import multipathChannel as mc

model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=True)
plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
los, PLfree, SF = plinfo
nClusters = clusters.shape[0]
nNLOSsp=subpaths.loc[1,:].shape[0]

pathAmplitudes = ( np.sqrt( subpaths.P )*np.exp(1j* subpaths.phase00) ).to_numpy()

plt.close('all')
fig_ctr=0

#3D polar plots of AoA
Nant = 8

# compute the response of the antenna array with Nant antennas
AntennaResponsesTx =mc.fULA(subpaths.AoD.to_numpy() *np.pi/180,Nant)
AntennaResponsesRx =mc.fULA(subpaths.AoA.to_numpy() *np.pi/180,Nant)

Npointsplot=101
# compute the "beamforming vector". This vector is multiplied by the "response" when we want to receive from the desired angle
angles_plot = np.linspace(0,2*np.pi,Npointsplot)
BeamformingVectors =mc.fULA(angles_plot,Nant)

arrayGainAllPathsRx=BeamformingVectors.conj()@AntennaResponsesRx.T
arrayGainAllPathsTx=BeamformingVectors.conj()@AntennaResponsesTx.T

channelResponseCombined =  np.sum( arrayGainAllPathsTx[None,:,:]*arrayGainAllPathsRx[:,None,:]*pathAmplitudes , axis=2) 

channelResponseCombineddB = np.maximum(10*np.log10(Nant*Nant*np.abs(channelResponseCombined)**2),-30)

fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = fig.add_subplot(111, projection='3d')
angTx, angRx = np.meshgrid(angles_plot, angles_plot)
surf = ax.plot_surface(angTx, angRx, channelResponseCombineddB,rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)

plt.colorbar(surf,shrink=0.8, label = 'Analog Beam Gain dB')
plt.xticks(ticks=np.pi*np.array([0,.5,1,1.5,2]),labels=['0','$\\frac{\\pi}{2}$','$\\pi$','$\\frac{3\\pi}{2}$','$2\\pi$'])
plt.yticks(ticks=np.pi*np.array([0,.5,1,1.5,2]),labels=['0','$\\frac{\\pi}{2}$','$\\pi$','$\\frac{3\\pi}{2}$','$2\\pi$'])
plt.xlabel('AoD')
plt.ylabel('AoA')
plt.title('%d ULA MIMO directivity for sum of all paths'%(Nant))

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.pcolor(angTx, angRx, channelResponseCombineddB, cmap=cm.coolwarm)
plt.colorbar(label = 'Analog Beam Gain dB')
plt.xticks(ticks=np.pi*np.array([0,.5,1,1.5,2]),labels=['0','$\\frac{\\pi}{2}$','$\\pi$','$\\frac{3\\pi}{2}$','$2\\pi$'])
plt.yticks(ticks=np.pi*np.array([0,.5,1,1.5,2]),labels=['0','$\\frac{\\pi}{2}$','$\\pi$','$\\frac{3\\pi}{2}$','$2\\pi$'])
plt.xlabel('AoD')
plt.ylabel('AoA')
plt.title('%d UPA MIMO directivity for sum of all paths'%(Nant))
maxdB=np.max(10*np.log10(Nant*Nant*np.abs(channelResponseCombined)**2))
mindB=-30
for n in range(nClusters):   
    Nsp=subpaths.loc[n,:].shape[0]
    plt.scatter(np.mod(subpaths.loc[n,:].AoD*np.pi/180,2*np.pi),np.mod(subpaths.loc[n,:].AoA*np.pi/180,2*np.pi),
                # color=cm.coolwarm((10*np.log10(Nant*Nant*subpaths.loc[n,:].P)-mindB)/((maxdB-mindB))),
                color=cm.jet(n/(nClusters-1)),
                marker='o')
