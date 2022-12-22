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

model = pg.ThreeGPPMultipathChannelModel()
model.bLargeBandwidthOption=True
model.create_channel((0,0,10),(40,0,1.5))
chparams = model.dChansGenerated[(0,0,40,0)]

plt.close('all')
fig_ctr=0

#3D polar plots of AoA
Nant = 16
AoDs = np.array([x.azimutOfDeparture[0] for x in chparams.channelPaths])
AoAs = np.array([x.azimutOfArrival[0] for x in chparams.channelPaths])
pathAmplitudes = np.array([x.complexAmplitude[0] for x in chparams.channelPaths])
Npath=np.size(AoAs)

# compute the response of the antenna array with Nant antennas
Nant = 16
AntennaResponsesRx =mc.fULA(AoAs,Nant)
AntennaResponsesTx =mc.fULA(AoDs,Nant)
Npointsplot=101
# compute the "beamforming vector". This vector is multiplied by the "response" when we want to receive from the desired angle
angles_plot = np.linspace(0,2*np.pi,Npointsplot)
BeamformingVectors =mc.fULA(angles_plot,Nant)

arrayGainAllPathsRx=(AntennaResponsesRx.transpose([0,2,1]).conj()@BeamformingVectors[:,None,:,:]).reshape((Npointsplot,Npath))
arrayGainAllPathsTx=(AntennaResponsesTx.transpose([0,2,1]).conj()@BeamformingVectors[:,None,:,:]).reshape((Npointsplot,Npath))

arrayGainAllPathsTx = arrayGainAllPathsTx.reshape(Npointsplot,1,Npath)#move the txAngle to a new axis
channelResponseCombined =  np.sum( arrayGainAllPathsTx*arrayGainAllPathsRx*pathAmplitudes , axis=2) 

channelResponseCombineddB = np.maximum(10*np.log10(Nant*Nant*np.abs(channelResponseCombined)**2),-20)

fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = Axes3D(fig)
angTx, angRx = np.meshgrid(angles_plot, angles_plot)
surf = ax.plot_surface(angTx, angRx, channelResponseCombineddB, cmap=cm.coolwarm, linewidth=0, antialiased=False)

plt.colorbar(surf, label = 'Directive Array Channel Gain dB')
plt.xticks(ticks=np.pi*np.array([0,.5,1,1.5,2]),labels=['0','$\\frac{\\pi}{2}$','$\pi$','$\\frac{3\\pi}{2}$','$\\pi$'])
plt.yticks(ticks=np.pi*np.array([0,.5,1,1.5,2]),labels=['0','$\\frac{\\pi}{2}$','$\pi$','$\\frac{3\\pi}{2}$','$\\pi$'])
plt.xlabel('AoD')
plt.ylabel('AoA')

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.pcolor(angTx, angRx, channelResponseCombineddB, cmap=cm.coolwarm)
plt.colorbar(surf, label = 'Directive Array Channel Gain dB')
plt.xticks(ticks=np.pi*np.array([0,.5,1,1.5,2]),labels=['0','$\\frac{\\pi}{2}$','$\pi$','$\\frac{3\\pi}{2}$','$\\pi$'])
plt.yticks(ticks=np.pi*np.array([0,.5,1,1.5,2]),labels=['0','$\\frac{\\pi}{2}$','$\pi$','$\\frac{3\\pi}{2}$','$\\pi$'])
plt.xlabel('AoD')
plt.ylabel('AoA')