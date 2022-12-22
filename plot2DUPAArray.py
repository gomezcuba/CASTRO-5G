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

#2D polar plots of AoA
AoAs = np.array([x.azimutOfArrival[0] for x in chparams.channelPaths])
ZoAs = np.array([np.pi/2-x.zenithOfArrival[0] for x in chparams.channelPaths])
Npath=np.size(AoAs)

# compute the response of the antenna array with Nant antennas
Nant = 8

AAntennaResponses =mc.fULA(AoAs,Nant)
ZAntennaResponses =mc.fULA(np.pi/2-ZoAs,Nant)#zenit goes from vertical down
#the full Uniform Planar Array response is the following but we do not need to calculate it for this plot
#AntennaResponses=np.zeros((Npath,Nant*Nant,1))
#for i in range(Npath):
#   AntennaResponses[i,:,:]=np.kron(AAntennaResponses[i,:,:], AAntennaResponses[i,:,:])
pathAmplitudes = np.array([x.complexAmplitude[0] for x in chparams.channelPaths])

Npointsplot=401
# compute the "beamforming vector". This vector is multiplied by the "response" when we want to receive from the desired angle
azimut_plot = np.linspace(0,2*np.pi,Npointsplot)
ABeamformingVectors =mc.fULA(azimut_plot,Nant)
zenit_plot = np.linspace(0,np.pi,Npointsplot)
ZBeamformingVectors =mc.fULA(np.pi/2-zenit_plot,Nant)

AarrayGainAllPathsRx=(AAntennaResponses.transpose([0,2,1]).conj()@ABeamformingVectors[:,None,:,:]).reshape((Npointsplot,Npath))
ZarrayGainAllPathsRx=(ZAntennaResponses.transpose([0,2,1]).conj()@ZBeamformingVectors[:,None,:,:]).reshape((Npointsplot,Npath))

ZarrayGainAllPathsRx = ZarrayGainAllPathsRx.reshape(Npointsplot,1,Npath)#move the rx zenith Angle to a new axis

# plot of one single path without power effects

arrayResponseOnePath =  ZarrayGainAllPathsRx[:,:,0]*AarrayGainAllPathsRx[None,:,0]
arrayResponseOnePathdBtrunc30 = np.maximum(10*np.log10(Nant*Nant*np.abs(arrayResponseOnePath)**2),-30)

fig_ctr+=1
fig = plt.figure(fig_ctr)
Aang,Zang = np.meshgrid(azimut_plot,zenit_plot)
plt.pcolor(Aang,Zang, arrayResponseOnePathdBtrunc30, cmap=cm.coolwarm)
plt.colorbar(label = 'Directive Array Channel Gain dB')
plt.xticks(ticks=np.pi*np.array([0,.5,1,1.5,2]),labels=['0','$\\frac{\\pi}{2}$','$\pi$','$\\frac{3\\pi}{2}$','$\\pi$'])
plt.yticks(ticks=np.pi*np.array([0,.25,.5,.75,1]),labels=['0','$\\frac{\\pi}{4}$','$\\frac{\\pi}{2}$','$\\frac{3\\pi}{4}$','$\pi$',])
plt.xlabel('AoA')
plt.ylabel('ZoA')


fig_ctr+=1
fig = plt.figure(fig_ctr)
sphereX = np.cos(Aang)*np.cos(np.pi/2-Zang)
sphereY = np.sin(Aang)*np.cos(np.pi/2-Zang)
sphereZ = np.sin(np.pi/2-Zang)
ax = Axes3D(fig)
manual_colors = (arrayResponseOnePathdBtrunc30 -np.min(arrayResponseOnePathdBtrunc30) )/(np.max(arrayResponseOnePathdBtrunc30)-np.min(arrayResponseOnePathdBtrunc30))
surf=ax.plot_surface(sphereX, sphereY, sphereZ, facecolors=cm.coolwarm(manual_colors), linewidth=0, antialiased=False)
ax.set_xlabel('cos(AoA)')
ax.set_ylabel('sin(AoA)')
ax.set_zlabel('cos(ZoA)')


fig_ctr+=1
fig = plt.figure(fig_ctr)
radius=np.maximum(arrayResponseOnePathdBtrunc30+25,0)
sphereX = radius*np.cos(Aang)*np.cos(np.pi/2-Zang)
sphereY = radius*np.sin(Aang)*np.cos(np.pi/2-Zang)
sphereZ = radius*np.sin(np.pi/2-Zang)
ax = Axes3D(fig)
manual_colors = (arrayResponseOnePathdBtrunc30 -np.min(arrayResponseOnePathdBtrunc30) )/(np.max(arrayResponseOnePathdBtrunc30)-np.min(arrayResponseOnePathdBtrunc30))
surf=ax.plot_surface(sphereX, sphereY, sphereZ, facecolors=cm.coolwarm(manual_colors), linewidth=0, antialiased=False)
ax.set_xlabel('cos(AoA)')
ax.set_ylabel('sin(AoA)')
ax.set_zlabel('cos(ZoA)')


 #plot of all the paths in the channel


channelResponseCombined =  np.sum( ZarrayGainAllPathsRx*AarrayGainAllPathsRx*pathAmplitudes , axis=2) 
channelResponseCombineddB = np.maximum(10*np.log10(Nant*Nant*np.abs(channelResponseCombined)**2),-30)

fig_ctr+=1
fig = plt.figure(fig_ctr)
Aang,Zang = np.meshgrid(azimut_plot,zenit_plot)
plt.pcolor(Aang,Zang, channelResponseCombineddB, cmap=cm.coolwarm)
plt.colorbar(label = 'Directive Array Channel Gain dB')
plt.xticks(ticks=np.pi*np.array([0,.5,1,1.5,2]),labels=['0','$\\frac{\\pi}{2}$','$\pi$','$\\frac{3\\pi}{2}$','$\\pi$'])
plt.yticks(ticks=np.pi*np.array([0,.25,.5,.75,1]),labels=['0','$\\frac{\\pi}{4}$','$\\frac{\\pi}{2}$','$\\frac{3\\pi}{4}$','$\pi$',])
plt.xlabel('AoA')
plt.ylabel('ZoA')


fig_ctr+=1
fig = plt.figure(fig_ctr)
sphereX = np.cos(Aang)*np.cos(np.pi/2-Zang)
sphereY = np.sin(Aang)*np.cos(np.pi/2-Zang)
sphereZ = np.sin(np.pi/2-Zang)
ax = Axes3D(fig)
manual_colors = (channelResponseCombineddB -np.min(channelResponseCombineddB) )/(np.max(channelResponseCombineddB)-np.min(channelResponseCombineddB))
surf=ax.plot_surface(sphereX, sphereY, sphereZ, facecolors=cm.coolwarm(manual_colors), linewidth=0, antialiased=False)
ax.set_xlabel('cos(AoA)')
ax.set_ylabel('sin(AoA)')
ax.set_zlabel('cos(ZoA)')


fig_ctr+=1
fig = plt.figure(fig_ctr)
radius=np.maximum(channelResponseCombineddB+25,0)
sphereX = radius*np.cos(Aang)*np.cos(np.pi/2-Zang)
sphereY = radius*np.sin(Aang)*np.cos(np.pi/2-Zang)
sphereZ = radius*np.sin(np.pi/2-Zang)
ax = Axes3D(fig)
manual_colors = (channelResponseCombineddB -np.min(channelResponseCombineddB) )/(np.max(channelResponseCombineddB)-np.min(channelResponseCombineddB))
surf=ax.plot_surface(sphereX, sphereY, sphereZ, facecolors=cm.coolwarm(manual_colors), linewidth=0, antialiased=False)
ax.set_xlabel('cos(AoA)')
ax.set_ylabel('sin(AoA)')
ax.set_zlabel('cos(ZoA)')