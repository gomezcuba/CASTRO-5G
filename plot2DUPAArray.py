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
tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths.T.to_numpy()

#4D  color intensity plots vs delay, AoA and AoD grid
AoAs = AOA_sp*np.pi/180#radians
ZoAs = ZOA_sp*np.pi/180#radians
delays = tau_sp*1e9#nanoseconds
Npath=np.size(delays)
pathAmplitudes = np.sqrt( pow_sp )*np.exp(2j*np.pi*np.random.rand(Npath))

plt.close('all')
fig_ctr=0

# compute the response of the antenna array with Nant antennas
Nant = 16

AAntennaResponses =mc.fULA(AoAs,Nant)
ZAntennaResponses =mc.fULA(np.pi/2-ZoAs,Nant)#zenit goes from vertical down
#the full Uniform Planar Array response is the following but we do not need to calculate it for this plot
#AntennaResponses=np.zeros((Npath,Nant*Nant,1))
#for i in range(Npath):
#   AntennaResponses[i,:,:]=np.kron(AAntennaResponses[i,:,:], AAntennaResponses[i,:,:])

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
plt.pcolor(Aang,Zang, arrayResponseOnePathdBtrunc30, cmap=cm.jet)
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
surf=ax.plot_surface(sphereX, sphereY, sphereZ, facecolors=cm.jet(manual_colors), linewidth=0, antialiased=False)
ax.set_xlabel('cos(AoA)')
ax.set_ylabel('sin(AoA)')
ax.set_zlabel('cos(ZoA)')
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cm.jet),shrink=0.8,label = 'Directive Array Channel Gain dB')
cbar.set_ticks((np.arange(-30,30,10) -np.min(arrayResponseOnePathdBtrunc30) )/(np.max(arrayResponseOnePathdBtrunc30)-np.min(arrayResponseOnePathdBtrunc30)))
cbar.set_ticklabels(['%.0f dB'%x for x in np.arange(-30,30,10)])

fig_ctr+=1
fig = plt.figure(fig_ctr)
radius=np.maximum(arrayResponseOnePathdBtrunc30+25,0)
sphereX = radius*np.cos(Aang)*np.cos(np.pi/2-Zang)
sphereY = radius*np.sin(Aang)*np.cos(np.pi/2-Zang)
sphereZ = radius*np.sin(np.pi/2-Zang)
ax = Axes3D(fig)
manual_colors = (arrayResponseOnePathdBtrunc30 -np.min(arrayResponseOnePathdBtrunc30) )/(np.max(arrayResponseOnePathdBtrunc30)-np.min(arrayResponseOnePathdBtrunc30))
surf=ax.plot_surface(sphereX, sphereY, sphereZ, facecolors=cm.jet(manual_colors), linewidth=0, antialiased=False)
ax.set_xlabel('cos(AoA)')
ax.set_ylabel('sin(AoA)')
ax.set_zlabel('cos(ZoA)')
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cm.jet),shrink=0.8,label = 'Directive Array Channel Gain dB')
cbar.set_ticks((np.arange(-30,30,10) -np.min(arrayResponseOnePathdBtrunc30) )/(np.max(arrayResponseOnePathdBtrunc30)-np.min(arrayResponseOnePathdBtrunc30)))
cbar.set_ticklabels(['%.0f dB'%x for x in np.arange(-30,30,10)])


 #plot of all the paths in the channel


channelResponseCombined =  np.sum( ZarrayGainAllPathsRx*AarrayGainAllPathsRx*pathAmplitudes , axis=2) 
channelResponseCombineddB = np.maximum(10*np.log10(Nant*Nant*np.abs(channelResponseCombined)**2),-30)

fig_ctr+=1
fig = plt.figure(fig_ctr)
Aang,Zang = np.meshgrid(azimut_plot,zenit_plot)
plt.pcolor(Aang,Zang, channelResponseCombineddB, cmap=cm.jet)
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
surf=ax.plot_surface(sphereX, sphereY, sphereZ, facecolors=cm.jet(manual_colors), linewidth=0, antialiased=False)
ax.set_xlabel('cos(AoA)')
ax.set_ylabel('sin(AoA)')
ax.set_zlabel('cos(ZoA)')
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cm.jet),shrink=0.8,label = 'Directive Array Channel Gain dB')
cbar.set_ticks((np.arange(-30,30,10) -np.min(arrayResponseOnePathdBtrunc30) )/(np.max(arrayResponseOnePathdBtrunc30)-np.min(arrayResponseOnePathdBtrunc30)))
cbar.set_ticklabels(['%.0f dB'%x for x in np.arange(-30,30,10)])


fig_ctr+=1
fig = plt.figure(fig_ctr)
radius=np.maximum(channelResponseCombineddB+25,0)
sphereX = radius*np.cos(Aang)*np.cos(np.pi/2-Zang)
sphereY = radius*np.sin(Aang)*np.cos(np.pi/2-Zang)
sphereZ = radius*np.sin(np.pi/2-Zang)
ax = Axes3D(fig)
manual_colors = (channelResponseCombineddB -np.min(channelResponseCombineddB) )/(np.max(channelResponseCombineddB)-np.min(channelResponseCombineddB))
surf=ax.plot_surface(sphereX, sphereY, sphereZ, facecolors=cm.jet(manual_colors), linewidth=0, antialiased=False)
ax.set_xlabel('cos(AoA)')
ax.set_ylabel('sin(AoA)')
ax.set_zlabel('cos(ZoA)')
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cm.jet),shrink=0.8,label = 'Directive Array Channel Gain dB')
cbar.set_ticks((np.arange(-30,30,10) -np.min(arrayResponseOnePathdBtrunc30) )/(np.max(arrayResponseOnePathdBtrunc30)-np.min(arrayResponseOnePathdBtrunc30)))
cbar.set_ticklabels(['%.0f dB'%x for x in np.arange(-30,30,10)])
