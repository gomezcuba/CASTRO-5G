#!/usr/bin/python
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

from CASTRO5G import threeGPPMultipathGenerator as pg
from CASTRO5G import multipathChannel as mc

fig_ctr=0

model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=True)
plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
los, PLfree, SF = plinfo
nClusters = clusters.shape[0]
nNLOSsp=subpaths.loc[1,:].shape[0]#cluster 0 may include the LOS path, 1 and onwards only nlos

#TODO - insert adapted AOA and compare response

#4D  color intensity plots vs delay, AoA and AoD grid

pathAmplitudes = ( np.sqrt( subpaths.P )*np.exp(1j* subpaths.phase00) ).to_numpy()

plt.close('all')

# compute the response of the antenna array with Nant antennas
Nant = 8

AAntennaResponses =mc.fULA( subpaths.AoA.to_numpy() *np.pi/180 ,Nant)
ZAntennaResponses =mc.fULA(np.pi/2-subpaths.ZoA.to_numpy() *np.pi/180 ,Nant)#zenit goes from vertical down
#the full Uniform Planar Array response is the following but we do not need to calculate it for this plot
#AntennaResponses=np.zeros((Npath,Nant*Nant,1))
#for i in range(Npath):
#   AntennaResponses[i,:,:]=np.kron(AAntennaResponses[i,:,:], AAntennaResponses[i,:,:])

Npointsplot=101
# compute the "beamforming vector". This vector is multiplied by the "response" when we want to receive from the desired angle
azimut_plot = np.linspace(0,2*np.pi,Npointsplot)
ABeamformingVectors =mc.fULA(azimut_plot,Nant)
zenit_plot = np.linspace(0,np.pi,Npointsplot)
ZBeamformingVectors =mc.fULA(np.pi/2-zenit_plot,Nant)

AarrayGainAllPathsRx=ABeamformingVectors.conj()@AAntennaResponses.T
ZarrayGainAllPathsRx=ZBeamformingVectors.conj()@ZAntennaResponses.T

# plot of one single path without power effects

arrayResponseOnePath =  ZarrayGainAllPathsRx[:,None,0]*AarrayGainAllPathsRx[None,:,0]
arrayResponseOnePathdBtrunc30 = np.maximum(10*np.log10(Nant*Nant*np.abs(arrayResponseOnePath)**2),-30)

fig_ctr+=1
fig = plt.figure(fig_ctr)
Aang,Zang = np.meshgrid(azimut_plot,zenit_plot)
plt.pcolor(Aang,Zang, arrayResponseOnePathdBtrunc30, cmap=cm.jet)
plt.gca().invert_yaxis()#this is so ZoA=0 is seen 'up' in the plot
plt.colorbar(label = 'Analog Beam Gain dB')
plt.xticks(ticks=np.pi*np.array([0,.5,1,1.5,2]),labels=['0','$\\frac{\\pi}{2}$','$\\pi$','$\\frac{3\\pi}{2}$','$2\\pi$'])
plt.yticks(ticks=np.pi*np.array([0,.25,.5,.75,1]),labels=['0','$\\frac{\\pi}{4}$','$\\frac{\\pi}{2}$','$\\frac{3\\pi}{4}$','$\\pi$',])
plt.xlabel('AoA')
plt.ylabel('ZoA')
plt.title('%dx%d UPA directivity for 1st subpath AoA=%.1fº ZoA=%.1fº'%(Nant,Nant,subpaths.loc[0,0].AoA,subpaths.loc[0,0].ZoA))


fig_ctr+=1
fig = plt.figure(fig_ctr)
sphereX = np.cos(Aang)*np.cos(np.pi/2-Zang)
sphereY = np.sin(Aang)*np.cos(np.pi/2-Zang)
sphereZ = np.sin(np.pi/2-Zang)
ax = fig.add_subplot(111, projection='3d')
manual_colors = (arrayResponseOnePathdBtrunc30 -np.min(arrayResponseOnePathdBtrunc30) )/(np.max(arrayResponseOnePathdBtrunc30)-np.min(arrayResponseOnePathdBtrunc30))
surf=ax.plot_surface(sphereX, sphereY, sphereZ,rstride=1, cstride=1, facecolors=cm.jet(manual_colors), linewidth=0, antialiased=False)
ax.set_xlabel('cos(AoA)')
ax.set_ylabel('sin(AoA)')
ax.set_zlabel('cos(ZoA)')
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cm.jet),ax=ax,shrink=0.8,label = 'Analog Beam Gain dB')
cbar.set_ticks((np.arange(-30,30,10) -np.min(arrayResponseOnePathdBtrunc30) )/(np.max(arrayResponseOnePathdBtrunc30)-np.min(arrayResponseOnePathdBtrunc30)))
cbar.set_ticklabels(['%.0f dB'%x for x in np.arange(-30,30,10)])
plt.title('%dx%d UPA directivity for 1st subpath AoA=%.1fº ZoA=%.1fº'%(Nant,Nant,subpaths.loc[0,0].AoA,subpaths.loc[0,0].ZoA))

fig_ctr+=1
fig = plt.figure(fig_ctr)
radius=np.maximum(arrayResponseOnePathdBtrunc30+25,0)
sphereX = radius*np.cos(Aang)*np.cos(np.pi/2-Zang)
sphereY = radius*np.sin(Aang)*np.cos(np.pi/2-Zang)
sphereZ = radius*np.sin(np.pi/2-Zang)
ax = fig.add_subplot(111, projection='3d')
manual_colors = (arrayResponseOnePathdBtrunc30 -np.min(arrayResponseOnePathdBtrunc30) )/(np.max(arrayResponseOnePathdBtrunc30)-np.min(arrayResponseOnePathdBtrunc30))
surf=ax.plot_surface(sphereX, sphereY, sphereZ,rstride=1, cstride=1, facecolors=cm.jet(manual_colors), linewidth=0, antialiased=False)
ax.set_xlabel('cos(AoA)')
ax.set_ylabel('sin(AoA)')
ax.set_zlabel('cos(ZoA)')
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cm.jet),ax=ax,shrink=0.8,label = 'Analog Beam Gain dB')
cbar.set_ticks((np.arange(-30,30,10) -np.min(arrayResponseOnePathdBtrunc30) )/(np.max(arrayResponseOnePathdBtrunc30)-np.min(arrayResponseOnePathdBtrunc30)))
cbar.set_ticklabels(['%.0f dB'%x for x in np.arange(-30,30,10)])
plt.title('%dx%d UPA directivity for 1st subpath AoA=%.1fº ZoA=%.1fº'%(Nant,Nant,subpaths.loc[0,0].AoA,subpaths.loc[0,0].ZoA))


 #plot of all the paths in the channel


channelResponseCombined =  np.sum( ZarrayGainAllPathsRx[:,None,:]*AarrayGainAllPathsRx[None,:,:]*pathAmplitudes[None,None,:] , axis=2) 
channelResponseCombineddB = np.maximum(10*np.log10(Nant*Nant*np.abs(channelResponseCombined)**2),-30)

fig_ctr+=1
fig = plt.figure(fig_ctr)
Aang,Zang = np.meshgrid(azimut_plot,zenit_plot)
plt.pcolor(Aang,Zang, channelResponseCombineddB, cmap=cm.coolwarm)
plt.gca().invert_yaxis()#this is so ZoA=0 is seen 'up' in the plot
plt.colorbar(label = 'Analog Beam Gain dB')
plt.xticks(ticks=np.pi*np.array([0,.5,1,1.5,2]),labels=['0','$\\frac{\\pi}{2}$','$\\pi$','$\\frac{3\\pi}{2}$','$2\\pi$'])
plt.yticks(ticks=np.pi*np.array([0,.25,.5,.75,1]),labels=['0','$\\frac{\\pi}{4}$','$\\frac{\\pi}{2}$','$\\frac{3\\pi}{4}$','$\\pi$',])
plt.xlabel('AoA')
plt.ylabel('ZoA')
plt.title('sum of %dx%d UPA directivity times all chan coefs'%(Nant,Nant))

for n in range(nClusters):   
    Nsp=subpaths.loc[n,:].shape[0]
    plt.scatter(np.mod(subpaths.loc[n,:].AoA*np.pi/180,2*np.pi),np.mod(subpaths.loc[n,:].ZoA*np.pi/180,2*np.pi),color=cm.jet(n/(nClusters-1)),marker='o')


fig_ctr+=1
fig = plt.figure(fig_ctr)
sphereX = np.cos(Aang)*np.cos(np.pi/2-Zang)
sphereY = np.sin(Aang)*np.cos(np.pi/2-Zang)
sphereZ = np.sin(np.pi/2-Zang)
ax = fig.add_subplot(111, projection='3d')
manual_colors = (channelResponseCombineddB -np.min(channelResponseCombineddB) )/(np.max(channelResponseCombineddB)-np.min(channelResponseCombineddB))
surf=ax.plot_surface(sphereX, sphereY, sphereZ,rstride=1, cstride=1, facecolors=cm.jet(manual_colors), linewidth=0, antialiased=False)
ax.set_xlabel('cos(AoA)')
ax.set_ylabel('sin(AoA)')
ax.set_zlabel('cos(ZoA)')
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cm.jet),ax=ax,shrink=0.8,label = 'Analog Beam Gain dB')
cbar.set_ticks((np.arange(-30,30,10) -np.min(arrayResponseOnePathdBtrunc30) )/(np.max(arrayResponseOnePathdBtrunc30)-np.min(arrayResponseOnePathdBtrunc30)))
cbar.set_ticklabels(['%.0f dB'%x for x in np.arange(-30,30,10)])
plt.title('sum of %dx%d UPA directivity times all chan coefs'%(Nant,Nant))


fig_ctr+=1
fig = plt.figure(fig_ctr)
radius=np.maximum(channelResponseCombineddB+25,0)
sphereX = radius*np.cos(Aang)*np.cos(np.pi/2-Zang)
sphereY = radius*np.sin(Aang)*np.cos(np.pi/2-Zang)
sphereZ = radius*np.sin(np.pi/2-Zang)
ax = fig.add_subplot(111, projection='3d')
manual_colors = (channelResponseCombineddB -np.min(channelResponseCombineddB) )/(np.max(channelResponseCombineddB)-np.min(channelResponseCombineddB))
surf=ax.plot_surface(sphereX, sphereY, sphereZ,rstride=1, cstride=1, facecolors=cm.jet(manual_colors), linewidth=0, antialiased=False)
ax.set_xlabel('cos(AoA)')
ax.set_ylabel('sin(AoA)')
ax.set_zlabel('cos(ZoA)')
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cm.jet),ax=ax,shrink=0.8,label = 'Analog Beam Gain dB')
cbar.set_ticks((np.arange(-30,30,10) -np.min(arrayResponseOnePathdBtrunc30) )/(np.max(arrayResponseOnePathdBtrunc30)-np.min(arrayResponseOnePathdBtrunc30)))
cbar.set_ticklabels(['%.0f dB'%x for x in np.arange(-30,30,10)])
plt.title('sum of %dx%d UPA directivity times all chan coefs'%(Nant,Nant))
plt.savefig("../Figures/radiationDoA.eps")
