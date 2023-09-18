#!/usr/bin/python

from CASTRO5G import threeGPPMultipathGenerator as pg
from CASTRO5G import multipathChannel as mc

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

plt.close('all')
fig_ctr=0

model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=True)
plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(20,0,1.5))
los, PLfree, SF = plinfo
nClusters = clusters.shape[0]
nNLOSsp=subpaths.loc[1,:].shape[0]

pathAmplitudes = ( np.sqrt( subpaths.P )*np.exp(1j* subpaths.phase00) ).to_numpy()

fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = Axes3D(fig)

########################################################################
#since there is no plot.polar() in 3D, we need to draw the axis manually
########################################################################
def plot3DPolarCilinder(ax,N,zval,lval,maxv):
    #polar "circle" levels axis    
    angles_plot = np.linspace(0,2*np.pi,Npointsplot)
    for vref in lval:
        radius=vref - zval
        ax.plot3D(radius*np.cos(angles_plot),radius*np.sin(angles_plot),-np.ones_like(angles_plot),color='k')
        ax.text3D(0,radius,0,'%s dB'%vref,color='k')
        
    #polar "pizza alice" angles axis
    radialLinesAngles = np.arange(0,2*np.pi,2*np.pi/8)
    maxRad_plot=np.max(lval)-zval+2
    radius_plot=np.linspace(0,maxRad_plot,2)
    for labelAngle in radialLinesAngles:    
        ax.plot3D(radius_plot*np.cos(labelAngle),radius_plot*np.sin(labelAngle),-np.ones_like(radius_plot),color='k')
        ax.text3D(maxRad_plot*np.cos(labelAngle),maxRad_plot*np.sin(labelAngle),-1,'$\\frac{% .0f}{4}\\pi$'%(4*labelAngle/np.pi),color='k')
        ax.text3D(maxRad_plot*np.cos(np.pi/16),maxRad_plot*np.sin(np.pi/16),-1,'AoA',color='k')
    
    #vertical delay axis
    maxdelCentenas = np.ceil(maxv/100)*100 # for example if max delay is 437ns, the axis goes to 500ns.
    ax.plot3D([0,0],[0,0],[0,maxdelCentenas],color='k')
    ax.text3D(0,0,maxdelCentenas,"delay [ns]",color='k')

plot3DPolarCilinder(ax,101,-40,[-30,-20,-10,0],np.max(np.max(subpaths.tau)*1e9))

###############################################################################
# Plot each path power- AoA - delay profile in analog domain
###############################################################################

for n,m in subpaths.index:#plot3D needs to be called 1 line at a time
    clr = cm.jet(n/(nClusters-1))
    coefdB=10*np.log10( subpaths.loc[n,m].P )
    radius = coefdB - dBat0polar
    x=np.maximum(radius,0)*np.cos(subpaths.loc[n,m].AOA)
    y=np.maximum(radius,0)*np.sin(subpaths.loc[n,m].AOA)
    #plot the paths
    ax.plot3D([0,x],[0,y],[subpaths.loc[n,m].tau*1e9,subpaths.loc[n,m].tau*1e9],color=clr)
    ax.scatter3D(x,y,subpaths.loc[n,m].tau*1e9,marker=(3,0,-90+clusters.loc[n].AOA),color=clr)
fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = Axes3D(fig)

plot3DPolarCilinder(ax,101,-40,[-30,-20,-10,0],np.max(np.max(subpaths.tau)*1e9))

########################################################################
# compute the response of the antenna array with Nant antennas
Nant = 16
# compute the "beamforming vector". This vector is multiplied by the "response" when we want to receive from the desired angle
BeamformingVectors =mc.fULA(angles_plot,Nant)

for n,m in subpaths.index:#plot3D needs to be called 1 line at a time
    clr = cm.jet(n/(nClusters-1))
    
    AntennaResponse1Path =mc.fULA(subpaths.loc[n,m].AOA *np.pi/180 ,Nant)    
    arrayGain1Path=(BeamformingVectors.transpose([0,2,1]).conj()@ AntennaResponse1Path ).reshape(-1)
    arrayGain1PathdBtrunc25 = np.maximum(10*np.log10(subpaths.loc[n,m].P*Nant*np.abs(arrayGain1Path)**2),-40)

    coefdB=10*np.log10( subpaths.loc[n,m].P )
    radius = coefdB - dBat0polar
    x=np.maximum(radius,0)*np.cos(subpaths.loc[n,m].AOA)
    y=np.maximum(radius,0)*np.sin(subpaths.loc[n,m].AOA)
    #plot the paths
    ax.plot3D([0,x],[0,y],[subpaths.loc[n,m].tau*1e9,subpaths.loc[n,m].tau*1e9],color=clr)
    ax.scatter3D(x,y,subpaths.loc[n,m].tau*1e9,marker=(3,0,-90+clusters.loc[n].AOA),color=clr)
    #plot the array gains for each path
    radius = arrayGain1PathdBtrunc25 - dBat0polar
    x=radius*np.cos(angles_plot)
    y=radius*np.sin(angles_plot)
    ax.plot3D(x,y,subpaths.loc[n,m].tau*1e9*np.ones_like(x),linestyle=':',color=clr)