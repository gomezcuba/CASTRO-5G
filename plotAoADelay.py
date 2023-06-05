#!/usr/bin/python

import threeGPPMultipathGenerator as pg
import multipathChannel as mc

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

plt.close('all')
fig_ctr=0

model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=False)
plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
tau,powC,AOA,AOD,ZOA,ZOD = clusters.T.to_numpy()
nClusters=tau.size
los, PLfree, SF = plinfo
tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths.T.to_numpy()

#3D delay-and-polar plots of delay vs AoA
AoAs = AOA_sp*np.pi/180#radians
delays = tau_sp*1e9#nanoseconds
Npath=np.size(delays)
pathAmplitudes = np.sqrt( pow_sp )*np.exp(2j*np.pi*np.random.rand(Npath))
Npath=np.size(AoAs)

fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = Axes3D(fig)



#since there is no plot.polar() in 3D, we need to draw the axis manually
#polar "circle" levels axis
dBlevels=[-30,-20,-10,0]
dBat0polar=-40
Npointsplot=1001
angles_plot = np.linspace(0,2*np.pi,Npointsplot)
for dBref in dBlevels:
    radius=dBref - dBat0polar
    ax.plot3D(radius*np.cos(angles_plot),radius*np.sin(angles_plot),-np.ones_like(angles_plot),color='k')
    ax.text3D(0,radius,0,'%s dB'%dBref,color='k')

#polar "pizza alice" angles axis
radialLinesAngles = np.arange(0,2*np.pi,2*np.pi/8)
maxRad_plot=np.max(dBlevels)-dBat0polar+2
radius_plot=np.linspace(0,maxRad_plot,2)
for labelAngle in radialLinesAngles:    
    ax.plot3D(radius_plot*np.cos(labelAngle),radius_plot*np.sin(labelAngle),-np.ones_like(radius_plot),color='k')
    ax.text3D(maxRad_plot*np.cos(labelAngle),maxRad_plot*np.sin(labelAngle),-1,'$\\frac{% .0f}{4}\\pi$'%(4*labelAngle/np.pi),color='k')
ax.text3D(maxRad_plot*np.cos(np.pi/16),maxRad_plot*np.sin(np.pi/16),-1,'AoA',color='k')
#vertical delay axis
maxdel=np.max(delays)
maxdelCentenas = np.ceil(maxdel/100)*100 # for example if max delay is 437ns, the axis goes to 500ns.
ax.plot3D([0,0],[0,0],[0,maxdelCentenas],color='k')
ax.text3D(0,0,maxdelCentenas,"delay [ns]",color='k')
print(nClusters)
# compute the response of the antenna array with Nant antennas
Nant = 16
AntennaResponses =mc.fULA(AoAs,Nant)
# compute the "beamforming vector". This vector is multiplied by the "response" when we want to receive from the desired angle
BeamformingVectors =mc.fULA(angles_plot,Nant)

arrayGainAllPaths=(AntennaResponses.transpose([0,2,1]).conj()@BeamformingVectors[:,None,:,:]).reshape((Npointsplot,Npath))

clusterInd = subpaths.reset_index(inplace=False).loc[:,'n']
for pind in range(0,Npath):#plot3D needs to be called 1 line at a time
    clr = cm.jet(clusterInd[pind]/(nClusters-1))
    coefdB=10*np.log10(np.abs(pathAmplitudes[pind])**2)
    radius = coefdB - dBat0polar
    x=np.maximum(radius,0)*np.cos(AoAs[pind])
    y=np.maximum(radius,0)*np.sin(AoAs[pind])
    #plot the paths
    ax.plot3D([0,x],[0,y],[delays[pind],delays[pind]],color=clr)
    ax.scatter3D(x,y,delays[pind],marker='o',color=clr)
fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = Axes3D(fig)

#since there is no plot.polar() in 3D, we need to draw the axis manually
#polar "circle" levels axis
dBlevels=[-30,-20,-10,0]
dBat0polar=-40
Npointsplot=1001
angles_plot = np.linspace(0,2*np.pi,Npointsplot)
for dBref in dBlevels:
    radius=dBref - dBat0polar
    ax.plot3D(radius*np.cos(angles_plot),radius*np.sin(angles_plot),-np.ones_like(angles_plot),color='k')
    ax.text3D(0,radius,0,'%s dB'%dBref,color='k')

#polar "pizza alice" angles axis
radialLinesAngles = np.arange(0,2*np.pi,2*np.pi/8)
maxRad_plot=np.max(dBlevels)-dBat0polar+2
radius_plot=np.linspace(0,maxRad_plot,2)
for labelAngle in radialLinesAngles:    
    ax.plot3D(radius_plot*np.cos(labelAngle),radius_plot*np.sin(labelAngle),-np.ones_like(radius_plot),color='k')
    ax.text3D(maxRad_plot*np.cos(labelAngle),maxRad_plot*np.sin(labelAngle),-1,'$\\frac{% .0f}{4}\\pi$'%(4*labelAngle/np.pi),color='k')
ax.text3D(maxRad_plot*np.cos(np.pi/16),maxRad_plot*np.sin(np.pi/16),-1,'AoA',color='k')
#vertical delay axis
maxdel=np.max(delays)
maxdelCentenas = np.ceil(maxdel/100)*100 # for example if max delay is 437ns, the axis goes to 500ns.
ax.plot3D([0,0],[0,0],[0,maxdelCentenas],color='k')
ax.text3D(0,0,maxdelCentenas,"delay [ns]",color='k')

# compute the response of the antenna array with Nant antennas
Nant = 16
AntennaResponses =mc.fULA(AoAs,Nant)
Npointsplot=1001
# compute the "beamforming vector". This vector is multiplied by the "response" when we want to receive from the desired angle
angles_plot = np.linspace(0,2*np.pi,Npointsplot)
BeamformingVectors =mc.fULA(angles_plot,Nant)

arrayGainAllPaths=(AntennaResponses.transpose([0,2,1]).conj()@BeamformingVectors[:,None,:,:]).reshape((Npointsplot,Npath))

for pind in range(0,Npath):#plot3D needs to be called 1 line at a time
    clr = cm.jet(clusterInd[pind]/(nClusters-1))
    coefdB=10*np.log10(np.abs(pathAmplitudes[pind])**2)
    radius = coefdB - dBat0polar
    x=np.maximum(radius,0)*np.cos(AoAs[pind])
    y=np.maximum(radius,0)*np.sin(AoAs[pind])
    #plot the paths
    ax.plot3D([0,x],[0,y],[delays[pind],delays[pind]],color=clr)
    ax.scatter3D(x,y,delays[pind],marker='o',color=clr)
    #plot the array gains for each path
    chanGaindBtrunc=np.maximum(coefdB+10*np.log10(np.abs(arrayGainAllPaths[:,pind])**2),dBat0polar)
    radius = chanGaindBtrunc - dBat0polar
    x=radius*np.cos(angles_plot)
    y=radius*np.sin(angles_plot)
    ax.plot3D(x,y,delays[pind]*np.ones_like(x),linestyle='-.',marker='x',color=clr)
    
#the code below makes a "gif animation" that helps view the 3D but is not imprescindible
    
# from matplotlib import animation, rc
# rc('animation', html='html5')
#plt.savefig('animation_frame0.png')
#ax.view_init(azim=-61)
#plt.savefig('animation_frame1.png')
#
#import os
#os.system('convert -delay 10 -loop 0 animation_frame0.png animation_frame1.png animated.gif')
#
## animation function. This is called sequentially
#def animate(i):
#    if ax.azim==-60:
#        ax.view_init(azim=-61)
#    else:
#        ax.view_init(azim=-60)
#    return (ax,)
#
#anim = animation.FuncAnimation(fig, animate, frames=2, interval=100)
#
#anim.save('./testanim.gif', writer='imagemagick', fps=10, progress_callback =  lambda i, n: print(f'Saving frame {i} of {n}'))