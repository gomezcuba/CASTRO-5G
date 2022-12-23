#!/usr/bin/python

import threeGPPMultipathGenerator as pg
import multipathChannel as mc

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
rc('animation', html='html5')
from matplotlib import cm

plt.close('all')
fig_ctr=0

model = pg.ThreeGPPMultipathChannelModel()
model.bLargeBandwidthOption=True
model.create_channel((0,0,10),(40,0,1.5))
chparams = model.dChansGenerated[(0,0,40,0)]

#4D  color intensity plots vs delay, AoA and AoD grid
AoDs = np.array([x.azimutOfDeparture[0] for x in chparams.channelPaths])
AoAs = np.array([x.azimutOfArrival[0] for x in chparams.channelPaths])
delays = np.array([x.excessDelay[0] for x in chparams.channelPaths])
pathAmplitudes = np.array([x.complexAmplitude[0] for x in chparams.channelPaths])
Npath=np.size(AoAs)


#DEC
Ts=5 #ns
Ds=np.max(delays)
Ntaps = int(np.minimum( np.ceil(Ds/Ts), 100 ) )
n=np.linspace(0,Ntaps-1,Ntaps)
pulses = np.sinc(n[:,None]-delays/Ts)

# array responses
Nant = 16
AntennaResponsesRx =mc.fULA(AoAs,Nant)
AntennaResponsesTx =mc.fULA(AoDs,Nant)
Npointsplot=11 #in this case we use only a few beamforming vectors
angles_plot = np.linspace(0,2*np.pi,Npointsplot)
BeamformingVectors =mc.fULA(angles_plot,Nant)
arrayGainAllPathsRx=(AntennaResponsesRx.transpose([0,2,1]).conj()@BeamformingVectors[:,None,:,:]).reshape((Npointsplot,Npath))
arrayGainAllPathsTx=(AntennaResponsesTx.transpose([0,2,1]).conj()@BeamformingVectors[:,None,:,:]).reshape((Npointsplot,Npath))

arrayGainAllPathsTx = arrayGainAllPathsTx.reshape(Npointsplot,1,Npath)#move the txAngle to a new axis
pulses = pulses.reshape(Ntaps,1,1,Npath)#move the delay to a new axis

hnArray = np.sum(pulses*arrayGainAllPathsTx[None,:,:,:]*arrayGainAllPathsRx[None,None,:,:]*pathAmplitudes[None,None,None,:],axis=3)

NtapsPerFigure = 20

Nfigs = int(np.ceil(Ntaps/NtapsPerFigure))
chanGainsdB=10*np.log10(np.abs(hnArray)**2)
manual_colors= (chanGainsdB-np.min(chanGainsdB)) / ( np.max(chanGainsdB)-np.min(chanGainsdB)  )

for nfig in range(0,Nfigs):
    fig_ctr+=1
    fig = plt.figure(fig_ctr,figsize=(25, 4),dpi=80)#size is in in with dpi converting to pixels
    ax = Axes3D(fig)
    ax.set_proj_type('ortho')
    ax.view_init(azim=-3,elev=25)
    X = np.linspace(0, 2*np.pi, Npointsplot)
    Z = np.linspace(0, 2*np.pi, Npointsplot)
    X, Z = np.meshgrid(X, Z)
    for d_tap in range(0, NtapsPerFigure):
        tap = d_tap+NtapsPerFigure*nfig
        if tap < Ntaps:
            C=cm.jet(manual_colors[tap,:,:])
        else:
            C=np.zeros((Npointsplot,Npointsplot,4))
        Y = Ts*(tap+nfig*NtapsPerFigure)*np.ones((Npointsplot,Npointsplot))
        surf = ax.plot_surface(X, Y, Z, facecolors=C, linewidth=0, antialiased=False)
        surf = ax.plot_wireframe(X, Y, Z, color='k', linewidth=1)
    
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cm.jet),shrink=0.8,label = 'Directive Array Channel Gain dB')
    cbar.set_ticks((np.arange(np.ceil(np.min(chanGainsdB)/10)*10,np.max(chanGainsdB),10) -np.min(chanGainsdB) )/(np.max(chanGainsdB)-np.min(chanGainsdB)))
    cbar.set_ticklabels(['%.0f dB'%x for x in np.arange(np.ceil(np.min(chanGainsdB)/10)*10,np.max(chanGainsdB),10)])
