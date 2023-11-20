#!/usr/bin/python

from CASTRO5G import threeGPPMultipathGenerator as pg
from CASTRO5G import multipathChannel as mc

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
rc('animation', html='html5')
from matplotlib import cm

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as pg
from CASTRO5G import multipathChannel as mc

plt.close('all')
fig_ctr=0

model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=True)
model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=True)
plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
los, PLfree, SF = plinfo
nClusters = clusters.shape[0]
nNLOSsp=subpaths.loc[1,:].shape[0]#cluster 0 may include the LOS path, 1 and onwards only nlos

#TODO - insert adapted AOA and compare response

#4D  color intensity plots vs delay, AoA and AoD grid

pathAmplitudes = ( np.sqrt( subpaths.P )*np.exp(1j* subpaths.phase00) ).to_numpy()

#4D  color intensity plots vs delay, AoA and AoD grid

#DEC
Ts=2 #ns
Ds=np.max(subpaths.TDOA*1e9)
Ntaps = int(np.ceil(Ds/Ts))
n=np.linspace(0,Ntaps-1,Ntaps)
pulses = np.sinc(n[:,None]-subpaths.TDOA.to_numpy()*1e9/Ts)

# array responses
Nant = 16
AntennaResponsesRx =mc.fULA(subpaths.AOA.to_numpy()*np.pi/180,Nant)
AntennaResponsesTx =mc.fULA(subpaths.AOD.to_numpy()*np.pi/180,Nant)
Npointsplot=2*Nant #in this case we use only a few beamforming vectors
angles_plot = np.linspace(0,2*np.pi,Npointsplot)
BeamformingVectors =mc.fULA(angles_plot,Nant)
arrayGainAllPathsRx=(AntennaResponsesRx.transpose([0,2,1]).conj()@BeamformingVectors[:,None,:,:])[:,:,0,0]
arrayGainAllPathsTx=(AntennaResponsesTx.transpose([0,2,1]).conj()@BeamformingVectors[:,None,:,:])[:,:,0,0]

hnArray = np.sum(pulses[:,None,None,:]*arrayGainAllPathsTx[None,:,None,:]*arrayGainAllPathsRx[None,None,:,:]*pathAmplitudes[None,None,None,:],axis=3)

NtapsPerFigure = 10

Nfigs = int(np.ceil(Ntaps/NtapsPerFigure))
chanGainsdB=10*np.log10(np.abs(hnArray)**2)
manual_colors= (chanGainsdB-np.min(chanGainsdB)) / ( np.max(chanGainsdB)-np.min(chanGainsdB)  )

for nfig in range(0,Nfigs):
    fig_ctr+=1
    # fig = plt.figure(fig_ctr)
    fig = plt.figure(fig_ctr,figsize=(12, 9),dpi=80)#size is in in with dpi converting to pixels
    ax = fig.add_subplot(111, projection='3d')#Axes3D(fig)
    # ax.set_box_aspect((4, 25, 4))
    # ax.set_aspect("equal")
    ax.set_proj_type('ortho')
    ax.set_box_aspect((1, 4, 1))
    ax.view_init(azim=-20,elev=10)
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

plt.savefig("../Figures/sparseChannel4D.eps")