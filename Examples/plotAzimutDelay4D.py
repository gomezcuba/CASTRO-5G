#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as pg
from CASTRO5G import multipathChannel as mc

plt.close('all')
fig_ctr=0

# model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=True)
# model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=True)
# plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
# los, PLfree, SF = plinfo
# nClusters = clusters.shape[0]
# nNLOSsp=subpaths.loc[1,:].shape[0]#cluster 0 may include the LOS path, 1 and onwards only nlos

# #TODO - insert adapted AOA and compare response

# #4D  color intensity plots vs delay, AoA and AoD grid

pathAmplitudes = ( np.sqrt( subpaths.P )*np.exp(1j* subpaths.phase00) ).to_numpy()

#4D  color intensity plots vs delay, AoA and AoD grid

#DEC
Ts=2.5 #ns
Ntaps=20
Ds=Ntaps*Ts
subpaths.TDoA=subpaths.TDoA*Ds/np.max(subpaths.TDoA*1e9)
n=np.linspace(0,Ntaps-1,Ntaps)
pulses = np.sinc(n[:,None]-subpaths.TDoA.to_numpy()*1e9/Ts)

# array responses
Nant = 16
AntennaResponsesRx =mc.fULA(subpaths.AoA.to_numpy()*np.pi/180,Nant)
AntennaResponsesTx =mc.fULA(subpaths.AoD.to_numpy()*np.pi/180,Nant)
Npointsplot=2*Nant #in this case we use only a few beamforming vectors
angles_plot = np.linspace(0,2*np.pi,Npointsplot)
BeamformingVectors =mc.fULA(angles_plot,Nant)
arrayGainAllPathsRx=BeamformingVectors.conj() @AntennaResponsesRx.T
arrayGainAllPathsTx=BeamformingVectors.conj() @AntennaResponsesTx.T

hnArray = np.sum(pulses[:,None,None,:]*arrayGainAllPathsTx[None,:,None,:]*arrayGainAllPathsRx[None,None,:,:]*pathAmplitudes[None,None,None,:],axis=3)

NtapsPerFigure = Ntaps

Nfigs = int(np.ceil(Ntaps/NtapsPerFigure))
chanGainsdB=10*np.log10(np.abs(hnArray)**2)
manual_colors= (chanGainsdB-np.min(chanGainsdB)) / ( np.max(chanGainsdB)-np.min(chanGainsdB)  )

for nfig in range(0,Nfigs):
    fig_ctr+=1
    # fig = plt.figure(fig_ctr)
    fig = plt.figure(fig_ctr,figsize=(16,12))#size is in in with dpi converting to pixels
    ax = fig.add_subplot(projection='3d',aspect='auto')#Axes3D(fig)
    ax.set_proj_type('ortho')
    ax.set_box_aspect((1,3.5,1), zoom=1.15)
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
        # surf = ax.plot_wireframe(X, Y, Z, color='k', linewidth=1)
    
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cm.jet),shrink=0.8,label = 'Directive Array Channel Gain dB')
    cbar.set_ticks((np.arange(np.ceil(np.min(chanGainsdB)/10)*10,np.max(chanGainsdB),10) -np.min(chanGainsdB) )/(np.max(chanGainsdB)-np.min(chanGainsdB)))
    cbar.set_ticklabels(['%.0f dB'%x for x in np.arange(np.ceil(np.min(chanGainsdB)/10)*10,np.max(chanGainsdB),10)])
    ax.set_xlabel('AoD')
    ax.set_ylabel('TDoA')
    ax.set_zlabel('AoA')

plt.savefig("../Figures/sparseChannel4D.svg")
plt.savefig("../Figures/sparseChannel4D.png")