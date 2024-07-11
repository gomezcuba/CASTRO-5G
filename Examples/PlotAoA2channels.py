#!/usr/bin/python
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
from itertools import cycle
# equivalent to rcParams['animation.html'] = 'html5'
rc('animation', html='html5')
from matplotlib import cm
import os

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as pg
from CASTRO5G import multipathChannel as mc

model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=False,smallCorrDist=1)
plinfo1, macro1, clusters1, subpaths1 = model.create_channel((0, 0, 10), (10, 2, 1))
plinfo2, macro2, clusters2, subpaths2 = model.create_channel((0, 0, 10), (10, 2.5, 1))
plinfo3, macro3, clusters3, subpaths3 = model.create_channel((0, 0, 10), (10, 5, 1))

los1, PLfree1, SF1 = plinfo1
los2, PLfree2, SF2 = plinfo2
los3, PLfree3, SF3 = plinfo3

nClusters1 = clusters1.shape[0]
nClusters2 = clusters2.shape[0]
nClusters3 = clusters3.shape[0]

plt.close('all')
fig_ctr = 0

fig_ctr += 1
fig1 = plt.figure(fig_ctr)

for n in range(nClusters1):
    Nsp=subpaths1.loc[n,:].shape[0]
    pathAmplitudesdBtrunc25 = np.maximum(10*np.log10( subpaths1.loc[n,:].P ),-45)
    plt.polar(
        np.tile(subpaths1.loc[n,:].AoA*np.pi/180,(2,1)),
        np.vstack([-40 * np.ones((1, Nsp)), pathAmplitudesdBtrunc25]),
        ":",
        color=cm.jet(n / (nClusters1 - 1))
    )
    plt.scatter(
        np.tile(subpaths1.loc[n,:].AoA*np.pi/180,(2,1)),
        np.vstack([-40 * np.ones((1, Nsp)), pathAmplitudesdBtrunc25]),
        color=cm.jet(n / (nClusters1 - 1)),
        marker="<"
    )
#plt.xticks(np.arange(0, 2*np.pi, np.pi/36))  # División cada 5 grados
plt.yticks(ticks=[-40, -30, -20, -10], labels=['-40dB', '-30dB', '-20dB', '-10dB'])

# plot of rx AoAs and channel gains for channel 2
fig_ctr += 1
fig2 = plt.figure(fig_ctr)
for n in range(nClusters2):
    Nsp=subpaths2.loc[n,:].shape[0]
    pathAmplitudesdBtrunc25 = np.maximum(10*np.log10( subpaths2.loc[n,:].P ),-45)
    plt.polar(
        np.tile(subpaths2.loc[n,:].AoA*np.pi/180,(2,1)),
        np.vstack([-40 * np.ones((1, Nsp)), pathAmplitudesdBtrunc25]),
        ":",
        color=cm.jet(n / (nClusters2 - 1))
    )
    plt.scatter(
        np.tile(subpaths2.loc[n,:].AoA*np.pi/180,(2,1)),
        np.vstack([-40 * np.ones((1, Nsp)), pathAmplitudesdBtrunc25]),
        color=cm.jet(n / (nClusters2 - 1)),
        marker="<"
    )
#plt.xticks(np.arange(0, 2*np.pi, np.pi/36))  # División cada 5 grados
plt.yticks(ticks=[-40, -30, -20, -10], labels=['-40dB', '-30dB', '-20dB', '-10dB'])



# 2D polar plots of AoA for channel 3

fig_ctr += 1
fig3 = plt.figure(fig_ctr)
for n in range(nClusters3):
    Nsp=subpaths3.loc[n,:].shape[0]
    pathAmplitudesdBtrunc25 = np.maximum(10*np.log10( subpaths3.loc[n,:].P ),-45)
    plt.polar(
        np.tile(subpaths3.loc[n,:].AoA*np.pi/180,(2,1)),
        np.vstack([-40 * np.ones((1, Nsp)), pathAmplitudesdBtrunc25]),
        ":",
        color=cm.jet(n / (nClusters3 - 1))
    )
    plt.scatter(
        np.tile(subpaths3.loc[n,:].AoA*np.pi/180,(2,1)),
        np.vstack([-40 * np.ones((1, Nsp)), pathAmplitudesdBtrunc25]),
        color=cm.jet(n / (nClusters3 - 1)),
        marker="<"
    )
#plt.xticks(np.arange(0, 2*np.pi, np.pi/36))  # División cada 5 grados
plt.yticks(ticks=[-40, -30, -20, -10], labels=['-40dB', '-30dB', '-20dB', '-10dB'])
plt.title('Canal 3')



# Crear figura con dos subfiguras
fig = plt.figure(figsize=(12, 6))

# Subfigura 1: Canal 1
fig_ctr += 1
ax1 = fig.add_subplot(121, polar=True)


for n in range(nClusters1):
    Nsp=subpaths1.loc[n,:].shape[0]
    pathAmplitudesdBtrunc25 = np.maximum(10*np.log10( subpaths1.loc[n,:].P ),-45)
    plt.polar(
        np.tile(subpaths1.loc[n,:].AoA*np.pi/180,(2,1)),
        np.vstack([-40 * np.ones((1, Nsp)), pathAmplitudesdBtrunc25]),
        ":",
        color=cm.jet(n / (nClusters1 - 1))
    )
    plt.scatter(
        np.tile(subpaths1.loc[n,:].AoA*np.pi/180,(2,1)),
        np.vstack([-40 * np.ones((1, Nsp)), pathAmplitudesdBtrunc25]),
        color=cm.jet(n / (nClusters1 - 1)),
        marker="<"
    )

#plt.xticks(np.arange(0, 2*np.pi, np.pi/36))  # División cada 5 grados
ax1.set_yticks(ticks=[-40, -30, -20, -10])
ax1.set_yticklabels(['-40dB', '-30dB', '-20dB', '-10dB'])
ax1.set_title('Canal 1')

# Subfigura 2: Canal 2
ax2 = fig.add_subplot(122, polar=True)

# plot of rx AoAs and channel gains for channel 2
for n in range(nClusters2):
    Nsp=subpaths2.loc[n,:].shape[0]
    pathAmplitudesdBtrunc25 = np.maximum(10*np.log10( subpaths2.loc[n,:].P ),-45)
    plt.polar(
        np.tile(subpaths2.loc[n,:].AoA*np.pi/180,(2,1)),
        np.vstack([-40 * np.ones((1, Nsp)), pathAmplitudesdBtrunc25]),
        ":",
        color=cm.jet(n / (nClusters2 - 1))
    )
    plt.scatter(
        np.tile(subpaths2.loc[n,:].AoA*np.pi/180,(2,1)),
        np.vstack([-40 * np.ones((1, Nsp)), pathAmplitudesdBtrunc25]),
        color=cm.jet(n / (nClusters2 - 1)),
        marker="<"
    )
ax2.set_yticks(ticks=[-40, -30, -20, -10])
ax2.set_yticklabels(['-40dB', '-30dB', '-20dB', '-10dB'])
ax2.set_title('Canal 2')

plt.tight_layout()

plt.show()

