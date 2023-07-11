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

model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=False)
plinfo1, macro1, clusters1, subpaths1 = model.create_channel((0, 0, 10), (0, 1, 1.5))
plinfo2, macro2, clusters2, subpaths2 = model.create_channel((0, 0, 10), (0, 1.5, 1.5))

tau1, powC1, AOA1, AOD1, ZOA1, ZOD1 = clusters1.T.to_numpy()
tau2, powC2, AOA2, AOD2, ZOA2, ZOD2 = clusters2.T.to_numpy()

los1, PLfree1, SF1 = plinfo1
los2, PLfree2, SF2 = plinfo2

nClusters1 = tau1.size
nClusters2 = tau2.size

if los1:
    M1 = max(subpaths1.loc[0, :].index)
    (tau_los1, pow_los1, losAoA1, losAoD1, losZoA1, losZoD1) = subpaths1.loc[(0, M1), :]
    tau_sp1, powC_sp1, AOA_sp1, AOD_sp1, ZOA_sp1, ZOD_sp1 = subpaths1.drop((0, M1)).T.to_numpy()
else:
    M1 = max(subpaths1.loc[0, :].index) + 1
    tau_sp1, powC_sp1, AOA_sp1, AOD_sp1, ZOA_sp1, ZOD_sp1 = subpaths1.T.to_numpy()

if los2:
    M2 = max(subpaths2.loc[0, :].index)
    (tau_los2, pow_los2, losAoA2, losAoD2, losZoA2, losZoD2) = subpaths2.loc[(0, M2), :]
    tau_sp2, powC_sp2, AOA_sp2, AOD_sp2, ZOA_sp2, ZOD_sp2 = subpaths2.drop((0, M2)).T.to_numpy()
else:
    M2 = max(subpaths2.loc[0, :].index) + 1
    tau_sp2, powC_sp2, AOA_sp2, AOD_sp2, ZOA_sp2, ZOD_sp2 = subpaths2.T.to_numpy()

tau_sp1 = tau_sp1.reshape(nClusters1, -1)
powC_sp1 = powC_sp1.reshape(nClusters1, -1)
AOA_sp1 = AOA_sp1.reshape(nClusters1, -1)
AOD_sp1 = AOD_sp1.reshape(nClusters1, -1)
ZOA_sp1 = ZOA_sp1.reshape(nClusters1, -1)
ZOD_sp1 = ZOD_sp1.reshape(nClusters1, -1)

tau_sp2 = tau_sp2.reshape(nClusters2, -1)
powC_sp2 = powC_sp2.reshape(nClusters2, -1)
AOA_sp2 = AOA_sp2.reshape(nClusters2, -1)
AOD_sp2 = AOD_sp2.reshape(nClusters2, -1)
ZOA_sp2 = ZOA_sp2.reshape(nClusters2, -1)
ZOD_sp2 = ZOD_sp2.reshape(nClusters2, -1)

plt.close('all')
fig_ctr = 0

# 2D polar plots of AoA for channel 1
AoAs1 = AOA_sp1.reshape(-1) * np.pi / 180  # radians
Npath1 = np.size(AoAs1)
pathAmplitudes1 = np.sqrt(powC_sp1.reshape(-1)) * np.exp(2j * np.pi * np.random.rand(Npath1))


fig_ctr += 1
fig1 = plt.figure(fig_ctr)
pathAmplitudesdBtrunc25_1 = np.maximum(10 * np.log10(np.abs(pathAmplitudes1) ** 2), -45)

Nsp1 = AOA_sp1.shape[1]

if los1:
    plt.polar(
        losAoA1 * np.pi / 180 * np.ones((2, 1)),
        np.vstack([[-40], 10 * np.log10(pow_los1)]),
        ":",
        color=cm.jet(0)
    )
    plt.scatter(
        losAoA1 * np.pi / 180 * np.ones((2, 1)),
        np.vstack([[-40], 10 * np.log10(pow_los1)]),
        color=cm.jet(0),
        marker="<"
    )

for n in range(nClusters1):
    pathAmplitudes_sp1 = np.sqrt(powC_sp1[n, :]) * np.exp(2j * np.pi * np.random.rand(Nsp1))
    pathAmplitudesdBtrunc25_sp1 = np.maximum(10 * np.log10(np.abs(pathAmplitudes_sp1) ** 2), -45)
    plt.polar(
        AOA_sp1[n, :] * np.pi / 180 * np.ones((2, 1)),
        np.vstack([-40 * np.ones((1, Nsp1)), pathAmplitudesdBtrunc25_sp1]),
        ":",
        color=cm.jet(n / (nClusters1 - 1))
    )
    plt.scatter(
        AOA_sp1[n, :] * np.pi / 180,
        pathAmplitudesdBtrunc25_sp1,
        color=cm.jet(n / (nClusters1 - 1)),
        marker="<"
    )

plt.yticks(ticks=[-40, -30, -20, -10], labels=['-40dB', '-30dB', '-20dB', '-10dB'])

# 2D polar plots of AoA for channel 2
AoAs2 = AOA_sp2.reshape(-1) * np.pi / 180  # radians
Npath2 = np.size(AoAs2)
pathAmplitudes2 = np.sqrt(powC_sp2.reshape(-1)) * np.exp(2j * np.pi * np.random.rand(Npath2))

# plot of rx AoAs and channel gains for channel 2
fig_ctr += 1
fig2 = plt.figure(fig_ctr)
pathAmplitudesdBtrunc25_2 = np.maximum(10 * np.log10(np.abs(pathAmplitudes2) ** 2), -45)

Nsp2 = AOA_sp2.shape[1]

if los2:
    plt.polar(
        losAoA2 * np.pi / 180 * np.ones((2, 1)),
        np.vstack([[-40], 10 * np.log10(pow_los2)]),
        ":",
        color=cm.jet(0)
    )
    plt.scatter(
        losAoA2 * np.pi / 180 * np.ones((2, 1)),
        np.vstack([[-40], 10 * np.log10(pow_los2)]),
        color=cm.jet(0),
        marker="<"
    )

for n in range(nClusters2):
    pathAmplitudes_sp2 = np.sqrt(powC_sp2[n, :]) * np.exp(2j * np.pi * np.random.rand(Nsp2))
    pathAmplitudesdBtrunc25_sp2 = np.maximum(10 * np.log10(np.abs(pathAmplitudes_sp2) ** 2), -45)
    plt.polar(
        AOA_sp2[n, :] * np.pi / 180 * np.ones((2, 1)),
        np.vstack([-40 * np.ones((1, Nsp2)), pathAmplitudesdBtrunc25_sp2]),
        ":",
        color=cm.jet(n / (nClusters2 - 1))
    )
    plt.scatter(
        AOA_sp2[n, :] * np.pi / 180,
        pathAmplitudesdBtrunc25_sp2,
        color=cm.jet(n / (nClusters2 - 1)),
        marker="<"
    )

plt.yticks(ticks=[-40, -30, -20, -10], labels=['-40dB', '-30dB', '-20dB', '-10dB'])

# compute the response of the antenna array with Nant antennas
Nant = 16
AntennaResponses1 = mc.fULA(AoAs1, Nant)
AntennaResponses2 = mc.fULA(AoAs2, Nant)

Npointsplot = 1001
# compute the "beamforming vector". This vector is multiplied by the "response" when we want to receive from the desired angle
angles_plot = np.linspace(0, 2 * np.pi, Npointsplot)
BeamformingVectors1 = mc.fULA(angles_plot, Nant)
BeamformingVectors2 = mc.fULA(angles_plot, Nant)



# Crear figura con dos subfiguras
fig = plt.figure(figsize=(12, 6))

# Subfigura 1: Canal 1
fig_ctr += 1
ax1 = fig.add_subplot(121, polar=True)

# 2D polar plots of AoA for channel 1
AoAs1 = AOA_sp1.reshape(-1) * np.pi / 180  # radians
Npath1 = np.size(AoAs1)
pathAmplitudes1 = np.sqrt(powC_sp1.reshape(-1)) * np.exp(2j * np.pi * np.random.rand(Npath1))
pathAmplitudesdBtrunc25_1 = np.maximum(10 * np.log10(np.abs(pathAmplitudes1) ** 2), -45)
Nsp1 = AOA_sp1.shape[1]

if los1:
    ax1.plot(losAoA1 * np.pi / 180 * np.ones((2, 1)), np.vstack([[-40], 10 * np.log10(pow_los1)]), ':', color=cm.jet(0))
    ax1.scatter(losAoA1 * np.pi / 180 * np.ones((2, 1)), np.vstack([[-40], 10 * np.log10(pow_los1)]), color=cm.jet(0), marker='<')

for n in range(nClusters1):
    pathAmplitudes_sp1 = np.sqrt(powC_sp1[n, :]) * np.exp(2j * np.pi * np.random.rand(Nsp1))
    pathAmplitudesdBtrunc25_sp1 = np.maximum(10 * np.log10(np.abs(pathAmplitudes_sp1) ** 2), -45)
    ax1.plot(AOA_sp1[n, :] * np.pi / 180 * np.ones((2, 1)), np.vstack([-40 * np.ones((1, Nsp1)), pathAmplitudesdBtrunc25_sp1]), ':', color=cm.jet(n / (nClusters1 - 1)))
    ax1.scatter(AOA_sp1[n, :] * np.pi / 180, pathAmplitudesdBtrunc25_sp1, color=cm.jet(n / (nClusters1 - 1)), marker='<')

ax1.set_yticks(ticks=[-40, -30, -20, -10])
ax1.set_yticklabels(['-40dB', '-30dB', '-20dB', '-10dB'])
ax1.set_title('Canal 1')

# Subfigura 2: Canal 2
fig_ctr += 1
ax2 = fig.add_subplot(122, polar=True)

# 2D polar plots of AoA for channel 2
AoAs2 = AOA_sp2.reshape(-1) * np.pi / 180  # radians
Npath2 = np.size(AoAs2)
pathAmplitudes2 = np.sqrt(powC_sp2.reshape(-1)) * np.exp(2j * np.pi * np.random.rand(Npath2))
pathAmplitudesdBtrunc25_2 = np.maximum(10 * np.log10(np.abs(pathAmplitudes2) ** 2), -45)
Nsp2 = AOA_sp2.shape[1]

if los2:
    ax2.plot(losAoA2 * np.pi / 180 * np.ones((2, 1)), np.vstack([[-40], 10 * np.log10(pow_los2)]), ':', color=cm.jet(0))
    ax2.scatter(losAoA2 * np.pi / 180 * np.ones((2, 1)), np.vstack([[-40], 10 * np.log10(pow_los2)]), color=cm.jet(0), marker='<')

for n in range(nClusters2):
    pathAmplitudes_sp2 = np.sqrt(powC_sp2[n, :]) * np.exp(2j * np.pi * np.random.rand(Nsp2))
    pathAmplitudesdBtrunc25_sp2 = np.maximum(10 * np.log10(np.abs(pathAmplitudes_sp2) ** 2), -45)
    ax2.plot(AOA_sp2[n, :] * np.pi / 180 * np.ones((2, 1)), np.vstack([-40 * np.ones((1, Nsp2)), pathAmplitudesdBtrunc25_sp2]), ':', color=cm.jet(n / (nClusters2 - 1)))
    ax2.scatter(AOA_sp2[n, :] * np.pi / 180, pathAmplitudesdBtrunc25_sp2, color=cm.jet(n / (nClusters2 - 1)), marker='<')

ax2.set_yticks(ticks=[-40, -30, -20, -10])
ax2.set_yticklabels(['-40dB', '-30dB', '-20dB', '-10dB'])
ax2.set_title('Canal 2')

plt.tight_layout()

# Mostrar la figura
plt.show()