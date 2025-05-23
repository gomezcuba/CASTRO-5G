#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 10:23:33 2025

@author: fgomez
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import sys
sys.path.append('../')
from CASTRO5G import multipathChannel as mc
plt.close('all')

Nd=2
Na=2
Nrft=1
Nrfr=2 # Tienes que poner aqui Na cuando uses wp matrices identidad
Nsym=6
K=64

# Lista de pilotos a comparar
pilots_config = [
    ("DMRS", "DMRS")
]

algConfig={
    "M":2,
    "MappingType": 'A',
    "SymbolAllocation": (0, 14),
    "DMRSConfigurationType": 1,
    "DMRSLength": 1,
    "DMRSAdditionalPosition": 0,
    "DMRSTypeAPosition": 2,
    "NIDNSCID": 10,
    "NSCID": 0
    }

for ipilot, (pilot_name, pilot_alg) in enumerate(tqdm(pilots_config, desc="Pilot Types: ")):
    pilgen = mc.MIMOPilotChannel(pilot_alg,algConfig)
    (wp, vp) = pilgen.generatePilots(Nsym*K*Nrft, Na, Nd, Npr=Nsym*K*Nrfr,
                                    rShape=(Nsym, K, Nrfr, Na), tShape=(Nsym, K, Nd, Nrft))
    # print("wp=",wp)
    # print("vp=",vp)
    print("wp_shape=",wp.shape)
    print("vp_shape=",vp.shape)
    fig = plt.figure(1)
    Nport = vp.shape[2]
    for cp in range(Nport):
        ax = fig.add_subplot(Nport,1,cp+1)
        # ax = plt.figure(cp)
        unique_vals,inds = np.unique(vp[:,:,cp,0],return_inverse=True)        
        im=ax.pcolor(inds.reshape(Nsym,K).T,cmap=cm.jet)
        # im=plt.pcolor(inds.reshape(5,64).T,cmap=cm.jet)
        cbar = plt.colorbar(im, ax=ax, ticks=np.arange(len(unique_vals)))
        cbar.ax.set_yticklabels([f"{val:.2f}" for val in unique_vals])