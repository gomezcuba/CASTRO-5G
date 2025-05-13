#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 10:23:33 2025

@author: fgomez
"""

import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('../')
from CASTRO5G import multipathChannel as mc
plt.close('all')
Nrft=1
Nrfr=2
Nd=2
Na=2
Nsym=5
K=64
# Lista de pilotos a comparar
pilots_config = [
    ("DMRS", "DMRS")
]
M_PSK=2
for ipilot, (pilot_name, pilot_alg) in enumerate(tqdm(pilots_config, desc="Pilot Types: ")):
    pilgen = mc.MIMOPilotChannel(pilot_alg,M_PSK,Nsym,Nd)
    (wp, vp) = pilgen.generatePilots(Nsym*K*Nrft, Na, Nd, Npr=Nsym*K*Nrfr,
                                    rShape=(Nsym, K, Nrfr, Na), tShape=(Nsym, K, Nd, Nrft))
    # print("wp=",wp)
    # print("vp=",vp)
    # print("wp_shape=",wp.shape)
    # print("vp_shape=",vp.shape)