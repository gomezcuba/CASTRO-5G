# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:37:08 2022

@author: user
"""
import threeGPPMultipathGenerator as pg
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd


#-----------------PLOTEO PROBABILIDAD LOS-------------------------------
model = pg.ThreeGPPMultipathChannelModel(sce="UMi")
model1 = pg.ThreeGPPMultipathChannelModel(sce="UMa")
model2 = pg.ThreeGPPMultipathChannelModel(sce="RMa")
model3 = pg.ThreeGPPMultipathChannelModel(sce="InH-Office-Open")
model4 = pg.ThreeGPPMultipathChannelModel(sce="InH-Office-Mixed")

distance = []
LOSprobabilityUMi = []
LOSprobabilityUMa = []
LOSprobabilityRMa = []
LOSprobabilityOpen = []
LOSprobabilityMixed = []

for i in range(100):
    LOSprobabilityUMi.append(model.scenarioLosProb(i))
    LOSprobabilityUMa.append(model1.scenarioLosProb(i,1.5))
    LOSprobabilityRMa.append(model2.scenarioLosProb(i))
    LOSprobabilityOpen.append(model3.scenarioLosProb(i))
    LOSprobabilityMixed.append(model4.scenarioLosProb(i))
    distance.append(i)

plt.plot(distance, LOSprobabilityUMi, color = 'tab:red', linestyle = 'dashed' , label = 'UMi')
plt.plot(distance, LOSprobabilityUMa, color = 'tab:blue', linestyle = 'dashed' , label = 'UMa')
plt.plot(distance, LOSprobabilityRMa, color = 'tab:orange', linestyle = 'dashed' , label = 'RMa' )
plt.plot(distance, LOSprobabilityOpen, color = 'tab:green', linestyle = 'dashed' , label = 'InH-Office-Open' )
plt.plot(distance, LOSprobabilityMixed, color = 'tab:purple', linestyle = 'dashed' , label = 'InH-Office-Mixed' )

plt.legend()
plt.grid(axis='both', color='gray')
plt.xlabel('Distance (m)')
plt.ylabel('LOS Probability')
plt.show()