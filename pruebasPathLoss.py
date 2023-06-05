# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:51:53 2022

@author: user
"""

import threeGPPMultipathGenerator as pg
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

model = pg.ThreeGPPMultipathChannelModel(scenario="RMa")
model1 = pg.ThreeGPPMultipathChannelModel(scenario="UMi")
model2 = pg.ThreeGPPMultipathChannelModel(scenario="UMa")
model3 = pg.ThreeGPPMultipathChannelModel(scenario="InH-Office-Mixed")
#model.bLargeBandwidthOption=True

txPos = (0,0,20)
rxPos = (10,0,1.5)
aPos = np.array(txPos)
bPos = np.array(rxPos)
d3D=np.linalg.norm(bPos-aPos)

d2D = rxPos[0]
hbs = txPos[2]
hut = rxPos[2]
h = 5
W = 20



distance = []
pathLossUMiNLOS = []
pathLossUMaNLOS = []
pathLossRMaNLOS = []
pathLossInHOpenNLOS = []
pathLossInHMixedNLOS = []
pathlossRMaLOS = []



for i in range(5000):
    d3D=np.sqrt(np.power(i,2) + np.power(hbs-hut,2))
    # print(d3D)
    pathlossRMaLOS.append(model.scenarioPlossRMaLOS(d3D,i))
    pathLossRMaNLOS.append(model.scenarioPlossRMaNLOS(d3D,i))
    distance.append(i)
    
plt.plot(distance,pathlossRMaLOS, color='tab:blue', linestyle = 'dashed' , label = 'RMa LOS')
plt.plot(distance,pathLossRMaNLOS, color='tab:red', linestyle = 'solid' , label = 'RMa NLOS')
plt.legend()
plt.xlabel("Distance")
plt.ylabel("Path Loss (dB)")
plt.grid(axis='both', color='gray')
plt.show()









"""

#Pedimos la casilla y la mostramos
print('These are the grids with users: \n')
print(keyResult)
print('Enter the desire grid: ')
entrada = input()
macroUser = dictMacro.get(entrada)

lista = []
for i in range(len(macroUser)):
    lista.append(macroUser[i])

for i in range(len(lista)):
    if isinstance(lista[i], int):
        lista[i] = str(lista[i])
    
print("Macro for BS: (" + str(txPos[0]) +"," + str(txPos[1]) + ")m and UT: " + entrada + " Grid.")
df = pd.Series({'LOS' : str(lista[0]), 'Path Loss' : lista[1], 'DS' : lista[2], 'ASA' : lista[3],'ASD' : lista[4],'ZSA' : lista[5],'ZSD' : lista[6],'K' : lista[7],'SF' : lista[8],'ZOD' : str(lista[9])})
print(df)


#-----------------------------------
   
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    major_ticks = np.arange(0, 2250, 250)
    ax.plot(x,y, marker='*')
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.grid(which='both')
    plt.show()
    
    
    
    
    pathloss = []
    aa = []
    distance = []
    bb = []
    cc = []
    dd = []
    
    #for i in range(100):
      #  aa.append(model.ZODUMiNLOS(i,1.5))
        #bb.append(model1.ZODUMaNLOS(i,1.5))
        #cc.append(model2.ZODRMaNLOS(i+1,1.5))
       # distance.append(i)
    
    
    #plt.plot(distance,aa, color='tab:blue', linestyle = 'dashed' , label = 'Umi')
    #plt.plot(distance,bb, color='tab:red', linestyle = 'dashed' , label = 'UMa')
    #plt.plot(distance,cc, color='tab:green', linestyle = 'dashed' , label = 'RMa')
    
    
    plt.legend()
    plt.xlabel("Distance")
    plt.ylabel("ZSD UMa")
    plt.grid(axis='both', color='gray')
    
    for i in range(200):
        dd.append(model.scenarioLosProb(i))
        aa.append(model1.scenarioLosProb(i))
        aux=np.sqrt(np.power(i,2) + np.power(hbs-hut,2))
        #pathloss.append(model.scenarioPlossInLOS(aux))
        #aa.append(model.scenarioPlossInNLOS(aux))
        #bb.append(model.scenarioPlossUMaLOS(aux,i,hut,True))
        #cc.append(model.scenarioPlossUMaNLOS(aux,i,hut,True))
        distance.append(i)
        
    plt.plot(distance,dd, color='tab:blue', linestyle = 'dashed' , label = 'InH Open')
    plt.plot(distance,aa, color='tab:red', linestyle = 'dashed' , label = 'InH Mixed')
    #plt.plot(distance,pathloss, color='tab:green', linestyle = 'dashed' , label = 'UMa LOS O2I')
    #plt.plot(distance,aa, color='tab:purple', linestyle = 'dashed' , label = 'UMa NLOS O2I')
    plt.legend()
    plt.xlabel("Distance")
    plt.ylabel("LOS Probability")
    plt.grid(axis='both', color='gray')
    
    pathloss = []
    aa = []
    distance = []
    for i in range(1000):
        aux=np.sqrt(np.power(i,2) + np.power(hbs-hut,2))
        pathloss.append(model.scenarioPlossRMaLOS(aux,i))
        aa.append(model.scenarioPlossRMaNLOS(aux,i,W,h,hbs,hut))
        distance.append(i)
        
    plt.plot(distance,pathloss, color='tab:blue', linestyle = 'dashed' , label = 'RMa LOS')
    plt.plot(distance,aa, color='tab:red', linestyle = 'dashed' , label = 'RMa NLOS')
    plt.legend()
    plt.xlabel("Distance")
    plt.ylabel("Path Loss (dB)")
    plt.grid(axis='both', color='gray')
    pathloss = []
    aa = []
    distance = []
    for i in range(175):
        aux=np.sqrt(np.power(i,2) + np.power(hbs-hut,2))
        pathloss.append(model.scenarioPlossInLOS(aux))
        aa.append(model.scenarioPlossInNLOS(aux))
        distance.append(i)
        
    plt.plot(distance,pathloss, color='tab:blue', linestyle = 'dashed' , label = 'InH LOS')
    plt.plot(distance,aa, color='tab:red', linestyle = 'dashed' , label = 'Inh NLOS')
    plt.legend()
    plt.xlabel("Distance")
    plt.ylabel("Path Loss (dB)")
    plt.grid(axis='both', color='gray'
    """