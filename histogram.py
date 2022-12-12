# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:48:43 2022

@author: user
"""

import threeGPPMultipathGenerator as pg

import matplotlib.pyplot as plt
import numpy as np
import math as mt

model = pg.ThreeGPPMultipathChannelModel(sce="UMi")

txPos = (0,0,25)
rxPos = (27.34,56.65,1.5)
rxPos1 = (2.34,160.65,1.5)
channel = []
for i in range(50):    
    channel.append(model.create_channel(txPos,rxPos))
    channel.append(model.create_channel(txPos,rxPos1))
#channel = model.create_channel(txPos,rxPos)

macro = channel[0]
print(macro)
"""
ds = float(macro.ds)*1e6

small = channel[1]

delaysList = []

for i in range(len(small)):
    delaysList.append(small[i][0]) #10x12

delay = []
for i in range(len(delaysList)):
    for j in range(len(delaysList[i])):
        delay.append(float(delaysList[i][j])*1e6)
        

numBarras = 13
theoryFunction = []
tau = np.sort(delay)

#n,bins,patches = plt.hist(delay,numBarras, weights= (np.ones_like(delay)/np.sum(delay)))

bottom, top = plt.ylim()

for i in range(len(delay)):
    theoryFunction.append(np.exp(-(tau[i]/(ds*3))))


final = []
for i in range(len(theoryFunction)):
    if theoryFunction[i] == 1.0 and theoryFunction[0]:
        final.append(theoryFunction[i])
    else:
        final.append(theoryFunction[i])

n,bins,patches = plt.hist(delay,numBarras, weights= (np.ones_like(delay)/np.sum(delay)))
x = np.linspace(0,np.max(delay),len(theoryFunction))
plt.plot(x,theoryFunction,color='red')
plt.show()
"""