# -- coding: utf-8 --
"""
Created on Wed Nov 30 16:28:39 2022

@author: user
"""
import threeGPPMultipathGenerator as pg
import numpy as np
import matplotlib.pyplot as plt
import random
import math as m



plt.close('all')

#-------------------------PLOTEO MACRO-------------------------------------
txPos = np.array((0, 0, 10))
Nusers = 300
corrDist = 1  # m
numberCellsMap = 10
distance = numberCellsMap * corrDist  # m

contadorY = []
contadorX = []
key = []

posX = np.random.uniform(0, distance, size=(Nusers))
posY = np.random.uniform(0, distance, size=(Nusers))

posZ = 1.5 * np.ones(Nusers)
users = np.vstack([posX, posY, posZ]).T
cellIndex = (users[:, 0:-1] - txPos[0:-1]) // corrDist

vals, counts = np.unique(cellIndex, axis=0, return_counts=True)
mm = np.zeros((numberCellsMap, numberCellsMap))
mm[vals[:, 0].astype(int), vals[:, 1].astype(int)] = counts

model = pg.ThreeGPPMultipathChannelModel(scenario="UMi",bLargeBandwidthOption=True)
AOA = np.zeros(Nusers) 
tau = np.zeros(Nusers)
AOD = np.zeros(Nusers)
ZOA = np.zeros(Nusers)
ZOD = np.zeros(Nusers)
                    
losState = np.zeros(Nusers,dtype=bool) 


for i in range(Nusers):
    plinfo,macro,clusters,subpaths = model.create_channel(txPos,users[i])
    losState[i],PLdBmu,shadowing = plinfo
    AOA[i]= clusters.AOA[1]
    tau[i]= clusters.tau[1]
    AOD[i]= clusters.AOD[1]
    ZOA[i]= clusters.ZOA[1]
    ZOD[i]= clusters.ZOD[1]
fig=0
fig +=1
fig1 = plt.figure(fig)
ax = plt.gca()
plt.xlim([0, distance])
plt.ylim([0, distance])
plt.yticks(np.arange(0, distance + 1, corrDist))
plt.xticks(np.arange(0, distance + 1, corrDist))
plt.grid(axis='both', color='red')
sc = plt.scatter(posX, posY, s=15, c=AOA, cmap='RdYlBu_r')  
plt.colorbar(sc, label="AOA", orientation="vertical")
plt.title('User Distribution')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
#plt.show()
plt.savefig("1")
fig +=1
fig2 = plt.figure(fig)
ax = plt.gca()
plt.xlim([0, distance])
plt.ylim([0, distance])
plt.yticks(np.arange(0, distance + 1, corrDist))
plt.xticks(np.arange(0, distance + 1, corrDist))
plt.grid(axis='both', color='red')
sc = plt.scatter(posX, posY, s=15, c=tau, cmap='RdYlBu_r')
plt.colorbar(sc, label="tau", orientation="vertical")
plt.title('User Distribution')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
#plt.show()
plt.savefig("2")

fig +=1
fig3 = plt.figure(fig)
ax = plt.gca()
plt.xlim([0, distance])
plt.ylim([0, distance])
plt.yticks(np.arange(0, distance + 1, corrDist))
plt.xticks(np.arange(0, distance + 1, corrDist))
plt.grid(axis='both', color='red')
sc = plt.scatter(posX, posY, s=15, c=AOD, cmap='RdYlBu_r')
plt.colorbar(sc, label="AOD", orientation="vertical")
plt.title('User Distribution')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
#plt.show()
plt.savefig("3")

fig +=1
fig4 = plt.figure(fig)
ax = plt.gca()
plt.xlim([0, distance])
plt.ylim([0, distance])
plt.yticks(np.arange(0, distance + 1, corrDist))
plt.xticks(np.arange(0, distance + 1, corrDist))
plt.grid(axis='both', color='red')
sc = plt.scatter(posX, posY, s=15, c=ZOA, cmap='RdYlBu_r')
plt.colorbar(sc, label="ZOA", orientation="vertical")
plt.title('User Distribution')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
#plt.show()
plt.savefig("4")

fig +=1
fig5 = plt.figure(fig)
ax = plt.gca()
plt.xlim([0, distance])
plt.ylim([0, distance])
plt.yticks(np.arange(0, distance + 1, corrDist))
plt.xticks(np.arange(0, distance + 1, corrDist))
plt.grid(axis='both', color='red')
sc = plt.scatter(posX, posY, s=15, c=ZOD, cmap='RdYlBu_r')
plt.colorbar(sc, label="ZOD", orientation="vertical")
plt.title('User Distribution')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
plt.savefig("5")
plt.show()