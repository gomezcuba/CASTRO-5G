# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:28:39 2022

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt

from CASTRO5G import threeGPPMultipathGenerator as pg

plt.close('all')

#-------------------------PLOTEO MACRO-------------------------------------
txPos = np.array((0,0,10))
Nusers = 300
corrDist = 15 #m
numberCellsMap = 6
distance = numberCellsMap*corrDist#m

contadorY = []
contadorX = []
key=[]

posX = np.random.uniform(0, distance, size=(Nusers))
posY = np.random.uniform(0, distance, size=(Nusers))
posZ = 1.5*np.ones(Nusers)
users=np.vstack([posX,posY,posZ]).T
cellIndex = (users[:,0:-1]-txPos[0:-1])//corrDist

vals,counts = np.unique(cellIndex,axis=0,return_counts=True)
mm = np.zeros((numberCellsMap,numberCellsMap))
mm[vals[:,0].astype(int),vals[:,1].astype(int)]=counts

plt.figure(1)
plt.pcolor(mm, cmap='RdYlBu_r')
plt.colorbar(label="Number of users", orientation="vertical")
plt.show()

model = pg.ThreeGPPMultipathChannelModel(scenario="UMi")
macroDS = np.zeros(Nusers) 
macroASD = np.zeros(Nusers)
macroASA = np.zeros(Nusers)
macroZSD_lslog = np.zeros(Nusers)
macroZSA = np.zeros(Nusers)
for i in range(Nusers):
    plinfo,macro,clusters,subpaths = model.create_channel(txPos,users[i])
    macroDS[i]= macro.ds
    macroASD[i]= macro.asd
    macroASA[i]= macro.asa
    macroZSD_lslog[i]= macro.zsd_lslog
    macroZSA[i]= macro.zsa
    
fig = plt.figure(2)
ax = plt.gca()
plt.xlim([0, distance+5])
plt.ylim([0, distance+5])
plt.yticks(np.arange(0, distance+5, corrDist))
plt.xticks(np.arange(0, distance+5, corrDist))
plt.grid(axis='both',color='red')
sc = plt.scatter(posX,posY,s=15,c=1e9*macroDS, cmap='RdYlBu_r')
plt.colorbar(label="DS (ns)", orientation="vertical")
plt.title('User Distribution')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')

fig = plt.figure(3)
ax = plt.gca()
plt.xlim([0, distance])
plt.ylim([0, distance])
plt.yticks(np.arange(0, distance+1, corrDist))
plt.xticks(np.arange(0, distance+1, corrDist))
plt.grid(axis='both',color='red')
sc = plt.scatter(posX,posY,s=15,c=macroASD, cmap='RdYlBu_r')
plt.colorbar(label="ASD (º)", orientation="vertical")
plt.title('User Distribution')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')


fig = plt.figure(4)
ax = plt.gca()
plt.xlim([0, distance])
plt.ylim([0, distance])
plt.yticks(np.arange(0, distance+1, corrDist))
plt.xticks(np.arange(0, distance+1, corrDist))
plt.grid(axis='both',color='red')
sc = plt.scatter(posX,posY,s=15,c=macroASA, cmap='RdYlBu_r')
plt.colorbar(label="ASA (º)", orientation="vertical")
plt.title('User Distribution')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')


fig = plt.figure(5)
ax = plt.gca()
plt.xlim([0, distance])
plt.ylim([0, distance])
plt.yticks(np.arange(0, distance+1, corrDist))
plt.xticks(np.arange(0, distance+1, corrDist))
plt.grid(axis='both',color='red')
sc = plt.scatter(posX,posY,s=15,c=macroZSD_lslog, cmap='RdYlBu_r')
plt.colorbar(label="$\sigma_{logZSD}$ (º)", orientation="vertical")
plt.title('User Distribution')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')

fig = plt.figure(6)
ax = plt.gca()
plt.xlim([0, distance])
plt.ylim([0, distance])
plt.yticks(np.arange(0, distance+1, corrDist))
plt.xticks(np.arange(0, distance+1, corrDist))
plt.grid(axis='both',color='red')
sc = plt.scatter(posX,posY,s=15,c=macroZSA, cmap='RdYlBu_r')
plt.colorbar(label="ZSA (º)", orientation="vertical")
plt.title('User Distribution')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')