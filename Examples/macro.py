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
numberMacrosCell = 9
cellRadius = numberMacrosCell*corrDist#m

contadorY = []
contadorX = []
key=[]

posX = np.random.uniform(-cellRadius/2, cellRadius/2, size=(Nusers))
posY = np.random.uniform(-cellRadius/2, cellRadius/2, size=(Nusers))
posZ = 1.5*np.ones(Nusers)
users=np.vstack([posX,posY,posZ]).T
macroIndex = (users[:,0:-1]-txPos[0:-1]+corrDist/2)//corrDist

vals,counts = np.unique(macroIndex,axis=0,return_counts=True)
vals=vals+numberMacrosCell//2
mm = np.zeros((numberMacrosCell,numberMacrosCell))
mm[vals[:,0].astype(int),vals[:,1].astype(int)]=counts

plt.figure(1)
X,Y=np.meshgrid(np.arange(0, cellRadius+5, corrDist)-cellRadius/2,np.arange(0, cellRadius+5, corrDist)-cellRadius/2)
plt.pcolor(X,Y,mm, cmap='RdYlBu_r',label=None)
plt.plot(0,0,'^k',label='BS')
plt.colorbar(label="Number of users", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()
plt.show()

model = pg.ThreeGPPMultipathChannelModel(scenario="UMi",corrDistance=corrDist)
losState = np.zeros(Nusers,dtype=bool) 
macroDS = np.zeros(Nusers) 
macroASD = np.zeros(Nusers)
macroASA = np.zeros(Nusers)
macroZSD_lslog = np.zeros(Nusers)
macroZSA = np.zeros(Nusers)
for i in range(Nusers):
    plinfo,macro,clusters,subpaths = model.create_channel(txPos,users[i])
    losState[i],PLdBmu,shadowing = plinfo
    macroDS[i]= macro.ds
    macroASD[i]= macro.asd
    macroASA[i]= macro.asa
    macroZSD_lslog[i]= macro.zsd_lslog
    macroZSA[i]= macro.zsa
    
fig = plt.figure(2)
ax = plt.gca()
plt.xlim([-cellRadius/2, cellRadius/2])
plt.ylim([-cellRadius/2, cellRadius/2])
plt.yticks(np.arange(-cellRadius/2, cellRadius/2+5, corrDist))
plt.xticks(np.arange(-cellRadius/2, cellRadius/2+5, corrDist))
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[losState],posY[losState],s=20,c=1e9*macroDS[losState], marker='o', cmap='RdYlBu_r',label='LOS')
sc = plt.scatter(posX[np.logical_not(losState)],posY[np.logical_not(losState)],s=20,c=1e9*macroDS[np.logical_not(losState)],marker='x', cmap='RdYlBu_r',label='NLOS')
plt.colorbar(label="DS (ns)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()

fig = plt.figure(3)
ax = plt.gca()
plt.xlim([-cellRadius/2, cellRadius/2])
plt.ylim([-cellRadius/2, cellRadius/2])
plt.yticks(np.arange(-cellRadius/2, cellRadius/2+5, corrDist))
plt.xticks(np.arange(-cellRadius/2, cellRadius/2+5, corrDist))
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[losState],posY[losState],s=20,c=macroASD[losState], marker='o', cmap='RdYlBu_r', label='LOS')
sc = plt.scatter(posX[np.logical_not(losState)],posY[np.logical_not(losState)],s=20,c=macroASD[np.logical_not(losState)],marker='x', cmap='RdYlBu_r', label='NLOS')
plt.colorbar(label="ASD (º)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()


fig = plt.figure(4)
ax = plt.gca()
plt.xlim([-cellRadius/2, cellRadius/2])
plt.ylim([-cellRadius/2, cellRadius/2])
plt.yticks(np.arange(-cellRadius/2, cellRadius/2+5, corrDist))
plt.xticks(np.arange(-cellRadius/2, cellRadius/2+5, corrDist))
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[losState],posY[losState],s=20,c=macroASA[losState], marker='o', cmap='RdYlBu_r', label='LOS')
sc = plt.scatter(posX[np.logical_not(losState)],posY[np.logical_not(losState)],s=20,c=macroASA[np.logical_not(losState)], marker='x', cmap='RdYlBu_r', label='NLOS')
plt.colorbar(label="ASA (º)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()


fig = plt.figure(5)
ax = plt.gca()
plt.xlim([-cellRadius/2, cellRadius/2])
plt.ylim([-cellRadius/2, cellRadius/2])
plt.yticks(np.arange(-cellRadius/2, cellRadius/2+5, corrDist))
plt.xticks(np.arange(-cellRadius/2, cellRadius/2+5, corrDist))
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[losState],posY[losState],s=20,c=macroZSD_lslog[losState], marker='o', cmap='RdYlBu_r', label='LOS')
sc = plt.scatter(posX[np.logical_not(losState)],posY[np.logical_not(losState)],s=20,c=macroZSD_lslog[np.logical_not(losState)], marker='x', cmap='RdYlBu_r', label='NLOS')
plt.colorbar(label="$\sigma_{logZSD}$ (º)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()

fig = plt.figure(6)
ax = plt.gca()
plt.xlim([-cellRadius/2, cellRadius/2])
plt.ylim([-cellRadius/2, cellRadius/2])
plt.yticks(np.arange(-cellRadius/2, cellRadius/2+5, corrDist))
plt.xticks(np.arange(-cellRadius/2, cellRadius/2+5, corrDist))
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[losState],posY[losState],s=20,c=macroZSA[losState], marker='o', cmap='RdYlBu_r', label='LOS')
sc = plt.scatter(posX[np.logical_not(losState)],posY[np.logical_not(losState)],s=20,c=macroZSA[np.logical_not(losState)], marker='x', cmap='RdYlBu_r', label='NLOS')
plt.colorbar(label="ZSA (º)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()