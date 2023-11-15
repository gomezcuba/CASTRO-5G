# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:28:39 2022

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as pg

plt.close('all')
fig_ctr=0


fig_ctr+=1
fig = plt.figure(fig_ctr)

model = pg.ThreeGPPMultipathChannelModel(scenario="UMi")

Npoint = 101
distance=np.arange(0,200,10)

LOSprobabilityUMi = model.tableFunLOSprob["UMi"](distance,1.5)
LOSprobabilityUMa = model.tableFunLOSprob["UMa"](distance,1.5)
LOSprobabilityRMa = model.tableFunLOSprob["RMa"](distance,1.5)
LOSprobabilityOpen = model.tableFunLOSprob["InH-Office-Open"](distance,1.5)
LOSprobabilityMixed = model.tableFunLOSprob["InH-Office-Mixed"](distance,1.5)

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

fig_ctr+=1
fig = plt.figure(fig_ctr)

distance=np.logspace(0,3+np.log10(5),31)
hut = 1.5
hbs = 10
d3D=np.sqrt(distance**2 + (hbs-hut)**2)
pathlossUMiLOS = model.scenarioPlossUMiLOS(d3D,distance)
pathLossUMiNLOS = model.scenarioPlossUMiNLOS(d3D,distance)
pathlossUMaLOS = model.scenarioPlossUMaLOS(d3D,distance)
pathLossUMaNLOS = model.scenarioPlossUMaNLOS(d3D,distance)
pathlossRMaLOS = model.scenarioPlossRMaLOS(d3D,distance)
pathLossRMaNLOS = model.scenarioPlossRMaNLOS(d3D,distance)
pathlossInHLOS = model.scenarioPlossInLOS(d3D,distance)
pathLossInHNLOS = model.scenarioPlossInNLOS(d3D,distance)
plt.semilogx(distance,pathlossUMiLOS, color='tab:red', linestyle = 'dashed' , label = 'UMi LOS')
plt.semilogx(distance,pathLossUMiNLOS, color='tab:red', linestyle = 'solid' , label = 'UMi NLOS')
plt.semilogx(distance,pathlossUMaLOS, color='tab:blue', linestyle = 'dashed' , label = 'UMa LOS')
plt.semilogx(distance,pathLossUMaNLOS, color='tab:blue', linestyle = 'solid' , label = 'UMa NLOS')
plt.semilogx(distance,pathlossRMaLOS, color='tab:orange', linestyle = 'dashed' , label = 'RMa LOS')
plt.semilogx(distance,pathLossRMaNLOS, color='tab:orange', linestyle = 'solid' , label = 'RMa NLOS')
plt.semilogx(distance,pathlossInHLOS, color='tab:green', linestyle = 'dashed' , label = 'InH LOS')
plt.semilogx(distance,pathLossInHNLOS, color='tab:green', linestyle = 'solid' , label = 'InH NLOS')
plt.legend()
plt.grid(axis='both', color='gray')
plt.xlabel("Distance (m)")
plt.ylabel("Path Loss (dB)")
plt.grid(axis='both', color='gray')
#-------------------------PLOTEO MACRO-------------------------------------
txPos = np.array((0,0,10))
Nusers = 300
numberMacrosCell = 9
cellRadius = numberMacrosCell*corrDist#m

contadorY = []
contadorX = []
key=[]

posX = np.random.uniform(-cellRadius/2, cellRadius/2, size=(Nusers))
posY = np.random.uniform(-cellRadius/2, cellRadius/2, size=(Nusers))
hut = 1.5
posZ = hut*np.ones(Nusers)
users=np.vstack([posX,posY,posZ]).T
macroIndex = (users[:,0:-1]-txPos[0:-1]+corrDist/2)//corrDist

vals,counts = np.unique(macroIndex,axis=0,return_counts=True)
vals=vals+numberMacrosCell//2
mm = np.zeros((numberMacrosCell,numberMacrosCell))
mm[vals[:,0].astype(int),vals[:,1].astype(int)]=counts


fig_ctr+=1
fig = plt.figure(fig_ctr)
X,Y=np.meshgrid(np.arange(0, cellRadius+5, corrDist)-cellRadius/2,np.arange(0, cellRadius+5, corrDist)-cellRadius/2)
plt.pcolor(X,Y,mm, cmap='RdYlBu_r',label=None)
plt.plot(0,0,'^k',label='BS')
plt.colorbar(label="Number of users", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()
plt.show()

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



fig_ctr+=1
fig = plt.figure(fig_ctr)
Xd,Yd=np.meshgrid(np.arange(0, cellRadius, 1.0)-cellRadius/2,np.arange(0, cellRadius, 1.0)-cellRadius/2)
losPgrid = model.scenarioLosProb(np.sqrt(Xd**2+Yd**2),hut)
hiddenUlos = np.array([[model.get_LOSUnif_from_location((0,0,25),(x,y,1.5)) for x in np.arange(0, cellRadius, 1.0)-cellRadius/2] for y in np.arange(0, cellRadius, 1.0)-cellRadius/2 ])
losBgrid = losPgrid >= hiddenUlos
plt.pcolor(Xd,Yd,losBgrid)
hiddenPlosTxt = np.array([['%.2f'%model.get_LOSUnif_from_location((0,0,25),(x*corrDist,y*corrDist,1.5)) for x in np.arange(-(numberMacrosCell//2),numberMacrosCell//2+1)] for y in np.arange(-(numberMacrosCell//2), numberMacrosCell//2+1) ])

for x in np.arange(-(numberMacrosCell//2),numberMacrosCell//2+1):
    for y in np.arange(-(numberMacrosCell//2), numberMacrosCell//2+1):
        U=model.get_LOSUnif_from_location((0,0,25),(x*corrDist,y*corrDist,1.5))
        P=model.scenarioLosProb(np.sqrt((x*corrDist)**2+(y*corrDist)**2),hut)
        S = '<=' if U<=P else '>'
        plt.text(x*corrDist-corrDist/2,y*corrDist-corrDist/2,'%.2f %s %.2f'%(U,S,P))
plt.colorbar(label="LOS areas", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.title('hiden LOS Uniform variable per square< pLos at discanceper')

    
fig_ctr+=1
fig = plt.figure(fig_ctr)
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

fig_ctr+=1
fig = plt.figure(fig_ctr)
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


fig_ctr+=1
fig = plt.figure(fig_ctr)
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


fig_ctr+=1
fig = plt.figure(fig_ctr)
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

fig_ctr+=1
fig = plt.figure(fig_ctr)
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