# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:28:39 2022

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as pg

plt.close('all')
fig_ctr=0


fig_ctr+=1
fig = plt.figure(fig_ctr)

modelUMi = pg.ThreeGPPMultipathChannelModel(scenario="UMi")
modelUMa = pg.ThreeGPPMultipathChannelModel(scenario="UMa")
modelRMa = pg.ThreeGPPMultipathChannelModel(scenario="RMa")
modelOpen = pg.ThreeGPPMultipathChannelModel(scenario="InH-Office-Open")
modelMixed = pg.ThreeGPPMultipathChannelModel(scenario="InH-Office-Mixed")

Npoint = 101
distance=np.linspace(0,200,Npoint)

LOSprobabilityUMi = modelUMi.scenarioLosProb(distance,1.5)
LOSprobabilityUMa = modelUMa.scenarioLosProb(distance,1.5)
LOSprobabilityRMa = modelRMa.scenarioLosProb(distance,1.5)
LOSprobabilityOpen = modelOpen.scenarioLosProb(distance,1.5)
LOSprobabilityMixed = modelMixed.scenarioLosProb(distance,1.5)

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

modelSL = pg.ThreeGPPMultipathChannelModel(scenario="InF-SL")
modelDL = pg.ThreeGPPMultipathChannelModel(scenario="InF-DL")
modelSH = pg.ThreeGPPMultipathChannelModel(scenario="InF-SH")
modelDH = pg.ThreeGPPMultipathChannelModel(scenario="InF-DH")
modelHH = pg.ThreeGPPMultipathChannelModel(scenario="InF-HH")

Npoint = 101
distance=np.linspace(0,80,Npoint)

LOSprobabilitySL = modelSL.scenarioLosProb(distance,1.5)
LOSprobabilityDL = modelDL.scenarioLosProb(distance,1.5)
LOSprobabilitySH = modelSH.scenarioLosProb(distance,1.5,8)#hbs mandatory
LOSprobabilityDH = modelDH.scenarioLosProb(distance,1.5,8)
LOSprobabilityHH = modelHH.scenarioLosProb(distance,1.5)

plt.plot(distance, LOSprobabilitySL, color = 'tab:red', linestyle = 'dashed' , label = 'SL')
plt.plot(distance, LOSprobabilityDL, color = 'tab:blue', linestyle = 'dashed' , label = 'DL')
plt.plot(distance, LOSprobabilitySH, color = 'tab:orange', linestyle = 'dashed' , label = 'SH' )
plt.plot(distance, LOSprobabilityDH, color = 'tab:green', linestyle = 'dashed' , label = 'DH' )
plt.plot(distance, LOSprobabilityHH*np.ones_like(distance), color = 'tab:purple', linestyle = 'dashed' , label = 'HH' )

plt.legend()
plt.grid(axis='both', color='gray')
plt.xlabel('Distance (m)')
plt.ylabel('LOS Probability InF')
plt.show()

fig_ctr+=1
fig = plt.figure(fig_ctr)

distance=np.logspace(0,3+np.log10(5),31)
hut = 1.5
hbs = 10
d3D=np.sqrt(distance**2 + (hbs-hut)**2)
pathlossUMiLOS = modelUMi.scenarioPlossUMiLOS(d3D,distance)
pathLossUMiNLOS = modelUMi.scenarioPlossUMiNLOS(d3D,distance)
pathlossUMaLOS = modelUMa.scenarioPlossUMaLOS(d3D,distance)
pathLossUMaNLOS = modelUMa.scenarioPlossUMaNLOS(d3D,distance)
pathlossRMaLOS = modelRMa.scenarioPlossRMaLOS(d3D,distance)
pathLossRMaNLOS = modelRMa.scenarioPlossRMaNLOS(d3D,distance)
pathlossInHLOS = modelOpen.scenarioPlossInHLOS(d3D,distance)
pathLossInHNLOS = modelOpen.scenarioPlossInHNLOS(d3D,distance)
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


fig_ctr+=1
fig = plt.figure(fig_ctr)

distance=np.logspace(0,2+np.log10(5),31)
hut = 1.5
hbs = 10
d3D=np.sqrt(distance**2 + (hbs-hut)**2)
pathlossInFLOS = modelHH.scenarioPlossInFLOS(d3D,distance)
pathLossSLNLOS = modelSL.scenarioPlossInFSLNLOS(d3D,distance)
pathLossDLNLOS = modelDL.scenarioPlossInFDLNLOS(d3D,distance)
pathLossSHNLOS = modelSH.scenarioPlossInFSHNLOS(d3D,distance)
pathlossDHNLOS = modelDH.scenarioPlossInFDHNLOS(d3D,distance)
plt.semilogx(distance,pathlossInFLOS, color='tab:red', linestyle = 'dashed' , label = 'InF LOS all')
plt.semilogx(distance,pathLossSLNLOS, color='tab:red', linestyle = 'solid' , label = 'INF-SL NLOS')
plt.semilogx(distance,pathLossDLNLOS, color='tab:blue', linestyle = 'solid' , label = 'INF-DL NLOS')
plt.semilogx(distance,pathLossSHNLOS, color='tab:orange', linestyle = 'solid' , label = 'INF-SH LOS')
plt.semilogx(distance,pathlossDHNLOS, color='tab:green', linestyle = 'solid' , label = 'INF-DH NLOS')
plt.legend()
plt.grid(axis='both', color='gray')
plt.xlabel("Distance (m)")
plt.ylabel("Path Loss (dB)")
plt.grid(axis='both', color='gray')
#-------------------------PLOTEO MACRO-------------------------------------
txPos = np.array((0,0,10))
Nusers = 300
cellDiameter=150#m
corrLOS=modelUMi.scenarioParams.loc["LOS"].corrLOS
corrStatLOS=modelUMi.scenarioParams.loc["LOS"].corrStatistics
corrStatNLOS=modelUMi.scenarioParams.loc["NLOS"].corrStatistics
numberBinsLOS = np.ceil(cellDiameter/corrLOS)
numberBinsStatLOS = np.ceil(cellDiameter/corrStatLOS)
numberBinsStatNLOS = np.ceil(cellDiameter/corrStatNLOS)

contadorY = []
contadorX = []
key=[]

posX = np.random.uniform(-cellDiameter/2, cellDiameter/2, size=(Nusers))
posY = np.random.uniform(-cellDiameter/2, cellDiameter/2, size=(Nusers))
hut = 1.5
posZ = hut*np.ones(Nusers)
users=np.vstack([posX,posY,posZ]).T

losState = np.zeros(Nusers,dtype=bool) 
macroDS = np.zeros(Nusers) 
macroASD = np.zeros(Nusers)
macroASA = np.zeros(Nusers)
macroZSD_lslog = np.zeros(Nusers)
macroZSA = np.zeros(Nusers)
for i in range(Nusers):
    plinfo,macro,clusters,subpaths = modelUMi.create_channel(txPos,users[i])
    losState[i],PLdBmu,shadowing = plinfo
    macroDS[i]= macro.ds
    macroASD[i]= macro.asd
    macroASA[i]= macro.asa
    macroZSD_lslog[i]= macro.zsd_lslog
    macroZSA[i]= macro.zsa



fig_ctr+=1
fig = plt.figure(fig_ctr)
Xd,Yd=np.meshgrid(np.arange(0, cellDiameter, 1.0)-cellDiameter/2,np.arange(0, cellDiameter, 1.0)-cellDiameter/2)
losPgrid = modelUMi.scenarioLosProb(np.sqrt(Xd**2+Yd**2),hut)
hiddenUlos = np.array([[modelUMi.get_LOSUnif_from_location((0,0,25),(x,y,1.5)) for x in np.arange(0, cellDiameter, 1.0)-cellDiameter/2] for y in np.arange(0, cellDiameter, 1.0)-cellDiameter/2 ])
losBgrid = losPgrid >= hiddenUlos
plt.pcolor(Xd,Yd,losBgrid,cmap="Pastel1")
# hiddenPlosTxt = np.array([['%.2f'%model.get_LOSUnif_from_location((0,0,25),(x*corrLOS,y*corrLOS,1.5)) for x in np.arange(-(numberBinsLOS//2),numberBinsLOS//2+1)] for y in np.arange(-(numberBinsLOS//2), numberBinsLOS//2+1) ])

for x in np.arange(-(numberBinsLOS//2),numberBinsLOS//2+1):
    for y in np.arange(-(numberBinsLOS//2), numberBinsLOS//2+1):
        U=modelUMi.get_LOSUnif_from_location((0,0,25),(x*corrLOS,y*corrLOS,1.5))
        P=modelUMi.scenarioLosProb(np.sqrt((x*corrLOS)**2+(y*corrLOS)**2),hut)
        S = '<=' if U<=P else '>'
        plt.text(x*corrLOS,y*corrLOS,'%.2f $%s$ %.2f'%(U,S,P),horizontalalignment='center',verticalalignment='center')
        
proxy = [plt.Rectangle((0,0),1,1,fc = cm.Pastel1(x)) 
    for x in [0,255]]

plt.legend(proxy, ["NLOS", "LOS"])
        
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.title('hiden LOS Uniform < pLos at discance')

    
fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = plt.gca()
plt.xlim([-cellDiameter/2, cellDiameter/2])
plt.ylim([-cellDiameter/2, cellDiameter/2])
plt.yticks(np.arange(-numberBinsStatLOS/2, 1+numberBinsStatLOS/2, 1)*corrStatLOS)
plt.xticks(np.arange(-numberBinsStatLOS/2, 1+numberBinsStatLOS/2, 1)*corrStatLOS)
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[losState],posY[losState],s=20,c=1e9*macroDS[losState], marker='o', cmap='RdYlBu_r',label='LOS')
plt.colorbar(label="LOS DS (ns)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()


fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = plt.gca()
plt.xlim([-cellDiameter/2, cellDiameter/2])
plt.ylim([-cellDiameter/2, cellDiameter/2])
plt.yticks(np.arange(-numberBinsStatNLOS/2, 1+numberBinsStatNLOS/2, 1)*corrStatNLOS)
plt.xticks(np.arange(-numberBinsStatNLOS/2, 1+numberBinsStatNLOS/2, 1)*corrStatNLOS)
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[np.logical_not(losState)],posY[np.logical_not(losState)],s=20,c=1e9*macroDS[np.logical_not(losState)],marker='x', cmap='RdYlBu_r',label='NLOS')
plt.colorbar(label="NLOS DS (ns)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()

fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = plt.gca()
plt.xlim([-cellDiameter/2, cellDiameter/2])
plt.ylim([-cellDiameter/2, cellDiameter/2])
plt.yticks(np.arange(-numberBinsStatLOS/2, 1+numberBinsStatLOS/2, 1)*corrStatLOS)
plt.xticks(np.arange(-numberBinsStatLOS/2, 1+numberBinsStatLOS/2, 1)*corrStatLOS)
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[losState],posY[losState],s=20,c=1e9*macroASD[losState], marker='o', cmap='RdYlBu_r',label='LOS')
plt.colorbar(label="LOS ASD (º)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()


fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = plt.gca()
plt.xlim([-cellDiameter/2, cellDiameter/2])
plt.ylim([-cellDiameter/2, cellDiameter/2])
plt.yticks(np.arange(-numberBinsStatNLOS/2, 1+numberBinsStatNLOS/2, 1)*corrStatNLOS)
plt.xticks(np.arange(-numberBinsStatNLOS/2, 1+numberBinsStatNLOS/2, 1)*corrStatNLOS)
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[np.logical_not(losState)],posY[np.logical_not(losState)],s=20,c=1e9*macroASD[np.logical_not(losState)],marker='x', cmap='RdYlBu_r',label='NLOS')
plt.colorbar(label="NLOS ASD (º)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()


fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = plt.gca()
plt.xlim([-cellDiameter/2, cellDiameter/2])
plt.ylim([-cellDiameter/2, cellDiameter/2])
plt.yticks(np.arange(-numberBinsStatLOS/2, 1+numberBinsStatLOS/2, 1)*corrStatLOS)
plt.xticks(np.arange(-numberBinsStatLOS/2, 1+numberBinsStatLOS/2, 1)*corrStatLOS)
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[losState],posY[losState],s=20,c=1e9*macroASA[losState], marker='o', cmap='RdYlBu_r',label='LOS')
plt.colorbar(label="LOS ASA (º)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()


fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = plt.gca()
plt.xlim([-cellDiameter/2, cellDiameter/2])
plt.ylim([-cellDiameter/2, cellDiameter/2])
plt.yticks(np.arange(-numberBinsStatNLOS/2, 1+numberBinsStatNLOS/2, 1)*corrStatNLOS)
plt.xticks(np.arange(-numberBinsStatNLOS/2, 1+numberBinsStatNLOS/2, 1)*corrStatNLOS)
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[np.logical_not(losState)],posY[np.logical_not(losState)],s=20,c=1e9*macroASA[np.logical_not(losState)],marker='x', cmap='RdYlBu_r',label='NLOS')
plt.colorbar(label="NLOS ASA (º)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()



fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = plt.gca()
plt.xlim([-cellDiameter/2, cellDiameter/2])
plt.ylim([-cellDiameter/2, cellDiameter/2])
plt.yticks(np.arange(-numberBinsStatLOS/2, 1+numberBinsStatLOS/2, 1)*corrStatLOS)
plt.xticks(np.arange(-numberBinsStatLOS/2, 1+numberBinsStatLOS/2, 1)*corrStatLOS)
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[losState],posY[losState],s=20,c=1e9*macroZSD_lslog[losState], marker='o', cmap='RdYlBu_r',label='LOS')
plt.colorbar(label="LOS $\sigma_{logZSD}$ (º)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()


fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = plt.gca()
plt.xlim([-cellDiameter/2, cellDiameter/2])
plt.ylim([-cellDiameter/2, cellDiameter/2])
plt.yticks(np.arange(-numberBinsStatNLOS/2, 1+numberBinsStatNLOS/2, 1)*corrStatNLOS)
plt.xticks(np.arange(-numberBinsStatNLOS/2, 1+numberBinsStatNLOS/2, 1)*corrStatNLOS)
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[np.logical_not(losState)],posY[np.logical_not(losState)],s=20,c=1e9*macroZSD_lslog[np.logical_not(losState)],marker='x', cmap='RdYlBu_r',label='NLOS')
plt.colorbar(label="NLOS $\sigma_{logZSD}$ (º)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()

fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = plt.gca()
plt.xlim([-cellDiameter/2, cellDiameter/2])
plt.ylim([-cellDiameter/2, cellDiameter/2])
plt.yticks(np.arange(-numberBinsStatLOS/2, 1+numberBinsStatLOS/2, 1)*corrStatLOS)
plt.xticks(np.arange(-numberBinsStatLOS/2, 1+numberBinsStatLOS/2, 1)*corrStatLOS)
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[losState],posY[losState],s=20,c=1e9*macroZSA[losState], marker='o', cmap='RdYlBu_r',label='LOS')
plt.colorbar(label="LOS ZSA (º)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()

fig_ctr+=1
fig = plt.figure(fig_ctr)
ax = plt.gca()
plt.xlim([-cellDiameter/2, cellDiameter/2])
plt.ylim([-cellDiameter/2, cellDiameter/2])
plt.yticks(np.arange(-numberBinsStatNLOS/2, 1+numberBinsStatNLOS/2, 1)*corrStatNLOS)
plt.xticks(np.arange(-numberBinsStatNLOS/2, 1+numberBinsStatNLOS/2, 1)*corrStatNLOS)
plt.grid(axis='both',color='red')
plt.plot(0,0,'^k',label='BS')
sc = plt.scatter(posX[np.logical_not(losState)],posY[np.logical_not(losState)],s=20,c=1e9*macroZSA[np.logical_not(losState)],marker='x', cmap='RdYlBu_r',label='NLOS')
plt.colorbar(label="NLOS ZSA (º)", orientation="vertical")
plt.xlabel('distance (m)')
plt.ylabel('distance (m)')
plt.legend()