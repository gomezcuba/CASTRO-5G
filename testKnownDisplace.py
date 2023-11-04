#!/usr/bin/python

import threeGPPMultipathGenerator as pg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model = pg.ThreeGPPMultipathChannelModel(scenario="UMi")

txPos=(0,0,10)
rxPos=(40,0,10)
vLOS=np.array(rxPos)-np.array(txPos)
P= np.sqrt([.5,.3,.1,.2])
ZOD = [90,45,60,30]
ZOA = [90,180-45,180-60,180-30]
AOD = [0,0,0,0]
AOA = [0,0,0,0]
l0=np.linalg.norm(vLOS)
li=l0/np.sin(np.array(ZOD)*np.pi/180)
tau = (li-l0)/3e8

clusters = pd.DataFrame(
    columns = ["tau","powC","AOA","AOD","ZOA","ZOD"],
    data = np.array([tau,P,AOA,AOD,ZOA,ZOD]).T
    )
print(clusters)

deltaRxPos = (4,0,0)
newRxPos = np.array(rxPos)+np.array(deltaRxPos)

clustersD = model.displaceMultipathChannel(clusters,(0,0,0),deltaRxPos,deltaRxPos,vLOS)
print(clustersD)

aod = np.array(AOD)*np.pi/180
aoa = np.array(AOA)*np.pi/180
zod = np.array(ZOD)*np.pi/180
zoa = np.array(ZOA)*np.pi/180
rDi = np.column_stack([np.cos(aod)*np.sin(zod),np.sin(aod)*np.sin(zod),np.cos(zod)])
rAi = np.column_stack([np.cos(aoa)*np.sin(zoa),np.sin(aoa)*np.sin(zoa),np.cos(zoa)])
#OLLO QUE NO PDF DO ESTANDARD 7.1-13 E 14 ESTAN EN ORDEN ZENIT-AZIMUT EN VEZ DE AZIMUT-ZENIT
zDi = np.column_stack([np.cos(aod)*np.cos(zod),np.sin(aod)*np.cos(zod),-np.sin(zod)])# eq 7.1-13
zAi = np.column_stack([np.cos(aoa)*np.cos(zoa),np.sin(aoa)*np.cos(zoa),-np.sin(zoa)])# eq 7.1-13
aDi = np.column_stack([-np.sin(aod),np.cos(aod),np.zeros_like(aod)])# eq 7.1-14 
aAi = np.column_stack([-np.sin(aoa),np.cos(aoa),np.zeros_like(aoa)])# eq 7.1-14

#o canal "antiguo" corresponde con k=0 en 7.6-9
tau_tilde_previo = tau + 0 + l0/3e8
# e o canal "desexado" corresponde con k=1
approxTau_tilde = tau_tilde_previo - (rAi@np.array(deltaRxPos) + rDi@np.zeros(3))/3e8
approxTau = approxTau_tilde - np.min(approxTau_tilde)
print(approxTau_tilde - np.min(tau_tilde_previo))#facemos trampa e usamos o min de normalizacion anterior para verificar que os retardos crecen ao alonxarnos
#solo caso LOS
vnr = np.array(deltaRxPos)
vnt = -vnr
approxAOD = 180/np.pi*(aod + (aDi@vnr)/(l0 + tau*3e8)/np.sin(ZOD))
approxAOA = 180/np.pi*(aoa + (aAi@vnt)/(l0 + tau*3e8)/np.sin(ZOD))
approxZOD = 180/np.pi*(zod + (zDi@vnr)/(l0 + tau*3e8))
approxZOA = 180/np.pi*(zoa + (zAi@vnt)/(l0 + tau*3e8))

approxClusters = pd.DataFrame(
    columns = ["tau","powC","AOA","AOD","ZOA","ZOD"],
    data = np.array([approxTau,P,approxAOA,approxAOD,approxZOA,approxZOD]).T
    )
print(approxClusters)

print("O verdadeiro cambio dos angulos por xeometría é: ",np.arctan2(22,20*np.tan(np.pi/2-zod))*180/np.pi)
print("A aproximacion aditiva do 3GPP se non tivera unha aproximacion lineal da arcotangente seria: ",(zod + np.arctan2(zDi@vnr,l0 + tau*3e8))*180/np.pi)