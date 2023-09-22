#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import os

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as pg
plt.close('all')


txPos=np.array((0,0,10))
rxPos=np.array((-10,3,1.5))

vLOS=rxPos-txPos
l0 = np.linalg.norm(vLOS[0:-1])
tau0 = l0 / 3e8
losAOD =np.mod( np.arctan( vLOS[1] / vLOS[0] )+np.pi*(vLOS[0]<0),2*np.pi)


refPos=np.array((-5,1.5,0))+txPos
AOD=np.mod( np.arctan( refPos[1] / refPos[0] )+np.pi*(refPos[0]<0),2*np.pi)

difPos=refPos-rxPos
AOA=np.mod( np.arctan( difPos[1] / difPos[0] )+np.pi*(difPos[0]<0),2*np.pi)

tau=(np.linalg.norm(refPos[0:-1])+np.linalg.norm(difPos[0:-1]))/3e8 - tau0

li = l0 + tau * 3e8
dAOD = (AOD-losAOD)

cosdAOD = np.cos(dAOD)
sindAOD = np.sin(dAOD)
nu = li/l0 #ollo li/l0 = (tau0+tau)/tau0

# Resolvemos:
A=nu**2+1-2*cosdAOD*nu
B=2*sindAOD*(1-nu*cosdAOD)#OLLO AQUI CAMBIOU O SIGNO
C=(sindAOD**2)*(1-nu**2)
# sol1= ( -B - np.sqrt(B**2- 4*A*C ))/(2*A)
sol1= -sindAOD # xust.matematica overleaf
# sol2= ( -B + np.sqrt(B**2- 4*A*C ))/(2*A)
sol2= sindAOD*(nu**2-1) /  ( nu**2+1-2*cosdAOD*nu )

#Posibles solucions:
sols = np.zeros((4,AOD.size)) 
sols[0,:] = np.arcsin(sol1)
sols[1,:] = np.arcsin(sol2)
sols[2,:] = np.pi - np.arcsin(sol1)
sols[3,:] = np.pi - np.arcsin(sol2)

print("Diferencia sols AOA (º):\n %s"%( ( np.mod(np.pi+losAOD-sols,2*np.pi)-AOA )*180/np.pi ))

#Ubicacion dos rebotes 
x=(vLOS[1]-vLOS[0]*np.tan(losAOD+np.pi-sols))/(np.tan( AOD )-np.tan(losAOD+np.pi-sols))
y=x*np.tan( AOD )
solPos=np.hstack([x,y])

print("Diferencia sols posicion (m):\n %s"%( np.linalg.norm(refPos[0:-1] - solPos,axis=1,keepdims=True) ))

###############################################################################
# IMPROVED ALGORITHM WITH 3d SUPPORT
###############################################################################
d0=vLOS
#only for the 2D case
d0[2]=0
ZOD=0
#____________
ui=np.array([np.cos(AOD)*np.cos(ZOD),np.sin(AOD)*np.cos(ZOD),np.sin(ZOD)])
eta=.5*(li**2-l0**2)/(li-ui[None,:]@vLOS[:,None])
solPos2=(eta*ui)[0,:]
print("Diferencia sols2 posicion (m):\n %s"%( np.linalg.norm(refPos[0:-1] - solPos2[0:-1],axis=0,keepdims=True) ))

dOA=solPos2-d0
solAOA2=np.arctan2(dOA[1],dOA[0])
print("Diferencia sols2 AOA (º):\n %s"%( ( np.mod(solAOA2,2*np.pi)-AOA )*180/np.pi ))
