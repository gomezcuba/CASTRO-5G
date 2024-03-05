#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as pg
plt.close('all')

txPos=np.array((0,0,10))
rxPos=np.array((-10,3,1.5))
depPos=np.array((3,-4,-3))
refPos=depPos+txPos

vLOS=rxPos-txPos
d2D = np.linalg.norm(vLOS[0:2])
d3D = np.linalg.norm(vLOS)

losAOD =np.mod( np.arctan2( vLOS[1] , vLOS[0] ) ,2*np.pi)
losZOD =np.mod(np.pi/2- np.arctan2( vLOS[2] , d2D ) ,2*np.pi)

difPos=refPos-rxPos
AOD=np.mod( np.arctan2( depPos[1] , depPos[0] ),2*np.pi)
AOA=np.mod( np.arctan2( difPos[1] , difPos[0] ),2*np.pi)
ZOD=np.mod(np.pi/2- np.arctan2( depPos[2] , np.linalg.norm(depPos[0:2]) ),2*np.pi)
ZOA=np.mod(np.pi/2- np.arctan2( difPos[2] , np.linalg.norm(difPos[0:2]) ),2*np.pi)

# 2D case
tau02D = d2D / 3e8
li2D = np.linalg.norm(depPos[0:2])+np.linalg.norm(difPos[0:2])
TDOA2D=li2D/3e8 - tau02D

# 3D case
tau0 = d3D / 3e8
li = np.linalg.norm(depPos)+np.linalg.norm(difPos)
TDOA=li/3e8 - tau0

###############################################################################
# OLD ALGORITHM FOR 2D ONLY
###############################################################################
dAOD = (AOD-losAOD)

cosdAOD = np.cos(dAOD)
sindAOD = np.sin(dAOD)
nu = li2D/d2D #ollo li/d2D = (tau0+tau)/tau0

# # Resolvemos:
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

print("Diference sols old 2D AOA (º):\n %s"%( ( np.mod(np.pi+losAOD-sols,2*np.pi)-AOA )*180/np.pi ))

#Ubicacion dos rebotes 
x=(vLOS[1]-vLOS[0]*np.tan(losAOD+np.pi-sols))/(np.tan( AOD )-np.tan(losAOD+np.pi-sols))
y=x*np.tan( AOD )
solPos=np.hstack([x,y])

print("Diference sols old 2D posicion (m):\n %s"%( np.linalg.norm(refPos[0:-1] - solPos,axis=1,keepdims=True) ))

###############################################################################
# IMPROVED ALGORITHM WITH 3d SUPPORT
###############################################################################
#2D case
ui2D=np.array([np.cos(AOD),np.sin(AOD)])
eta2D=.5*(li2D**2-d2D**2)/(li2D-ui2D[None,:]@vLOS[0:2,None])
solPos2D=(eta2D*ui2D)[0,:]
print("Diference sols new 2D posicion (m):\n %s"%( np.linalg.norm(refPos[0:-1] - solPos2D,axis=0,keepdims=True) ))

dOA=solPos2D-vLOS[0:2]
solAOA2D=np.arctan2(dOA[1],dOA[0])
print("Diference sols new 2D AOA (º):\n %s"%( ( np.mod(solAOA2D,2*np.pi)-AOA )*180/np.pi ))

#3D case
ui=np.array([np.cos(AOD)*np.sin(ZOD),np.sin(AOD)*np.sin(ZOD),np.cos(ZOD)])
eta=.5*(li**2-d3D**2)/(li-ui[None,:]@vLOS[:,None])
solPos3D=(eta*ui)[0,:] + txPos
print("Diference sols new 3D posicion (m):\n %s"%( np.linalg.norm(refPos - solPos3D,axis=0) ))

dOA=solPos3D-rxPos
solAOA3D=np.arctan2(dOA[1],dOA[0])
print("Diference sols new 3D AOA (º): %s"%( ( np.mod(solAOA3D,2*np.pi)-AOA )*180/np.pi ))
solZOA3D=np.pi/2-np.arctan2(dOA[2],np.linalg.norm(dOA[0:2]))
print("Diference sols new 3D ZOA (º): %s"%( ( np.mod(solZOA3D,2*np.pi)-ZOA )*180/np.pi ))

#test of the implemented libraries, in 2D and 3D
model=pg.ThreeGPPMultipathChannelModel()
libAoA2D,libPos2D = model.fitAOA(txPos,rxPos,TDOA2D,AOD)
print("Diference sols lib 2D posicion (m): %s"%( np.linalg.norm(refPos[0:2] - libPos2D[:,0],axis=0) ))
print("Diference sols lib 2D AOA (º): %s"%( ( np.mod(libAoA2D[0],2*np.pi)-AOA )*180/np.pi ))
libAoA3D,lipZoA3D,libPos3D = model.fitAOA(txPos,rxPos,TDOA,AOD,ZOD)
print("Diference sols lib 3D posicion (m): %s"%( np.linalg.norm(refPos - libPos3D[:,0],axis=0) ))
print("Diference sols lib 3D AOA (º): %s"%( ( np.mod(libAoA3D[0],2*np.pi)-AOA )*180/np.pi ))
print("Diference sols lib 3D ZOA (º): %s"%( ( np.mod(lipZoA3D[0],2*np.pi)-ZOA )*180/np.pi ))

libAoD2D,libPos2D = model.fitAOD(txPos,rxPos,TDOA2D,AOA)
print("Diference sols lib 2D posicion (m): %s"%( np.linalg.norm(refPos[0:2] - libPos2D[:,0],axis=0) ))
print("Diference sols lib 2D AOD (º): %s"%( ( np.mod(libAoD2D[0],2*np.pi)-AOD )*180/np.pi ))
libAoD3D,lipZoD3D,libPos3D = model.fitAOD(txPos,rxPos,TDOA,AOA,ZOA)
print("Diference sols lib 3D posicion (m): %s"%( np.linalg.norm(refPos - libPos3D[:,0],axis=0) ))
print("Diference sols lib 3D AOD (º): %s"%( ( np.mod(libAoD3D[0],2*np.pi)-AOD )*180/np.pi ))
print("Diference sols lib 3D ZOD (º): %s"%( ( np.mod(lipZoD3D[0],2*np.pi)-ZOD )*180/np.pi ))

libTDoA2D,libPos2D,valid = model.fitTDOA(txPos,rxPos,AOA,AOD)
print("Diference sols lib 2D posicion (m): %s"%( np.linalg.norm(refPos[0:2] - libPos2D[:,0],axis=0) ))
print("Diference sols lib 2D TDOA (ns): %s"%( (libTDoA2D[0]-TDOA2D)*1e9 ) )
libTDoA3D,libPos3D,valid = model.fitTDOA(txPos,rxPos,AOA,AOD,ZOA,ZOD)
print("Diference sols lib 3D posicion (m): %s"%( np.linalg.norm(refPos - libPos3D[:,0],axis=0) ))
print("Diference sols lib 3D TDOA (ns): %s"%( (libTDoA3D[0]-TDOA)*1e9 ) )