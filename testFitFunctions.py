# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import threeGPPMultipathGenerator as mpg
import multipathChannel as mc
from matplotlib import cm

model = mpg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=False)
plinfo,macro,clusters,subpaths = model.create_channel(txPos,rxPos)
tau,powC,AOA,AOD,ZOA,ZOD = clusters.T.to_numpy()
los, PLfree, SF = plinfo
tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths.T.to_numpy()
AOA_spFix = model.fitAOA(txPos,rxPos,AOD_sp,tau_sp)


#%%
# O mesmo pero agora probamos as novas funcións:
# Datos iniciais - l0, tau0 e aod0 (losAOD)
clusters2 = clusters
subpaths2 = subpaths
# Datos iniciais - l0, tau0 e aod0 (losAOD)
vLOS = np.array(rxPos) - np.array(txPos)
l0 = np.linalg.norm(vLOS[0:-1])
tau0 = l0 / 3e8
losAOD =(np.mod( np.arctan( vLOS[1] / vLOS[0] )+np.pi*(vLOS[0]<0),2*np.pi))*(180.0/np.pi) # en graos

# Extraemos index. de clusters
#nClusters = clusters.shape[0]

#for i in range(0,nClusters -1):
    # de aquí sacamos aod e tau
    #TODO organizar para q procese ben valores do df
    #clusterValues = subpaths.loc()
aod = subpaths2['AOD'].astype(float)
tau = subpaths2['tau'].astype(float)
#TODO end

li = l0 + tau * 3e8
dAOD = (aod-losAOD)*(np.pi/180)

cosdAOD = np.cos(dAOD)
sindAOD = np.sin(dAOD)
nu = tau/tau0

# Resolvemos:
A=(nu-cosdAOD)**2+sindAOD**2
B=-2*nu*sindAOD*(nu-cosdAOD)
C=(sindAOD**2)*(nu**2-1)
sol1= ( -B + np.sqrt( B**2 - 4*A*C  ) )/( 2*A )
sol2= ( -B - np.sqrt( B**2 - 4*A*C  ) )/( 2*A )

#Posibles solucions:
sols = np.zeros((4,aod.size)) 
sols[0,:] = np.transpose(np.arcsin(sol1))
sols[1,:] = np.transpose(np.arcsin(sol2))
sols[2,:] = np.transpose(np.pi - np.arcsin(sol1))
sols[3,:] = np.transpose(np.pi - np.arcsin(sol2))

#Avaliamos consistencia e distancia:
dist = np.zeros((4,aod.size))
for i in range(0,3):
    numNu= sindAOD + np.sin(sols[i,:])
    denomNu= sindAOD*np.cos(sols[i,:]) + cosdAOD*np.sin(sols[i,:])
    dist[i]= (abs(numNu/denomNu)-nu)

distMod = np.sum(dist,axis=1)    
solIndx=np.argmin(distMod,0)
sol = sols[solIndx,range(li.size)]
# Norm., convertimos de novo a graos e achamos o aoaReal - non o aux.:
aoaDeg = np.mod(sol+np.pi,2*np.pi)
aoaDeg = aoaDeg*(180/np.pi)
subpaths2['AOA'] = aoaDeg

# Eliminamos valores de AOA dos backlobes
# Creo función aparte para poder chamala dende calquer lado