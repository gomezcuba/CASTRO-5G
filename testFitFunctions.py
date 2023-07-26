#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import os

from CASTRO5G import threeGPPMultipathGenerator as mpg
from CASTRO5G import multipathChannel as mc
plt.close('all')


txPos = (0,0,10)
rxPos = (25,45,1.5)
sce = "UMi"

modelA = mpg.ThreeGPPMultipathChannelModel(scenario = sce, bLargeBandwidthOption=True)
plinfoA,macroA,clustersA,subpathsA = modelA.create_channel(txPos,rxPos)

prob = np.array([0.4,0.4,0.2])

dataset = clustersA.copy()

totalSize = dataset.shape[0]
splitSizes = np.floor(prob * totalSize).astype(int)


indx = 0
dataset1= dataset.iloc[indx:indx+splitSizes[0]]       
dataset1= modelA.fitAOA(txPos,rxPos,dataset1.reset_index(drop=True))
indx += splitSizes[0]
dataset2= dataset.iloc[indx:indx+splitSizes[1]]       
dataset2= modelA.fitAOD(txPos,rxPos,dataset2.reset_index(drop=True))
indx += splitSizes[1]
dataset3= dataset.iloc[indx:totalSize]       
dataset3= modelA.fitDelay(txPos,rxPos,dataset3.reset_index(drop=True))
    
datasetFix = pd.concat([dataset1, dataset2, dataset3], axis=0)

#%%

txPos = (0,0,10)
rxPos=(-25,-25,1.5)
tau = np.array([0.00000000e+00, 75.52000e-08, 1.87257376e-07, 1.87257376e-07,1.92388075e-07,1.87257376e-07, 1.92388075e-08,1.1088075e-07,8.7257376e-08, 1.92388075e-08,14.1088075e-07])
aoa = np.array([  45.        ,45,170.0232,20,60.987,80,240.654,135,272.32,333.31,2.54])


vLOS = np.array(rxPos) - np.array(txPos)
l0 = np.linalg.norm(vLOS[0:-1])
li = l0+tau*3e8
losAOD =(np.mod(np.arctan(vLOS[1]/vLOS[0])+np.pi*(vLOS[0]<0),2*np.pi))
losAOA = np.mod(np.pi+losAOD,2*np.pi)
aoa[0] = losAOA*(180.0/np.pi) #necesario para consistencia do primeiro rebote

aoaR = aoa*(np.pi/180.0)
aoaAux = np.mod(-aoaR+losAOA*(vLOS[0]>0),2*np.pi)

cosdAOA = np.cos(aoaAux)
sindAOA = np.sin(aoaAux)
nu = li/l0

A=nu**2+1-2*cosdAOA*nu
B=2*sindAOA*(1-nu*cosdAOA)
C=(sindAOA**2)*(1-nu**2)

sol1= -sindAOA
sol2= sindAOA*(nu**2-1) /  ( nu**2+1-2*cosdAOA*nu )
sol2[(nu==1)&(cosdAOA==1)] = 0 #LOS path

#Posibles solucions:
sols = np.zeros((4,aoa.size)) 
# sols[0,:] = np.arcsin(sol1)
sols[0,:] = np.arcsin(sol1)
sols[1,:] = np.arcsin(sol2)
sols[2,:] = np.pi - np.arcsin(sol1)
sols[3,:] = np.pi - np.arcsin(sol2)

#Ubicacion dos rebotes 
x=(vLOS[1]-vLOS[0]*np.tan(losAOD+np.pi-aoaAux))/(np.tan(losAOD+sols)-np.tan(losAOD+np.pi-aoaAux))
x[1,(nu==1)&(cosdAOA==1)] = vLOS[0]/2
x[3,(nu==1)&(cosdAOA==1)] = vLOS[0]/2
y=x*np.tan(losAOD + sols) 

dist=np.sqrt(x**2+y**2)+np.sqrt((x-vLOS[0])**2+(y-vLOS[1])**2)
solIndx=np.argmin(np.abs(dist-li),axis=0)
aodAux =sols[solIndx,range(li.size)]
aodFix = np.mod(aodAux+losAOD,2*np.pi) * (180.0/np.pi)
aod = aodFix

xPathLoc = x[solIndx,range(li.size)]
yPathLoc = y[solIndx,range(li.size)]

liRX = np.sqrt((xPathLoc-rxPos[0])**2+(yPathLoc - rxPos[1])**2)

liTX = np.sqrt(xPathLoc**2+yPathLoc**2)


fig_ctr = 0

# ---
fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.title("AOD probas")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')
plt.plot(txPos[0],txPos[1],'^g',color='b',label='BS',linewidth = '4.5')
plt.plot(rxPos[0],rxPos[1],'^',color='r',label='UE', linewidth='4.5')
plt.plot([txPos[0],rxPos[0]],[txPos[1],rxPos[1]],'--',color='g',label='LOS')
plt.plot(xPathLoc,yPathLoc,'x',color='y',label='Rebotes')

for i in range(0,aod.size): 
    plt.plot([txPos[0],txPos[0]+liTX[i]*np.cos(aod[i]*np.pi/180.0)],[txPos[1],liTX[i]*np.sin(aod[i]*np.pi/180.0)],color = 'blue',linewidth = '0.5') 
    plt.plot([rxPos[0],rxPos[0]+liRX[i]*np.cos(aoaR[i])],[rxPos[1],rxPos[1]+liRX[i]*np.sin(aoaR[i])],color = 'red',linewidth = '0.5')
#%%
import numpy as np
import matplotlib.pyplot as plt
    

                
# #Cuadrantes positivos - Condición phi intervalo(aod,aod0+pi)
# if vLOS[0] > 0:
#     for n in range(aoa.size):
#         #Non cumple condicións
#         if aoa[n] < aod0 or aoa[n] > aoalim:
            
#             aoa[n] = np.random.uniform(aod0, aoalim)
        
# #Negativo - phi no intervalo(aod+pi,aod)
# else:
#     for n in range(aoa.size):
#         if aoa[n] > aod0 or aoa[n] < aoalim:
                
#             aoa[n] = np.random.uniform(aoalim, aod0)

rxPos = (45,-95)
txPos = (0,0)
vLOS = np.array(rxPos) - np.array(txPos)
aod = np.array([100,120,190,290,40,120,190,290])*(np.pi/180.0)
aoa = np.array([50,20,30,350,72,42,290,73])*(np.pi/180.0)
aoa0 = aoa.copy()
aod0 =(np.mod(np.arctan(vLOS[1]/vLOS[0])+np.pi*(vLOS[0]<0),2*np.pi))
aod0d = aod0*180.0/np.pi

aodmin = (aod>aod0)
rg = np.mod(aod+np.pi*(aod>np.pi)-aod0+np.pi*(aod>np.pi),2*np.pi)

aoalim = np.mod(aod0+np.pi,2*np.pi)
dif = np.mod(aoa-aod,2*np.pi)

if vLOS[0] > 0:
    for n in range(aoa.size):
        if (aod[n] < aoa[n] < aoalim) and aodmin[n]:
            aoa[n] = aoa[n]
        else:
            if(aodmin[n]):
                aoa[n] = np.random.uniform(aod[n],aoalim)+np.pi*(vLOS[1]<0)
            else:
                aoa[n] = np.random.uniform(aoalim, aod[n])+np.pi*(not aodmin[n])+np.pi*(vLOS[1]<0)
else:
    for n in range(aoa.size):
        if (aoalim < aoa[n] < aod[n]) and (not aodmin[n]):
            aoa[n] = aoa[n]
        else:
            if(aodmin[n]):
                aoa[n] = np.random.uniform(aoalim,aod[n])
            else:
                aoa[n] = np.mod(aoa[n]+np.pi,2*np.pi)+np.pi*(aodmin[n])+np.pi*(vLOS[1]<0)    
    


l0 = np.linalg.norm(vLOS[0:-1])
TA=np.tan(np.pi-aoa)
TD=np.tan(aod)
x=((rxPos[1]+rxPos[0]*TA)/(TD+TA))
y=x*TD
l=np.sqrt(x**2+y**2)+np.sqrt((x-rxPos[0])**2+(y-rxPos[1])**2)
l0=np.sqrt(rxPos[0]**2+rxPos[1]**2)
c=3e8
exdel=(l-l0)/c

liTX = np.sqrt(x**2+y**2)
liRX = np.sqrt((x-vLOS[0])**2+(y-vLOS[1])**2)

xloc = x[0:l.size]
yloc = y[0:l.size]
tau = (l-l0)/3e8

fig_ctr = 0

# ---
fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.title("AOD probas")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

plt.plot(txPos[0],txPos[1],'^g',label='BS',linewidth = '4.5')
plt.plot(rxPos[0],rxPos[1],'^r',label='UE', linewidth='4.5')
plt.plot([txPos[0],rxPos[0]],[txPos[1],rxPos[1]],'--',color='g',label='LOS')
plt.plot(xloc,yloc,'x',color='y',label='Rebotes')
for i in range(0,aod.size): 
    plt.plot([txPos[0],txPos[0]+liTX[i]*np.cos(aod[i])],[txPos[1],liTX[i]*np.sin(aod[i])],color = 'blue',linewidth = '0.5') 
    plt.plot([rxPos[0],rxPos[0]+liRX[i]*np.cos(aoa[i])],[rxPos[1],rxPos[1]+liRX[i]*np.sin(aoa[i])],color = 'red',linewidth = '0.5')
