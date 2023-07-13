# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import threeGPPMultipathGenerator as mpg
import multipathChannel as mc
from matplotlib import cm
import pandas as pd

txPos = (0,0,10)
rxPos=(55,0,1.5)
tau = np.array([0.00000000e+00, 75.52000e-08, 1.87257376e-07, 1.87257376e-07,1.92388075e-07,1.87257376e-07, 1.92388075e-08,1.1088075e-07,8.7257376e-08, 1.92388075e-08,14.1088075e-07])
aoa = np.array([  0.        ,135,170.0232,20,60.987,80,240.654,135,272.32,333.31,2.54])


vLOS = np.array(rxPos) - np.array(txPos)
l0 = np.linalg.norm(vLOS[0:-1])
li = l0+tau*3e8
aoaR = aoa*(np.pi/180.0)
losAOD =(np.mod(np.arctan(vLOS[1]/vLOS[0])*+np.pi*(vLOS[0]<0),2*np.pi))
aoaAux = np.mod(np.pi-aoaR-losAOD,2*np.pi)
cosdAOA = np.cos(aoaAux)
sindAOA = np.sin(aoaAux)
nu = li/l0

A=nu**2+1-2*cosdAOA*nu
B=2*sindAOA*(1-nu*cosdAOA)
C=(sindAOA**2)*(1-nu**2)

sol1= ( -B - np.sqrt(B**2- 4*A*C ))/(2*A)
sol2= ( -B + np.sqrt(B**2- 4*A*C ))/(2*A)
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

