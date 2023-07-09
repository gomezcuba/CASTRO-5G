# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import threeGPPMultipathGenerator as mpg
import multipathChannel as mc
from matplotlib import cm

txPos = (0,0,10)
rxPos=(25,25,1.5)
resAOD = np.array([45.        , 28.49586186, 28.3390465 , 46.02784762, 60.27996269,
       33.6888124 , 21.54977599, 54.88047962, 56.2608943 , 57.01498426,
       27.68095598, 19.54524614, 24.92657839, 17.92163775, 37.65870637,
       65.18886193, 56.71909545, 24.99859557, 67.55998501])
tau = np.array([0.00000000e+00, 6.93738407e-10, 2.84346353e-09, 4.75920658e-09,
       1.20598895e-08, 1.29800924e-08, 1.88437928e-08, 1.89868284e-08,
       2.38005438e-08, 2.49335502e-08, 2.79926058e-08, 3.09887604e-08,
       3.23993204e-08, 4.50983350e-08, 4.61027922e-08, 4.75980637e-08,
       5.92387878e-08, 6.12666568e-08, 1.04436567e-07])
aoa = np.array([225.        , 227.31841524, 234.30785054,  93.76093945,
       185.1111198 , 280.58421877, 264.26113426, 143.44714064,
       139.13586532, 140.45570102, 294.7546112 , 279.44815299,
       293.65051367, 292.40369109,   2.17617547, 138.31526304,
        99.13057907, 323.95597959, 111.01623302])


vLOS = np.array(rxPos) - np.array(txPos)
l0 = np.linalg.norm(vLOS[0:-1])
li = l0+tau*3e8
tau0 = l0 / 3e8
aoaR = aoa*(np.pi/180.0)
losAOD =(np.mod(np.arctan(vLOS[1]/vLOS[0])+np.pi*(vLOS[0]<0),2*np.pi)) # en radians
aoaAux = np.mod(losAOD+np.pi-aoaR,2*np.pi)
cosdAOA = np.cos(aoaR)
sindAOA = np.sin(aoaR)
nu = li/l0

A=nu**2+1-2*cosdAOA*nu
B=2*sindAOA*(1-nu*cosdAOA)
C=(sindAOA**2)*(1-nu**2)

sol1= -sindAOA
sol2= sindAOA*(nu**2-1) /  ( nu**2+1-2*cosdAOA*nu )
sol2[(nu==1)&(cosdAOA==1)] = 0 #LOS path

#Posibles solucions:
sols = np.zeros((4,aoa.size)) 
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
aod = np.mod(aodAux+losAOD+np.pi*(vLOS[0]<0),2*np.pi) * (180.0/np.pi)

aodiff = np.abs(aod-resAOD)

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
    plt.plot([txPos[0],txPos[0]+liTX[i]*np.cos(aod[i]*np.pi/180.0)],[txPos[1],liTX[i]*np.sin(aod[i]*np.pi/180.0)],'k',color = 'blue',linewidth = '0.5') 
    plt.plot([rxPos[0],rxPos[0]+liRX[i]*np.cos(aoaR[i])],[rxPos[1],rxPos[1]+liRX[i]*np.sin(aoaR[i])],color = 'red',linewidth = '0.5')


# %%
