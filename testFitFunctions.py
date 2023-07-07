# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import threeGPPMultipathGenerator as mpg
import multipathChannel as mc
from matplotlib import cm

txPos = (0,0,10)
rxPos=(50,0,1.5)
resAOD = np.array([-26.4070614 , -17.50105576, -11.19985824,  21.20177412,
        44.39181685, -27.23832243,  -5.6259779 ,  -0.54415715,
       -15.44211778, -16.1235986 , -30.87259028, -17.41318048,
         4.60819095, -34.58314023, -16.83662437,  24.74840315,
        13.64277166,   7.56088152, -11.35766037])
tau = np.array([0.00000000e+00, 1.26257744e-07, 1.90503024e-07, 3.05765970e-07,
       5.28757609e-07, 7.68003813e-07, 8.36037486e-07, 9.81261981e-07,
       9.90784676e-07, 1.14744248e-06, 1.30927350e-06, 1.37149868e-06,
       1.40778327e-06, 1.86788685e-06, 1.95813480e-06, 2.43761018e-06,
       2.49842475e-06, 3.97569578e-06, 5.14698794e-06])
aoa = np.array([  0.        , 301.47621265, 329.82226723,  42.7304772 ,
        67.26541771, 321.68196781, 352.13694532, 359.27099778,
       339.45964842, 339.28277639, 321.78496528, 338.44440155,
         5.69768743, 319.70810863, 340.34918265,  28.00704216,
        15.44226848,   8.19273178, 347.91206652])


vLOS = np.array(rxPos) - np.array(txPos)
l0 = np.linalg.norm(vLOS[0:-1])
li = l0+tau*3e8
losAOA = (np.mod( np.arctan(vLOS[1]/vLOS[0])+np.pi*(vLOS[0]<0)+np.pi,2*np.pi))
dAOA = aoa*(np.pi/180.0)-losAOA

nu= li/l0
cosdAOA = np.cos(dAOA)
sindAOA = np.sin(dAOA)

A=nu**2+1-2*cosdAOA*nu
B=2*sindAOA*(1-nu*cosdAOA)
C=(sindAOA**2)*(1-nu**2)
sol1= -sindAOA
sol2= sindAOA*(nu**2-1) /  ( nu**2+1-2*cosdAOA*nu )
sol2[(nu==1)&(cosdAOA==1)] = 0 #LOS path

sols = np.zeros((4,aoa.size)) 
sols[0,:] = np.arcsin(sol1)
sols[1,:] = np.arcsin(sol2)
sols[2,:] = np.pi - np.arcsin(sol1)
sols[3,:] = np.pi - np.arcsin(sol2)

#Ubicacion dos rebotes 
x=(vLOS[1]-vLOS[0]*np.tan(losAOA+np.pi-sols))/(np.tan(aoa *(np.pi/180) )-np.tan(losAOA+np.pi-sols))
x[1,(nu==1)&(cosdAOA==1)] = vLOS[0]/2
x[3,(nu==1)&(cosdAOA==1)] = vLOS[0]/2
y=x*np.tan(aoa *(np.pi/180) ) 

#Mellor solucion - a mais semellante á distancia do path evaluado
dist=np.sqrt(x**2+y**2)+np.sqrt((x-vLOS[0])**2+(y-vLOS[1])**2)
solIndx=np.argmin(np.abs(dist-li),axis=0)
aodAux =sols[solIndx,range(li.size)]
aod = np.mod(np.pi+losAOA-aodAux,2*np.pi) * (180.0/np.pi)


xPathLoc = x[solIndx,range(li.size)]
yPathLoc = y[solIndx,range(li.size)]

liRX = np.sqrt((xPathLoc-rxPos[0])**2+(yPathLoc - rxPos[1])**2)
liTX = np.sqrt(xPathLoc**2+yPathLoc**2)

# %%
