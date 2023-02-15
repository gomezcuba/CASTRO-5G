# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:28:39 2022

@author: user
"""
import threeGPPMultipathGenerator as pg
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

#-------------------------PLOTEO MACRO-------------------------------------
txPos = (0,0,10)
Nusers = 300
cell = 15 #m
distance = 15*cell#m
numberCells = np.ceil(distance/cell).astype(int)

users = []
x = []
y = []
contadorY = []
contadorX = []
key=[]

modelMacro = pg.ThreeGPPMultipathChannelModel(sce="UMi")

macroChannel = []
dfList = []
mm = (np.zeros((numberCells,numberCells))).astype(np.int32)


for i in range(Nusers):
    posX = round(np.random.uniform(0, distance-1), 2)
    posY = round(np.random.uniform(0, distance-1), 2)
    users.append(tuple([posX,posY,1.5]))
    x.append(posX)
    y.append(posY)
    RgridXIndex = int(np.abs((txPos[0]-x[i]) // cell)-1)
    RgridYIndex  = int(np.abs((txPos[1]-y[i]) // cell)-1)
    if RgridXIndex < 0:
        contadorX.append(0)
    else:
        contadorX.append(RgridXIndex)
    if RgridYIndex < 0:
        contadorY.append(0)
    else:
        contadorY.append(RgridYIndex)
    mm[contadorX[i],contadorY[i]] += 1 
    macroChannel.append(modelMacro.create_channel(txPos,users[i])[0])
    dfList.append(modelMacro.create_channel(txPos,users[i])[1])
    key.append([contadorX[i],contadorY[i]])
       
#plt.subplot(1,2,1)
#plt.imshow(mm, cmap='RdYlBu_r', interpolation='nearest')
#plt.colorbar(label="Number of users", orientation="vertical")
#plt.show()

#Eliminamos los macros que se repiten para quedarnos con uno por casilla
result = []
for item in macroChannel:
    if item not in result:
        result.append(item)
keyResult = []
for item in key:
    if item not in keyResult:
        keyResult.append(str(item))
     
result = tuple(result)        
keyResult = tuple(keyResult)
dictMacro = (dict(zip(keyResult, result)))


macroDS = [] 
macroASA = []


for i in range(len(macroChannel)):
    macroDS.append(str(macroChannel[i].ds)) 
    macroASA.append(str(macroChannel[i].asa))


contadorRepes = Counter(macroDS)

listaRepes = []
for i in range(len(contadorRepes)):
    listaRepes.append(contadorRepes[macroDS[i]])

colorDS = []
for item in macroDS:
    if item not in colorDS:
       colorDS.append(item)

listaRepes = []
for i in range(len(contadorRepes)):
    listaRepes.append(contadorRepes[colorDS[i]])
  
indexList = []
for j in range(len(colorDS)):
    for i in range(len(macroDS)):
        if macroDS[i] == colorDS[j]:
            indexList.append(i)
            
color = [] 
ayuda = []
for i in range(len(colorDS)):
    ayuda.append(i)
    color.append(np.random.choice(range(256), size=3))

ultima = []
colorUltima = []
for i in range(len(listaRepes)):
    cont = listaRepes[i]
    while cont != 0:
        ultima.append(ayuda[i])
        colorUltima.append(color[i])
        cont = cont-1

listaFinal = []
colorFinal = []
listaFinal = [0]*len(ultima)
colorFinal = [0]*len(ultima)
for index, value in zip(indexList, ultima):
    listaFinal[index] = value
    colorFinal[index] = value

fig,ax = plt.subplots()
plt.xlim([0, distance+5])
plt.ylim([0, distance+5])
plt.yticks(np.arange(0, distance+5, cell))
plt.xticks(np.arange(0, distance+5, cell))
plt.grid(axis='both',color='red')
sc = plt.scatter(x,y,s=15,c=np.take(colorUltima,listaFinal),data=macroDS)
plt.title('User Distribution')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
#plt.scatter(x,y,c='red',data=macroASA)
#plt.show()