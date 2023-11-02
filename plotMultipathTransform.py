#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import sys
sys.path.append('../')
import threeGPPMultipathGenerator as mpg
plt.close('all')

fig_ctr = 0

# Selección de escenario - UMi, UMa, RMa, InH-Office-Mixed, InH-Office-Open

sce = "UMa"

# Posicións transmisor e receptor

tx = (5, 5, 10)
rx = (5, 110, 1)
vLOS=np.array(rx)-np.array(tx)
d2D=np.linalg.norm(vLOS[0:2])
phi0 = 0

# ----------------------------
# ---- Canle A, largeBW = true
# ----------------------------

modelA = mpg.ThreeGPPMultipathChannelModel(scenario = sce, bLargeBandwidthOption=True)
plinfo,macro,clustersNAD,subpathsNAD = modelA.create_channel(tx,rx)
los, PLfree, SF = plinfo
nClusters = clustersNAD.shape[0]
nNLOSsp=subpathsNAD.loc[1,:].shape[0]


# Adaptación canle 1:


lsubpaths=[subpathsNAD]
lClusters=[clustersNAD]
Npos = 5
despla=(5,0,0)

lrxPosNext0=[rx[0]]
lrxPosNext1=[rx[1]]
for ctr in range(1,Npos):
    delta =[x*ctr for x in despla]
    rxPosNext = tuple(v1 + v2 for v1, v2 in zip(rx, delta))
    lrxPosNext0.append(rxPosNext[0])
    lrxPosNext1.append(rxPosNext[1])
    plinfo1, macro1, clustersSiguiente, subpaths1 = modelA.create_channel(tx,rxPosNext)
    lClusters.append(clustersSiguiente)
    lsubpaths.append(subpaths1)
    clustersONE = lClusters[ctr]


for subpathsCTR in range(Npos-1):
    subpathONE = lsubpaths[subpathsCTR]
    

    
z=0
for clustersONE in lClusters:
    fig_ctr += 1  # Incrementa el número de figura en cada iteración
    fig = plt.figure(fig_ctr)
    nSP=len(subpathsNAD)
    nCL=len(clustersNAD)

    #Distancia entre receptor e posición do rebote
    liRX_cNA = pd.Series(d2D/2, index=clustersONE.index)
    liRX_sNA = pd.Series(d2D/2, index=subpathONE.index)
    #Distancia entre transmisor e posicion do rebote
    liTX_cNA = pd.Series(d2D/2, index=clustersONE.index)
    liTX_sNA = pd.Series(d2D/2, index=subpathONE.index)

    # ---- Gráfica 1,clusters
   
    plt.grid(linestyle = '--')
    plt.xlabel('x-location (m)')
    plt.ylabel('y-location (m)')

    plt.plot([tx[0],lrxPosNext0[z]],[tx[1],lrxPosNext1[z]],'--')
    # plt.plot(clustersAD.Xs,clustersAD.Ys,'xk',label='C. Scatterers')
  
    for i in clustersONE.index: 
        
        plt.plot([tx[0],tx[0]+liTX_cNA[i]*np.cos(clustersONE.ZOD[i]*np.pi/180)],[tx[1],tx[1]+liTX_cNA[i]*np.sin(clustersONE.ZOD[i]*np.pi/180)],color=cm.jet(i/(nClusters-1)),linewidth = '0.9') 
   
        plt.plot([lrxPosNext0[z],lrxPosNext0[z]+liRX_cNA[i]*np.cos(clustersONE.ZOA[i]*np.pi/180)],[lrxPosNext1[z],lrxPosNext1[z]+liRX_cNA[i]*np.sin(clustersONE.ZOA[i]*np.pi/180)],color=cm.jet(i/(nClusters-1)),linewidth = '0.9')
     
    plt.plot(tx[0],tx[1],'^r',label='BS',linewidth = '4.5')
    plt.plot(lrxPosNext0[z],lrxPosNext1[z],'sb',label='UE', linewidth='4.5')
   
    legend = plt.legend(shadow=True, fontsize='10')
        
    filename = f"figura_{fig_ctr}.png"
    plt.savefig(filename)
    z=z+1
    
    
    # Incrementa el contador de figuras
    

z=0
#  Gráfica 2, subpaths
for subpathONE in lsubpaths:
    fig_ctr+=1
    fig = plt.figure(fig_ctr)
    plt.grid(linestyle = '--')
    plt.xlabel('x-location (m)')
    plt.ylabel('y-location (m)')

    plt.plot([tx[0],lrxPosNext0[z]],[tx[1],lrxPosNext1[z]],'--')
    
    for i in range(0,nClusters): 
        Nsp=subpathONE.AOD[i].size
        plt.plot(tx[0]+np.vstack([np.zeros(Nsp),liTX_sNA[i]*np.cos(subpathONE.AOD[i]*np.pi/180)]),tx[1]+np.vstack([np.zeros(Nsp),liTX_sNA[i]*np.sin(subpathONE.AOD[i]*np.pi/180)]),color=cm.jet(i/(nClusters-1)),linewidth = '0.9') 
        plt.plot(lrxPosNext0[z]+np.vstack([np.zeros(Nsp),liRX_sNA[i]*np.cos(subpathONE.AOA[i]*np.pi/180)]),lrxPosNext1[z]+np.vstack([np.zeros(Nsp),liRX_sNA[i]*np.sin(subpathONE.AOA[i]*np.pi/180)]),color=cm.jet(i/(nClusters-1)),linewidth = '0.9') 
    plt.plot(tx[0],tx[1],'^r',label='BS',linewidth = '4.5')
    plt.plot(lrxPosNext0[z],lrxPosNext1[z],'sb',label='UE', linewidth='4.5')
    legend = plt.legend(shadow=True, fontsize='10')
    filename = f"subpaths{fig_ctr}.png"
    plt.savefig(filename)
    z=z+1

# Gráfica 3
    fig_ctr+=1
    fig = plt.figure(fig_ctr)
    plt.subplot(2,2,1, projection='polar',title="AoD")
    for n in range(nClusters):   
        AOD_1c = subpathONE.loc[n,:].AOD *np.pi/180
        pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpathONE.loc[n,:].powC  ),-45)
        AOD_1c = AOD_1c.to_numpy()
        Nsp=len(AOD_1c)
        plt.polar(AOD_1c*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClusters-1)) )
        plt.scatter(AOD_1c,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClusters-1)),marker='<')
    plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize = 7)
    plt.subplot(2,2,2, projection='polar')
    for n in range(nClusters):  
        AOA_1cf = subpathONE.loc[n,:].AOA *np.pi/180
        pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10(subpathONE.loc[n,:].powC  ),-45)
        AOA_1cf =AOA_1cf.to_numpy()
        Nsp=len(AOA_1cf)
        plt.polar(AOA_1cf*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClusters-1)) )
        plt.scatter(AOA_1cf,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClusters-1)),marker='+')
    plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize=7)
    plt.subplot(2,1,2)
    plt.ylabel("power [dB]")
    plt.xlabel("TDoA (s)")
    for n in range(nClusters):   
        markerline, stemlines, baseline = plt.stem( subpathONE.loc[n,:].tau ,10*np.log10( subpathONE.loc[n,:].powC ),bottom=np.min(10*np.log10(subpathONE.powC)))
        plt.setp(stemlines, color=cm.jet(n/(nClusters-1)))
        plt.setp(markerline, color=cm.jet(n/(nClusters-1))) 
    plt.grid()

    plt.savefig('Deck:%d'%(Npos))
plt.show()