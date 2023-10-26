#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as mpg
plt.close('all')

fig_ctr = 0

""" Banco de gráficas e tests exclusivo da función fitAOD (e correción de backlobes se corresponde)"""

# Selección de escenario - UMi, UMa, RMa, InH-Office-Mixed, InH-Office-Open

sce = "UMa"
transform = "AOA" #type None for no transform
delBacklobe = False
bool3D=False

# Posicións transmisor e receptor

tx = (0,0,10)
rx = (5,-90,1.5)
txArrayAngle = 45
rxArrayAngle = -180
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

clustersAD = clustersNAD.copy()
subpathsAD = subpathsNAD.copy()

if transform:
    if transform == "AOA":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.fullFitAOA(tx,rx,plinfo,clustersAD,subpathsAD,mode3D=bool3D)
    elif transform == "AOD":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.fullFitAOD(tx,rx,plinfo,clustersAD,subpathsAD)
    elif transform == "Delay":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.attemptFullFitDelay(tx,rx,plinfo,clustersAD,subpathsAD)
    elif transform == "Random":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.randomFitEpctClusters(tx,rx,plinfo,clustersAD,subpathsAD,E=1,P=[0,1/3,1/3,1/3])
    elif transform == "SRandom":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.randomFitAllSubpaths(tx,rx,plinfo,clustersAD,subpathsAD,P=[0,1/3,1/3,1/3])
    elif transform == "CERandom":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.randomFitEpctClusters(tx,rx,plinfo,clustersAD,subpathsAD,Ec=.5,Es=1,P=[0,.5,.5,0])
    elif transform == "SERandom":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.randomFitEpctClusters(tx,rx,plinfo,clustersAD,subpathsAD,Ec=1,Es=.5,P=[0,.5,.5,0])
    elif transform == "DERandom":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.randomFitEpctClusters(tx,rx,plinfo,clustersAD,subpathsAD,Ec=.9,Es=.5,P=[0,.5,.5,0])
    else:
        print("Transform '",transform,"' not supported")
    
if delBacklobe:    
    (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.fullDeleteBacklobes(tx,rx,plinfo,clustersAD,subpathsAD,tAOD=txArrayAngle,rAOA=rxArrayAngle)
    
nSP=len(subpathsNAD)
nKSP=len(subpathsAD)
nASP=np.sum(subpathsAD.Xs<np.inf)
nCL=len(clustersNAD)
nKCL=len(clustersAD)
nACL=np.sum(clustersAD.Xs<np.inf)

#Distancia entre receptor e posición do rebote
liRX_cNA = pd.Series(d2D/2, index=clustersNAD.index)
liRX_cA=np.sqrt((clustersAD.Xs-rx[0])**2+(clustersAD.Ys - rx[1])**2).where(clustersAD.Xs<np.inf,d2D/2)
liRX_sNA = pd.Series(d2D/2, index=subpathsNAD.index)
liRX_sA=np.sqrt((subpathsAD.Xs-rx[0])**2+(subpathsAD.Ys - rx[1])**2).where(subpathsAD.Xs<np.inf,d2D/2)
#Distancia entre transmisor e posicion do rebote
liTX_cNA = pd.Series(d2D/2, index=clustersNAD.index)
liTX_cA=np.sqrt((clustersAD.Xs)**2+(clustersAD.Ys)**2).where(clustersAD.Xs<np.inf,d2D/2)
liTX_sNA = pd.Series(d2D/2, index=subpathsNAD.index)
liTX_sA=np.sqrt((subpathsAD.Xs)**2+(subpathsAD.Ys)**2).where(subpathsAD.Xs<np.inf,d2D/2)

# ---- Gráfica 1, camiños non adaptados:

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
# plt.plot(clustersAD.Xs,clustersAD.Ys,'xk',label='C. Scatterers')
for i in range(0,nClusters): 
    plt.plot([tx[0],tx[0]+liTX_cNA[i]*np.cos(clustersNAD.AOD[i]*np.pi/180)],[tx[1],tx[1]+liTX_cNA[i]*np.sin(clustersNAD.AOD[i]*np.pi/180)],color=cm.jet(i/(nClusters-1)),linewidth = '0.9') 
    plt.plot([rx[0],rx[0]+liRX_cNA[i]*np.cos(clustersNAD.AOA[i]*np.pi/180)],[rx[1],rx[1]+liRX_cNA[i]*np.sin(clustersNAD.AOA[i]*np.pi/180)],color=cm.jet(i/(nClusters-1)),linewidth = '0.9')
plt.plot(tx[0],tx[1],'^r',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'sb',label='UE', linewidth='4.5')

def drawShadedArc(ax,pos,angle,radius):
    arcfill = np.linspace(-np.pi/2,np.pi/2,100)+angle
    xfill=pos[0]+radius*np.cos(arcfill)
    yfillC=pos[1]+radius*np.sin(arcfill)
    if np.isclose(np.mod(angle+np.pi/2,np.pi),np.pi/2):        
        yfillL=pos[1]
    else:
        yfillL=pos[1]+(xfill-pos[0])*np.tan(angle+np.pi/2)
    ax.fill_between(xfill,yfillC,yfillL,alpha=0.2, color='k')
    
if delBacklobe:
    drawShadedArc(plt.gca(),tx,txArrayAngle*np.pi/180-np.pi,d2D/2)
    drawShadedArc(plt.gca(),rx,rxArrayAngle*np.pi/180-np.pi,d2D/2)

legend = plt.legend(shadow=True, fontsize='10')

plt.savefig("../Figures/fit%s_clusNAD.eps"%(transform))


#Gráfica 2 - Camiños clusters adaptados 

fig_ctr+=1
fig = plt.figure(fig_ctr)

plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
plt.plot(clustersAD.Xs,clustersAD.Ys,'xk',label='C. Scatterers')
for ctr in range(0,clustersAD.shape[0]):
    i=clustersAD.index[ctr]
    plt.plot([tx[0],tx[0]+liTX_cA[i]*np.cos(clustersAD.AOD[i]*np.pi/180)],[tx[1],tx[1]+liTX_cA[i]*np.sin(clustersAD.AOD[i]*np.pi/180)],color=cm.jet(i/(nClusters-1)),linewidth = '0.9') 
    plt.plot([rx[0],rx[0]+liRX_cA[i]*np.cos(clustersAD.AOA[i]*np.pi/180)],[rx[1],rx[1]+liRX_cA[i]*np.sin(clustersAD.AOA[i]*np.pi/180)],color=cm.jet(i/(nClusters-1)),linewidth = '0.9')
plt.plot(tx[0],tx[1],'^r',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'sb',label='UE', linewidth='4.5')
if delBacklobe:
    drawShadedArc(plt.gca(),tx,txArrayAngle*np.pi/180-np.pi,d2D/2)
    drawShadedArc(plt.gca(),rx,rxArrayAngle*np.pi/180-np.pi,d2D/2)
legend = plt.legend(shadow=True, fontsize='10')

plt.title("%d|%d|%d of %d clusters adapted|kept|deleted"%(nACL,nKCL,nCL-nKCL,nCL))
plt.savefig("../Figures/fit%s_clusAD.eps"%(transform))

# Gráfica 3 - Subpaths non adaptados

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
# plt.plot(subpathsAD.Xs,subpathsAD.Ys,'xk',label='S. Scatterers')
for i in range(0,nClusters): 
    Nsp=subpathsNAD.AOD[i].size
    plt.plot(tx[0]+np.vstack([np.zeros(Nsp),liTX_sNA[i]*np.cos(subpathsNAD.AOD[i]*np.pi/180)]),tx[1]+np.vstack([np.zeros(Nsp),liTX_sNA[i]*np.sin(subpathsNAD.AOD[i]*np.pi/180)]),color=cm.jet(i/(nClusters-1)),linewidth = '0.9') 
    plt.plot(rx[0]+np.vstack([np.zeros(Nsp),liRX_sNA[i]*np.cos(subpathsNAD.AOA[i]*np.pi/180)]),rx[1]+np.vstack([np.zeros(Nsp),liRX_sNA[i]*np.sin(subpathsNAD.AOA[i]*np.pi/180)]),color=cm.jet(i/(nClusters-1)),linewidth = '0.9') 
plt.plot(tx[0],tx[1],'^r',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'sb',label='UE', linewidth='4.5')
if delBacklobe:
    drawShadedArc(plt.gca(),tx,txArrayAngle*np.pi/180-np.pi,d2D/2)
    drawShadedArc(plt.gca(),rx,rxArrayAngle*np.pi/180-np.pi,d2D/2)
legend = plt.legend(shadow=True, fontsize='10')
if transform == "Delay":
    plt.axis([-d2D+vLOS[0]/2,d2D+vLOS[0]/2,-d2D+vLOS[1]/2,d2D+vLOS[1]/2])
plt.savefig("../Figures/fit%s_subpNAD.eps"%(transform))

# Gráfica 4 - Subpaths adaptados

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
plt.plot(subpathsAD.Xs,subpathsAD.Ys,'xk',label='S. Scatterers')
for ctr in range(0,clustersAD.shape[0]):
    i=clustersAD.index[ctr]
    Nsp=subpathsAD.AOD[i].size
    plt.plot(tx[0]+np.vstack([np.zeros(Nsp),liTX_sA[i]*np.cos(subpathsAD.AOD[i]*np.pi/180)]),tx[1]+np.vstack([np.zeros(Nsp),liTX_sA[i]*np.sin(subpathsAD.AOD[i]*np.pi/180)]),color=cm.jet(i/(nClusters-1)),linewidth = '0.9') 
    plt.plot(rx[0]+np.vstack([np.zeros(Nsp),liRX_sA[i]*np.cos(subpathsAD.AOA[i]*np.pi/180)]),rx[1]+np.vstack([np.zeros(Nsp),liRX_sA[i]*np.sin(subpathsAD.AOA[i]*np.pi/180)]),color=cm.jet(i/(nClusters-1)),linewidth = '0.9') 
plt.plot(tx[0],tx[1],'^r',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'sb',label='UE', linewidth='4.5')
if delBacklobe:
    drawShadedArc(plt.gca(),tx,txArrayAngle*np.pi/180-np.pi,d2D/2)
    drawShadedArc(plt.gca(),rx,rxArrayAngle*np.pi/180-np.pi,d2D/2)
plt.title("%d|%d|%d of %d subpaths adapted|kept|deleted"%(nASP,nKSP,nSP-nKSP,nSP))
legend = plt.legend(shadow=True, fontsize='10')

plt.savefig("../Figures/fit%s_subpAD.eps"%(transform))

# Gráfica 5: Deck de subpaths AOD, AOA e delay non correxido

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.subplot(2,2,1, projection='polar',title="AoD")
for n in range(nClusters):   
    AOD_1c = subpathsNAD.loc[n,:].AOD.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpathsNAD.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOD_1c)
    plt.polar(AOD_1c*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClusters-1)) )
    plt.scatter(AOD_1c,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClusters-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize = 7)
if delBacklobe:
    plt.fill_between(np.linspace(np.pi/2,3*np.pi/2,100)+txArrayAngle*np.pi/180,-45,-10,alpha=0.2, color='k')
plt.subplot(2,2,2, projection='polar')
for n in range(nClusters):  
    AOA_1cf = subpathsNAD.loc[n,:].AOA.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10(subpathsNAD.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOA_1cf)
    plt.polar(AOA_1cf*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClusters-1)) )
    plt.scatter(AOA_1cf,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClusters-1)),marker='+')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize=7)
if delBacklobe:
    plt.fill_between(np.linspace(np.pi/2,3*np.pi/2,100)+rxArrayAngle*np.pi/180,-45,-10,alpha=0.2, color='k')
plt.subplot(2,1,2)
plt.ylabel("power [dB]")
plt.xlabel("TDoA (s)")
for n in range(nClusters):   
    markerline, stemlines, baseline = plt.stem( subpathsNAD.loc[n,:].tau.to_numpy() ,10*np.log10( subpathsNAD.loc[n,:].P.to_numpy() ),bottom=np.min(10*np.log10(subpathsAD.P.to_numpy())))
    plt.setp(stemlines, color=cm.jet(n/(nClusters-1)))
    plt.setp(markerline, color=cm.jet(n/(nClusters-1))) 
plt.grid()

plt.savefig("../Figures/fit%s_deckNAD.eps"%(transform))


# Gráfica 6: Deck de subpaths AOD, AOA e delay correxido

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.subplot(2,2,1, projection='polar',title="AoD")
for ctr in range(0,clustersAD.shape[0]):
    n=clustersAD.index[ctr]
    AOD_1c = subpathsAD.loc[n,:].AOD.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpathsAD.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOD_1c)
    plt.polar(AOD_1c*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClusters-1)) )
    plt.scatter(AOD_1c,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClusters-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize = 7)
if delBacklobe:
    plt.fill_between(np.linspace(np.pi/2,3*np.pi/2,100)+txArrayAngle*np.pi/180,-45,-10,alpha=0.2, color='k')
plt.subplot(2,2,2, projection='polar')
for ctr in range(0,clustersAD.shape[0]):
    n=clustersAD.index[ctr]
    AOA_1cf = subpathsAD.loc[n,:].AOA.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpathsAD.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOA_1cf)
    plt.polar(AOA_1cf*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClusters-1)) )
    plt.scatter(AOA_1cf,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClusters-1)),marker='+')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize=7)
if delBacklobe:
    plt.fill_between(np.linspace(np.pi/2,3*np.pi/2,100)+rxArrayAngle*np.pi/180,-45,-10,alpha=0.2, color='k')
plt.subplot(2,1,2)
plt.ylabel("power [dB]")
plt.xlabel("%d/%d subpaths adapted (%.2f%%)"%(nASP,nSP,100*nASP/nSP))
for ctr in range(0,clustersAD.shape[0]):
    n=clustersAD.index[ctr]
    markerline, stemlines, baseline = plt.stem( subpathsAD.loc[n,:].tau.to_numpy() ,10*np.log10( subpathsAD.loc[n,:].P.to_numpy() ),bottom=np.min(10*np.log10(subpathsAD.P.to_numpy())))
    plt.setp(stemlines, color=cm.jet(n/(nClusters-1)))
    plt.setp(markerline, color=cm.jet(n/(nClusters-1))) 
plt.grid()

plt.savefig("../Figures/fit%s_deckAD.eps"%(transform))

