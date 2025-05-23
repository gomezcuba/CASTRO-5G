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

sce = "InF-DL"
transform = "DERandom" #type None for no transform
delBacklobe = False
bool3D=True

# Posicións transmisor e receptor

tx = (0,0,10)
rx = (100,0,1.5)
nItsctRad = 25 #10m
txArrayAngle = 0
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
nNLOSsp=subpathsNAD.loc[1,:].shape[0]

# print("Condition: ",(clustersNAD.P[0]<.9)&(los))
# clustersNAD.to_csv("../Results/changendataAdaptation3GPP/clustersNAD.csv")
# subpathsNAD.to_csv("../Results/changendataAdaptation3GPP/subpathsNAD.csv")
# macro.to_csv("../Results/changendataAdaptation3GPP/macro.csv")
# np.savez("../Results/changendataAdaptation3GPP/plinfo.npz",plinfo=plinfo)

# data=np.load("../Results/changendataAdaptation3GPP/plinfo.npz")
# plinfo=tuple(data['plinfo'])
# macro=pd.read_csv("../Results/changendataAdaptation3GPP/macro.csv")
# macro=macro.drop('Unnamed: 0',axis=1)
# clustersNAD=pd.read_csv("../Results/changendataAdaptation3GPP/clustersNAD.csv",index_col=["n"])
# subpathsNAD=pd.read_csv("../Results/changendataAdaptation3GPP/subpathsNAD.csv",index_col=["n","m"])
nClusters = clustersNAD.shape[0]
# plinfo=data.plinfo
# macro=data.macro
# clustersNAD=data.clustersNAD
# subpathsNAD=data.subpathsNAD

clustersAD = clustersNAD.copy()
subpathsAD = subpathsNAD.copy()

if transform:
    if transform == "AOA":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.fullFitAoA(tx,rx,plinfo,clustersAD,subpathsAD,mode3D=bool3D)
    elif transform == "AOD":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.fullFitAoD(tx,rx,plinfo,clustersAD,subpathsAD,mode3D=bool3D)
    elif transform == "TDOA":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.attemptFullFitTDoA(tx,rx,plinfo,clustersAD,subpathsAD,mode3D=bool3D,fallbackFun=("relax3D" if bool3D else None))
    elif transform == "SRandom":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.randomFitAllSubpaths(tx,rx,plinfo,clustersAD,subpathsAD,P=[0,1/3,1/3,1/3],mode3D=bool3D)
    elif transform == "CRandom":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.randomFitEpctClusters(tx,rx,plinfo,clustersAD,subpathsAD,Ec=1,Es=1,P=[0,1/3,1/3,1/3],mode3D=bool3D)
    elif transform == "CERandom":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.randomFitEpctClusters(tx,rx,plinfo,clustersAD,subpathsAD,Ec=.5,Es=1,P=[0,.5,.5,0],mode3D=bool3D)
    elif transform == "SERandom":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.randomFitEpctClusters(tx,rx,plinfo,clustersAD,subpathsAD,Ec=1,Es=.5,P=[0,.5,.5,0],mode3D=bool3D)
    elif transform == "DERandom":
        (tx,rx,plinfo,clustersAD,subpathsAD)  = modelA.randomFitEpctClusters(tx,rx,plinfo,clustersAD,subpathsAD,Ec=.75,Es=.75,P=[0,.5,.5,0],mode3D=bool3D)
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

c=3e8
#Distancia entre receptor e posición do rebote
liRX_cNA = (clustersNAD.TDoA*c+d2D)/2 if nItsctRad is None else nItsctRad*np.ones_like(clustersNAD.TDoA)
liRX_cA=np.sqrt((clustersAD.Xs-rx[0])**2+(clustersAD.Ys - rx[1])**2).where(clustersAD.Xs<np.inf,liRX_cNA)
liRX_sNA = (subpathsNAD.TDoA*c+d2D)/2 if nItsctRad is None else nItsctRad*np.ones_like(subpathsNAD.TDoA)
liRX_sA=np.sqrt((subpathsAD.Xs-rx[0])**2+(subpathsAD.Ys - rx[1])**2).where(subpathsAD.Xs<np.inf,liRX_sNA)
#Distancia entre transmisor e posicion do rebote
liTX_cNA = (clustersNAD.TDoA*c+d2D)/2 if nItsctRad is None else nItsctRad*np.ones_like(clustersNAD.TDoA)
liTX_cA=np.sqrt((clustersAD.Xs)**2+(clustersAD.Ys)**2).where(clustersAD.Xs<np.inf,liTX_cNA)
liTX_sNA = (subpathsNAD.TDoA*c+d2D)/2 if nItsctRad is None else nItsctRad*np.ones_like(subpathsNAD.TDoA)
liTX_sA=np.sqrt((subpathsAD.Xs)**2+(subpathsAD.Ys)**2).where(subpathsAD.Xs<np.inf,liTX_sNA)

# ---- Gráfica 1, camiños non adaptados:

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
# plt.plot(clustersAD.Xs,clustersAD.Ys,'xk',label='C. Scatterers')
uD = modelA.getUnitaryVectors(clustersNAD.AoD*np.pi/180)
uA = modelA.getUnitaryVectors(clustersNAD.AoA*np.pi/180)
for i in range(0,nClusters): 
    plt.plot([tx[0],tx[0]+liTX_cNA[i]*uD[i,0]],[tx[1],tx[1]+liTX_cNA[i]*uD[i,1]],color=cm.jet(i/(nClusters-1)),linewidth = '2')
    plt.plot(tx[0]+liTX_cNA[i]*uD[i,0],tx[1]+liTX_cNA[i]*uD[i,1],color=cm.jet(i/(nClusters-1)),linewidth = '3',markersize=10,marker=(3,0,-90+clustersNAD.AoD[i]))      
    plt.plot([rx[0],rx[0]+liRX_cNA[i]*uA[i,0]],[rx[1],rx[1]+liRX_cNA[i]*uA[i,1]],color=cm.jet(i/(nClusters-1)),linewidth = '2')
    plt.plot(rx[0]+.05*liRX_cNA[i]*uA[i,0],rx[1]+.05*liRX_cNA[i]*uA[i,1],color=cm.jet(i/(nClusters-1)),linewidth = '3',markersize=10,marker=(3,0,90+clustersNAD.AoA[i])) 
    # plt.gca().annotate("", xytext=tx[0:2], xy=tx[0:2]+liTX_cNA[i]*uD[i,:],arrowprops=dict(arrowstyle="->"),linecolor=cm.jet(i/(nClusters-1)))
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

plt.savefig("../Figures/fit%s_clusNAD.svg"%(transform))


#Gráfica 2 - Camiños clusters adaptados 

fig_ctr+=1
fig = plt.figure(fig_ctr)

plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
plt.plot(clustersAD.Xs,clustersAD.Ys,'xk',label='C. Scatterers',linewidth='2')
uD = modelA.getUnitaryVectors(clustersAD.AoD*np.pi/180)
uA = modelA.getUnitaryVectors(clustersAD.AoA*np.pi/180)
for ctr in range(0,clustersAD.shape[0]):
    i=clustersAD.index[ctr]
    plt.plot([tx[0],tx[0]+liTX_cA[i]*uD[i,0]],[tx[1],tx[1]+liTX_cA[i]*uD[i,1]],color=cm.jet(i/(nClusters-1)),linewidth = '2')
    plt.plot(tx[0]+.95*liTX_cA[i]*uD[i,0],tx[1]+.95*liTX_cA[i]*uD[i,1],color=cm.jet(i/(nClusters-1)),linewidth = '3',markersize=10,marker=(3,0,-90+clustersAD.AoD[i]))      
    plt.plot([rx[0],rx[0]+liRX_cA[i]*uA[i,0]],[rx[1],rx[1]+liRX_cA[i]*uA[i,1]],color=cm.jet(i/(nClusters-1)),linewidth = '2')
    plt.plot(rx[0]+.05*liRX_cA[i]*uA[i,0],rx[1]+.05*liRX_cA[i]*uA[i,1],color=cm.jet(i/(nClusters-1)),linewidth = '3',markersize=10,marker=(3,0,90+clustersAD.AoA[i])) 
plt.plot(tx[0],tx[1],'^r',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'sb',label='UE', linewidth='4.5')
if delBacklobe:
    drawShadedArc(plt.gca(),tx,txArrayAngle*np.pi/180-np.pi,d2D/2)
    drawShadedArc(plt.gca(),rx,rxArrayAngle*np.pi/180-np.pi,d2D/2)
legend = plt.legend(shadow=True, fontsize='10')

if delBacklobe:
    plt.title("%d|%d|%d of %d clusters adapted|kept|deleted"%(nACL,nKCL,nCL-nKCL,nCL))
else:
    plt.title("%d of %d clusters adapted"%(nACL,nCL))
plt.savefig("../Figures/fit%s_clusAD.svg"%(transform))

# Gráfica 3 - Subpaths non adaptadosz

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
# plt.plot(subpathsAD.Xs,subpathsAD.Ys,'xk',label='S. Scatterers')
uD = modelA.getUnitaryVectors(subpathsNAD.AoD*np.pi/180)
uA = modelA.getUnitaryVectors(subpathsNAD.AoA*np.pi/180)
for i in range(0,nClusters): 
    Nsp=subpathsNAD.AoD[i].size
    plt.plot(tx[0]+np.vstack([np.zeros(Nsp),liTX_sNA[i]*uD[i*Nsp:(i+1)*Nsp,0]]),tx[1]+np.vstack([np.zeros(Nsp),liTX_sNA[i]*uD[i*Nsp:(i+1)*Nsp,1]]),color=cm.jet(i/(nClusters-1)),linewidth = '2') 
    plt.plot(rx[0]+np.vstack([np.zeros(Nsp),liRX_sNA[i]*uA[i*Nsp:(i+1)*Nsp,0]]),rx[1]+np.vstack([np.zeros(Nsp),liRX_sNA[i]*uA[i*Nsp:(i+1)*Nsp,1]]),color=cm.jet(i/(nClusters-1)),linewidth = '2') 
plt.plot(tx[0],tx[1],'^r',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'sb',label='UE', linewidth='4.5')
if delBacklobe:
    drawShadedArc(plt.gca(),tx,txArrayAngle*np.pi/180-np.pi,d2D/2)
    drawShadedArc(plt.gca(),rx,rxArrayAngle*np.pi/180-np.pi,d2D/2)
legend = plt.legend(shadow=True, fontsize='10')
if transform == "TDOA":
    plt.axis([-d2D+vLOS[0]/2,d2D+vLOS[0]/2,-d2D+vLOS[1]/2,d2D+vLOS[1]/2])
plt.savefig("../Figures/fit%s_subpNAD.svg"%(transform))

# Gráfica 4 - Subpaths adaptados

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
plt.plot(subpathsAD.Xs,subpathsAD.Ys,'xk',label='S. Scatterers')
uD = modelA.getUnitaryVectors(subpathsAD.AoD*np.pi/180)
uA = modelA.getUnitaryVectors(subpathsAD.AoA*np.pi/180)
for ctr in range(0,clustersAD.shape[0]):
    i=clustersAD.index[ctr]
    Nsp=subpathsAD.AoD[i].size
    plt.plot(tx[0]+np.vstack([np.zeros(Nsp),liTX_sA[i]*uD[i*Nsp:(i+1)*Nsp,0]]),tx[1]+np.vstack([np.zeros(Nsp),liTX_sA[i]*uD[i*Nsp:(i+1)*Nsp,1]]),color=cm.jet(i/(nClusters-1)),linewidth = '2') 
    plt.plot(rx[0]+np.vstack([np.zeros(Nsp),liRX_sA[i]*uA[i*Nsp:(i+1)*Nsp,0]]),rx[1]+np.vstack([np.zeros(Nsp),liRX_sA[i]*uA[i*Nsp:(i+1)*Nsp,1]]),color=cm.jet(i/(nClusters-1)),linewidth = '2') 
plt.plot(tx[0],tx[1],'^r',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'sb',label='UE', linewidth='4.5')
if delBacklobe:
    drawShadedArc(plt.gca(),tx,txArrayAngle*np.pi/180-np.pi,d2D/2)
    drawShadedArc(plt.gca(),rx,rxArrayAngle*np.pi/180-np.pi,d2D/2)    
    plt.title("%d|%d|%d of %d subpaths adapted|kept|deleted"%(nASP,nKSP,nSP-nKSP,nSP))
else:
    plt.title("%d of %d subpaths adapted"%(nASP,nSP))
legend = plt.legend(shadow=True, fontsize='10')

plt.savefig("../Figures/fit%s_subpAD.svg"%(transform))

# Gráfica 5: Deck de subpaths AOD, AOA e TDOA non correxido

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.subplot(2,2,1, projection='polar',title="AoD")
for n in range(nClusters):   
    AOD_1c = subpathsNAD.loc[n,:].AoD.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpathsNAD.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOD_1c)
    plt.polar(AOD_1c*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClusters-1)) )
    plt.scatter(AOD_1c,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClusters-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize = 7)
if delBacklobe:
    plt.fill_between(np.linspace(np.pi/2,3*np.pi/2,100)+txArrayAngle*np.pi/180,-45,-10,alpha=0.2, color='k')
plt.subplot(2,2,2, projection='polar')
for n in range(nClusters):  
    AOA_1cf = subpathsNAD.loc[n,:].AoA.to_numpy() *np.pi/180
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
    markerline, stemlines, baseline = plt.stem( subpathsNAD.loc[n,:].TDoA.to_numpy() ,10*np.log10( subpathsNAD.loc[n,:].P.to_numpy() ),bottom=np.min(10*np.log10(subpathsAD.P.to_numpy())))
    plt.setp(stemlines, color=cm.jet(n/(nClusters-1)))
    plt.setp(markerline, color=cm.jet(n/(nClusters-1))) 
plt.grid()

plt.savefig("../Figures/fit%s_deckNAD.svg"%(transform))


# Gráfica 6: Deck de subpaths AOD, AOA e TDOA correxido

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.subplot(2,2,1, projection='polar',title="AoD")
for ctr in range(0,clustersAD.shape[0]):
    n=clustersAD.index[ctr]
    AOD_1c = subpathsAD.loc[n,:].AoD.to_numpy() *np.pi/180
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
    AOA_1cf = subpathsAD.loc[n,:].AoA.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpathsAD.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOA_1cf)
    plt.polar(AOA_1cf*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClusters-1)) )
    plt.scatter(AOA_1cf,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClusters-1)),marker='+')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize=7)
if delBacklobe:
    plt.fill_between(np.linspace(np.pi/2,3*np.pi/2,100)+rxArrayAngle*np.pi/180,-45,-10,alpha=0.2, color='k')
plt.subplot(2,1,2)
plt.ylabel("power [dB]")
plt.xlabel("%d/%d subpaths adapted (%.2f\\%%)"%(nASP,nSP,100*nASP/nSP))
for ctr in range(0,clustersAD.shape[0]):
    n=clustersAD.index[ctr]
    markerline, stemlines, baseline = plt.stem( subpathsAD.loc[n,:].TDoA.to_numpy() ,10*np.log10( subpathsAD.loc[n,:].P.to_numpy() ),bottom=np.min(10*np.log10(subpathsAD.P.to_numpy())))
    plt.setp(stemlines, color=cm.jet(n/(nClusters-1)))
    plt.setp(markerline, color=cm.jet(n/(nClusters-1))) 
plt.grid()

plt.savefig("../Figures/fit%s_deckAD.svg"%(transform))

