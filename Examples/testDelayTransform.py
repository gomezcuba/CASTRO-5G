#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as mpg
from CASTRO5G import multipathChannel as mc
plt.close('all')

fig_ctr = 0

""" Banco de gráficas e tests exclusivo da función fitAOD (e correción de backlobes se corresponde)"""

# Posicións transmisor e receptor

tx = (0,0,10)
rx = (45,45,1.5)
phi0 = 0

# Selección de escenario - UMi, UMa, RMa, InH-Office-Mixed, InH-Office-Open

sce = "UMi"

# ----------------------------
# ---- Canle A, largeBW = true
# ----------------------------

modelA = mpg.ThreeGPPMultipathChannelModel(scenario = sce, bLargeBandwidthOption=True)
plinfoA,macroA,clustersNAD,subpathsNAD = modelA.create_channel(tx,rx)


# Adaptación canle 1:

clustersAD = clustersNAD.copy()
subpathsAD = subpathsNAD.copy()

(tx,rx,plinfoA,clustersAD,subpathsAD)  = modelA.attemptFullFitDelay(tx,rx,plinfoA,clustersAD,subpathsAD)

# OPCIONAL -- Se queremos ademais correxir backlobes:
#-------
#clustersAD = modelA.deleteBacklobes(clustersAD,phi0)
#subpathsAD = modelA.deleteBacklobes(subpathsAD,phi0)
#-------

#Delays non correxidos
tau_cA = clustersNAD['tau'].T.to_numpy()
tau_sA = subpathsNAD['tau'].T.to_numpy()

#AOAs non modificados
AOA_cA = clustersNAD['AOA'].T.to_numpy() * (np.pi/(180.0))
AOA_sA = subpathsNAD['AOA'].T.to_numpy() * (np.pi/(180.0))


#Posición dos rebotes:
xc_A,yc_A = [clustersAD['Xs'].T.to_numpy(),clustersAD['Ys'].T.to_numpy()]
xs_A,ys_A = [subpathsAD['Xs'].T.to_numpy(),subpathsAD['Ys'].T.to_numpy()]

#Distancia entre receptor e posición do rebote
liRX_cA = np.where(xc_A<np.inf,np.sqrt((xc_A-rx[0])**2+(yc_A - rx[1])**2),25)
liRX_sA = np.where(xs_A<np.inf,np.sqrt((xs_A-rx[0])**2+(ys_A - rx[1])**2),25)
#Distancia entre transmisor e receptor do rebote
liTX_cA = np.where(xc_A<np.inf,np.sqrt(xc_A**2+yc_A**2) ,25)
liTX_sA = np.where(xs_A<np.inf,np.sqrt(xs_A**2+ys_A**2) ,25)

#delays correxidos
tau_cfA = clustersAD['tau'].T.to_numpy()
tau_sfA = subpathsAD['tau'].T.to_numpy()

#AOD non varía
AOD_cA = clustersAD['AOD'].T.to_numpy() * (np.pi/(180.0))
AOD_sA = subpathsAD['AOD'].T.to_numpy() * (np.pi/(180.0))


# ---- Gráfica 1, camiños non adaptados:

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.title("Delay sen correxir (clusters)")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

nClus = tau_cA.size
nSubp = tau_sA.size

plt.plot(tx[0],tx[1],'^g',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^r',label='UE', linewidth='4.5')
plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
plt.plot(xc_A,yc_A,'x',label='Rebotes')
for i in range(0,1): 
    plt.plot([tx[0],tx[0]+liTX_cA[i]*np.cos(AOD_cA[i])],[tx[1],tx[1]+liTX_cA[i]*np.sin(AOD_cA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.5') 
    plt.plot([rx[0],rx[0]+liRX_cA[i]*np.cos(AOA_cA[i])],[rx[1],rx[1]+liRX_cA[i]*np.sin(AOA_cA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')
plt.savefig("../Figures/fitDelay_clusNAD.png")

# Gráfica 2 - Camiños clusters adaptados 

fig_ctr+=1
fig = plt.figure(fig_ctr)
# plt.subplot(2,1,1)
plt.title("Delay correxido (clusters)")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

plt.plot(tx[0],tx[1],'^g',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^r',label='UE', linewidth='4.5')
plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
plt.plot(xc_A,yc_A,'x',label='Rebotes')
for i in range(0,AOD_cA.size): 
    plt.plot([tx[0],tx[0]+liTX_cA[i]*np.cos(AOD_cA[i])],[tx[1],tx[1]+liTX_cA[i]*np.sin(AOD_cA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.5') 
    plt.plot([rx[0],rx[0]+liRX_cA[i]*np.cos(AOA_cA[i])],[rx[1],rx[1]+liRX_cA[i]*np.sin(AOA_cA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')
plt.savefig("../Figures/fitDelay_clusAD.png")

# Gráfica 3 - Subpaths non adaptados

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.title("Delay non correxidos (subpaths)")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')
plt.xlim(-60,60)
plt.ylim(-60,60)
plt.plot(tx[0],tx[1],'^g',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^r',label='UE', linewidth='4.5')

plt.plot(xs_A,ys_A,'x',label='Rebotes subpaths')
for i in range(0,AOD_sA.size):
    plt.plot([tx[0],tx[0]+liTX_sA[i]*np.cos(AOD_sA[i])],[tx[1],tx[1]+liTX_sA[i]*np.sin(AOD_sA[i])],color=cm.jet(i/(nSubp-1)),linewidth = '0.5') 
    plt.plot([rx[0],rx[0]+liRX_sA[i]*np.cos(AOA_sA[i])],[rx[1],rx[1]+liRX_sA[i]*np.sin(AOA_sA[i])],color=cm.jet(i/(nSubp-1)),linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')
plt.savefig("../Figures/fitDelay_subpNAD.png")



# Gráfica 4 - Subpaths adaptados

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.title("Delay correxidos (subpaths)")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')
plt.xlim(-60,60)
plt.ylim(-60,60)
plt.plot(tx[0],tx[1],'^g',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^r',label='UE', linewidth='4.5')

plt.plot(xs_A,ys_A,'x',label='Rebotes subpaths')
for i in range(0,AOD_sA.size):
    plt.plot([tx[0],tx[0]+liTX_sA[i]*np.cos(AOD_sA[i])],[tx[1],tx[1]+liTX_sA[i]*np.sin(AOD_sA[i])],color=cm.jet(i/(nSubp-1)),linewidth = '0.5') 
    plt.plot([rx[0],rx[0]+liRX_sA[i]*np.cos(AOA_sfA[i])],[rx[1],rx[1]+liRX_sA[i]*np.sin(AOA_sfA[i])],color=cm.jet(i/(nSubp-1)),linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')
plt.savefig("../Figures/fitDelay_subpAD.png")


# Gráfica 5: Deck de subpaths AOD, AOA e delay non correxido
#%%
fig_ctr+=1
fig = plt.figure(fig_ctr)
nClus = clustersAD['tau'].size
nSubp = subpathsAD['tau'].size
plt.subplot(2,2,1, projection='polar',title="AoD")
for n in range(nClus):   
    AOD_1c = subpathsNAD.loc[n,:].AOD.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpathsNAD.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOD_1c)
    plt.polar(AOD_1c*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClus-1)) )
    plt.scatter(AOD_1c,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClus-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize = 7)
plt.subplot(2,2,2, projection='polar')
for n in range(nClus):  
    AOA_1cf = subpathsNAD.loc[n,:].AOA.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10(subpathsNAD.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOA_1cf)
    plt.polar(AOA_1cf*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClus-1)) )
    plt.scatter(AOA_1cf,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClus-1)),marker='+')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize=7)
plt.subplot(2,1,2)
plt.ylabel("power [dB]")
plt.xlabel("TDoA (s)")
for n in range(nClus):   
    markerline, stemlines, baseline = plt.stem( subpathsNAD.loc[n,:].tau.to_numpy() ,10*np.log10( subpathsNAD.loc[n,:].P.to_numpy() ),bottom=np.min(10*np.log10(subpathsNAD.P.to_numpy())))
    plt.setp(stemlines, color=cm.jet(n/(nClus-1)))
    plt.setp(markerline, color=cm.jet(n/(nClus-1))) 
plt.grid()

# Gráfica 6: Deck de subpaths AOD, AOA e delay correxido

fig_ctr+=1
fig = plt.figure(fig_ctr)
nClus = clustersNAD['tau'].size
plt.subplot(2,2,1, projection='polar',title="AoD")
for n in range(nClus):   
    AOD_1c = subpathsAD.loc[n,:].AOD.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpathsAD.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOD_1c)
    plt.polar(AOD_1c*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClus-1)) )
    plt.scatter(AOD_1c,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClus-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize = 7)
plt.subplot(2,2,2, projection='polar')
for n in range(nClus):  
    AOA_1cf = subpathsAD.loc[n,:].AOA.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpathsAD.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOA_1cf)
    plt.polar(AOA_1cf*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClus-1)) )
    plt.scatter(AOA_1cf,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClus-1)),marker='+')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize=7)
plt.subplot(2,1,2)
plt.ylabel("power [dB]")
plt.xlabel("TDoA (s) | %d subpaths cannot be adapted (%.2f%%)"%(np.sum(subpathsAD.Xs==np.inf),100*np.sum(subpathsAD.Xs==np.inf)/len(subpathsAD)))
for n in range(nClus):   
    markerline, stemlines, baseline = plt.stem( subpathsAD.loc[n,:].tau.to_numpy() ,10*np.log10( subpathsAD.loc[n,:].P.to_numpy() ),bottom=np.min(10*np.log10(subpathsAD.P.to_numpy())))
    plt.setp(stemlines, color=cm.jet(n/(nClus-1)))
    plt.setp(markerline, color=cm.jet(n/(nClus-1))) 
plt.grid()