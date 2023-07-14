#%% 
import numpy as np
import matplotlib.pyplot as plt
import threeGPPMultipathGenerator as mpg
import multipathChannel as mc
from matplotlib import cm
import pandas as pd

plt.close('all')
fig_ctr = 0

""" Banco de gráficas e tests exclusivo da función fitAOD (e correción de backlobes se corresponde)"""

# Posicións transmisor e receptor

tx = (0,0,10)
rx = (50,-60,1.5)
phi0 = 0

# Selección de escenario - UMi, UMa, RMa, InH-Office-Mixed, InH-Office-Open

sce = "UMi"

# ----------------------------
# ---- Canle A, largeBW = true
# ----------------------------

modelA = mpg.ThreeGPPMultipathChannelModel(scenario = sce, bLargeBandwidthOption=True)
plinfoA,macroA,clustersA,subpathsA = modelA.create_channel(tx,rx)

#AODs non correxidos
AOD_cA = clustersA['AOD'].T.to_numpy() * (np.pi/(180.0))
AOD_sA = subpathsA['AOD'].T.to_numpy() * (np.pi/(180.0))


# Adaptación canle 1:

ad_clustersA  = modelA.fitAOD(tx,rx,clustersA)
ad_subpathsA = modelA.fitAOD(tx,rx,subpathsA)

# OPCIONAL -- Se queremos ademais correxir backlobes:
#-------
#ad_clustersA = modelA.deleteBacklobes(ad_clustersA,phi0)
#ad_subpathsA = modelA.deleteBacklobes(ad_subpathsA,phi0)
#-------

#Posición dos rebotes:
xc_A,yc_A = [ad_clustersA['xloc'].T.to_numpy(),ad_clustersA['yloc'].T.to_numpy()]
xs_A,ys_A = [ad_subpathsA['xloc'].T.to_numpy(),ad_subpathsA['yloc'].T.to_numpy()]

#Distancia entre receptor e posición do rebote
liRX_cA = np.sqrt((xc_A-rx[0])**2+(yc_A - rx[1])**2)
liRX_sA = np.sqrt((xs_A-rx[0])**2+(ys_A - rx[1])**2)
#Distancia entre transmisor e receptor do rebote
liTX_cA = np.sqrt(xc_A**2+yc_A**2) 
liTX_sA = np.sqrt(xs_A**2+ys_A**2) 

#AOAs correxidos
AOD_cfA = ad_clustersA['AOD'].T.to_numpy() * (np.pi/(180.0))
AOD_sfA = ad_subpathsA['AOD'].T.to_numpy() * (np.pi/(180.0))

#AOAs e tau non varían
AOA_cA = ad_clustersA['AOA'].T.to_numpy() * (np.pi/(180.0))
AOA_sA = ad_subpathsA['AOA'].T.to_numpy() * (np.pi/(180.0))

tau_cA = ad_clustersA['tau'].T.to_numpy() * (np.pi/(180.0))
tau_sA = ad_subpathsA['tau'].T.to_numpy() * (np.pi/(180.0))


# ---- Gráfica 1, camiños non adaptados:

fig_ctr+=1
# plt.subplot(2,1,1)
fig = plt.figure(fig_ctr)
plt.title("AOD sen correxir (clusters)")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

nClus = tau_cA.size
nSubp = tau_sA.size

plt.plot(tx[0],tx[1],'^g',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^r',label='UE', linewidth='4.5')
plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
plt.plot(xc_A,yc_A,'x',label='Rebotes')
for i in range(0,AOD_cA.size): 
    plt.plot([tx[0],tx[0]+liTX_cA[i]*np.cos(AOD_cA[i])],[tx[1],tx[1]+liTX_cA[i]*np.sin(AOD_cA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.5') 
    plt.plot([rx[0],rx[0]+liRX_cA[i]*np.cos(AOA_cA[i])],[rx[1],rx[1]+liRX_cA[i]*np.sin(AOA_cA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')

# Gráfica 2 - Camiños clusters adaptados 

fig_ctr+=1
fig = plt.figure(fig_ctr)
# plt.subplot(2,1,1)
plt.title("AOD correxidos (clusters)")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

plt.plot(tx[0],tx[1],'^g',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^r',label='UE', linewidth='4.5')
plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
plt.plot(xc_A,yc_A,'x',label='Rebotes')
for i in range(0,AOD_cA.size): 
    plt.plot([tx[0],tx[0]+liTX_cA[i]*np.cos(AOD_cfA[i])],[tx[1],tx[1]+liTX_cA[i]*np.sin(AOD_cfA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.5') 
    plt.plot([rx[0],rx[0]+liRX_cA[i]*np.cos(AOA_cA[i])],[rx[1],rx[1]+liRX_cA[i]*np.sin(AOA_cA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')

# Gráfica 3 - Subpaths non adaptados

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.title("AOD non correxidos (subpaths)")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')
plt.plot(tx[0],tx[1],'^g',color='r',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^',color='g',label='UE', linewidth='4.5')

plt.plot(xs_A,ys_A,'x',label='Rebotes subpaths')
for i in range(0,AOD_sA.size):
    plt.plot([tx[0],tx[0]+liTX_sA[i]*np.cos(AOD_sA[i])],[tx[1],tx[1]+liTX_sA[i]*np.sin(AOD_sA[i])],color=cm.jet(i/(nSubp-1)),linewidth = '0.5') 
    plt.plot([rx[0],rx[0]+liRX_sA[i]*np.cos(AOA_sA[i])],[rx[1],rx[1]+liRX_sA[i]*np.sin(AOA_sA[i])],color=cm.jet(i/(nSubp-1)),linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')

# Gráfica 4 - Subpaths adaptados

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.title("AOD correxidos (subpaths)")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')
plt.plot(tx[0],tx[1],'^g',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^r',label='UE', linewidth='4.5')

plt.plot(xs_A,ys_A,'x',label='Rebotes subpaths')
for i in range(0,AOD_sA.size):
    plt.plot([tx[0],tx[0]+liTX_sA[i]*np.cos(AOD_sfA[i])],[tx[1],tx[1]+liTX_sA[i]*np.sin(AOD_sfA[i])],color=cm.jet(i/(nSubp-1)),linewidth = '0.5') 
    plt.plot([rx[0],rx[0]+liRX_sA[i]*np.cos(AOA_sA[i])],[rx[1],rx[1]+liRX_sA[i]*np.sin(AOA_sA[i])],color=cm.jet(i/(nSubp-1)),linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')


# Gráfica 5: Deck de subpaths AOD, AOA e delay non correxido

fig_ctr+=1
fig = plt.figure(fig_ctr)
nClus = ad_clustersA['tau'].size
plt.subplot(2,2,1, projection='polar',title="AoD")
for n in range(nClus):   
    AOD_1c = ad_subpathsA.loc[n,:].AOD.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( ad_subpathsA.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOD_1c)
    plt.polar(AOD_1c*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClus-1)) )
    plt.scatter(AOD_1c,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClus-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize = 7)
plt.subplot(2,2,2, projection='polar')
for n in range(nClus):  
    AOA_1cf = ad_subpathsA.loc[n,:].AOA.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10(ad_subpathsA.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOA_1cf)
    plt.polar(AOA_1cf*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClus-1)) )
    plt.scatter(AOA_1cf,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClus-1)),marker='+')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize=7)
plt.subplot(2,1,2)
plt.ylabel("power [dB]")
plt.xlabel("TDoA (s)")
for n in range(nClus):   
    markerline, stemlines, baseline = plt.stem( ad_subpathsA.loc[n,:].tau.to_numpy() ,10*np.log10( ad_subpathsA.loc[n,:].P.to_numpy() ),bottom=np.min(10*np.log10(ad_subpathsA.P.to_numpy())))
    plt.setp(stemlines, color=cm.jet(n/(nClus-1)))
    plt.setp(markerline, color=cm.jet(n/(nClus-1))) 
plt.grid()

# Gráfica 6: Deck de subpaths AOD, AOA e delay correxido

fig_ctr+=1
fig = plt.figure(fig_ctr)
nClus = clustersA['tau'].size
plt.subplot(2,2,1, projection='polar',title="AoD")
for n in range(nClus):   
    AOD_1c = subpathsA.loc[n,:].AOD.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpathsA.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOD_1c)
    plt.polar(AOD_1c*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClus-1)) )
    plt.scatter(AOD_1c,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClus-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize = 7)
plt.subplot(2,2,2, projection='polar')
for n in range(nClus):  
    AOA_1cf = subpathsA.loc[n,:].AOA.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpathsA.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOA_1cf)
    plt.polar(AOA_1cf*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClus-1)) )
    plt.scatter(AOA_1cf,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClus-1)),marker='+')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize=7)
plt.subplot(2,1,2)
plt.ylabel("power [dB]")
plt.xlabel("TDoA (s)")
for n in range(nClus):   
    markerline, stemlines, baseline = plt.stem( subpathsA.loc[n,:].tau.to_numpy() ,10*np.log10( ad_subpathsA.loc[n,:].P.to_numpy() ),bottom=np.min(10*np.log10(ad_subpathsA.P.to_numpy())))
    plt.setp(stemlines, color=cm.jet(n/(nClus-1)))
    plt.setp(markerline, color=cm.jet(n/(nClus-1))) 
plt.grid()