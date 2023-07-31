#%% 
import numpy as np
import matplotlib.pyplot as plt
import threeGPPMultipathGenerator as mpg
import multipathChannel as mc
from matplotlib import cm
import pandas as pd
import os

plt.close('all')
fig_ctr = 0

""" Testbench for fitAOA and backlobe correction functions """
phi0 = 0
# Selección de escenario
sce = "UMa"
# Posicións transmisor e receptor
tx = (0,0,10)
rx = (45,20,1.5)
#Creación de canle 
modelA = mpg.ThreeGPPMultipathChannelModel(scenario = "UMi", bLargeBandwidthOption=True)
#Xeramos clusters e subpaths non adaptados:
plinfoA,macroA,clustersNAD,subpathsNAD = modelA.create_channel(tx,rx)
#Creamos unha copia dos clusters e subpaths, e adaptámola:
clustersAD, subpathsAD = [clustersNAD.copy(), subpathsNAD.copy()]
clustersAD = modelA.fitAOA(tx,rx,clustersAD)
subpathsAD = modelA.fitAOA(tx,rx,subpathsAD)

#AOAs non correxidos
AOA_cA = clustersNAD['AOA'].T.to_numpy() * (np.pi/(180.0))
AOA_sA = subpathsNAD['AOA'].T.to_numpy() * (np.pi/(180.0))

clustersAD = clustersNAD.copy()
subpathsAD = subpathsNAD.copy()


# Adaptación canle 1:

clustersAD  = modelA.fitAOA(tx,rx,clustersAD)
subpathsAD = modelA.fitAOA(tx,rx,subpathsAD)

#Se queremos ademais correxir backlobes:

clustersADb = modelA.deleteBacklobes(tx,rx,clustersAD,phi0)
subpathsADb = modelA.deleteBacklobes(tx,rx,subpathsAD,phi0)

#Posición dos rebotes:
xc_A,yc_A = [clustersAD['xloc'].T.to_numpy(),clustersAD['yloc'].T.to_numpy()]
xs_A,ys_A = [subpathsAD['xloc'].T.to_numpy(),subpathsAD['yloc'].T.to_numpy()]

xc_Ab,yc_Ab = [clustersADb['xloc'].T.to_numpy(),clustersADb['yloc'].T.to_numpy()]
xs_Ab,ys_Ab = [subpathsADb['xloc'].T.to_numpy(),subpathsADb['yloc'].T.to_numpy()]


#Distancia entre receptor e posición do rebote
liRX_cA = np.sqrt((xc_A-rx[0])**2+(yc_A - rx[1])**2)
liRX_sA = np.sqrt((xs_A-rx[0])**2+(ys_A - rx[1])**2)
#Distancia entre transmisor e receptor do rebote
liTX_cA = np.sqrt(xc_A**2+yc_A**2) 
liTX_sA = np.sqrt(xs_A**2+ys_A**2) 

#Distancia entre receptor e posición do rebote
liRX_cAb = np.sqrt((xc_Ab-rx[0])**2+(yc_Ab - rx[1])**2)
liRX_sAb = np.sqrt((xs_Ab-rx[0])**2+(ys_Ab - rx[1])**2)
#Distancia entre transmisor e receptor do rebote
liTX_cAb = np.sqrt(xc_Ab**2+yc_Ab**2) 
liTX_sAb = np.sqrt(xs_Ab**2+ys_Ab**2) 

#Caso A: Sen correción de backlobes

#AOAs correxidos
AOA_cfA = clustersAD['AOA'].T.to_numpy() * (np.pi/(180.0))
AOA_sfA = subpathsAD['AOA'].T.to_numpy() * (np.pi/(180.0))

#AODs e tau non varían
AOD_cA = clustersAD['AOD'].T.to_numpy() * (np.pi/(180.0))
AOD_sA = subpathsAD['AOD'].T.to_numpy() * (np.pi/(180.0))

tau_cA = clustersAD['tau'].T.to_numpy() * (np.pi/(180.0))
tau_sA = subpathsAD['tau'].T.to_numpy() * (np.pi/(180.0))

#Caso B: Correción de backlobes

#AOAs correxidos
AOA_cfAb = clustersADb['AOA'].T.to_numpy() * (np.pi/(180.0))
AOA_sfAb = subpathsADb['AOA'].T.to_numpy() * (np.pi/(180.0))

#AODs e tau non varían
AOD_cAb = clustersADb['AOD'].T.to_numpy() * (np.pi/(180.0))
AOD_sAb = subpathsADb['AOD'].T.to_numpy() * (np.pi/(180.0))

tau_cAb = clustersADb['tau'].T.to_numpy() * (np.pi/(180.0))
tau_sAb = subpathsADb['tau'].T.to_numpy() * (np.pi/(180.0))

# ---- Gráfica 1, camiños non adaptados:

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

nClus = tau_cA.size
nClusb = tau_cAb.size
nSubp = tau_sA.size

plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
plt.plot(xc_A,yc_A,'x',label='Rebotes')
for i in range(0,AOD_cA.size): 
    plt.plot([tx[0],tx[0]+liTX_cA[i]*np.cos(AOD_cA[i])],[tx[1],tx[1]+liTX_cA[i]*np.sin(AOD_cA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.9') 
    plt.plot([rx[0],rx[0]+liRX_cA[i]*np.cos(AOA_cA[i])],[rx[1],rx[1]+liRX_cA[i]*np.sin(AOA_cA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.9')
plt.plot(tx[0],tx[1],'^g',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^r',label='UE', linewidth='4.5')
legend = plt.legend(shadow=True, fontsize='10')

ruta = os.path.join("img", "fitAOA_clusNAD.png")
plt.savefig(ruta)


#Gráfica 2 - Camiños clusters adaptados 

fig_ctr+=1
fig = plt.figure(fig_ctr)

plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
plt.plot(xc_A,yc_A,'x',label='Rebotes')
for i in range(0,AOD_cA.size): 
    plt.plot([tx[0],tx[0]+liTX_cA[i]*np.cos(AOD_cA[i])],[tx[1],tx[1]+liTX_cA[i]*np.sin(AOD_cA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.9') 
    plt.plot([rx[0],rx[0]+liRX_cA[i]*np.cos(AOA_cfA[i])],[rx[1],rx[1]+liRX_cA[i]*np.sin(AOA_cfA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.9')
plt.plot(tx[0],tx[1],'^g',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^r',label='UE', linewidth='4.5')
legend = plt.legend(shadow=True, fontsize='10')

ruta = os.path.join("img", "fitAOA_nbclusAD.png")
plt.savefig(ruta)

#Gráfica 2 - Camiños clusters adaptados con corrección de backlobes

fig_ctr+=1
fig = plt.figure(fig_ctr)

plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
plt.plot(xc_Ab,yc_Ab,'x',label='Rebotes clusters')
for i in range(0,AOD_cAb.size): 
    plt.plot([tx[0],tx[0]+liTX_cAb[i]*np.cos(AOD_cAb[i])],[tx[1],tx[1]+liTX_cAb[i]*np.sin(AOD_cAb[i])],color='red',linewidth = '0.9') 
    plt.plot([rx[0],rx[0]+liRX_cAb[i]*np.cos(AOA_cfAb[i])],[rx[1],rx[1]+liRX_cAb[i]*np.sin(AOA_cfAb[i])],color='blue',linewidth = '0.9')
plt.plot(tx[0],tx[1],'^g',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^r',label='UE', linewidth='4.5')
legend = plt.legend(shadow=True, fontsize='10')

ruta = os.path.join("img", "fitAOA_bclusAD.png")
plt.savefig(ruta)

# Gráfica 3 - Subpaths non adaptados

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')
plt.plot(tx[0],tx[1],'^g',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^r',label='UE', linewidth='4.5')

plt.plot(xs_A,ys_A,'x',label='Rebotes subpaths')
for i in range(0,AOD_sA.size):
    plt.plot([tx[0],tx[0]+liTX_sA[i]*np.cos(AOD_sA[i])],[tx[1],tx[1]+liTX_sA[i]*np.sin(AOD_sA[i])],color=cm.jet(i/(nSubp-1)),linewidth = '0.5') 
    plt.plot([rx[0],rx[0]+liRX_sA[i]*np.cos(AOA_sA[i])],[rx[1],rx[1]+liRX_sA[i]*np.sin(AOA_sA[i])],color=cm.jet(i/(nSubp-1)),linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')

ruta = os.path.join("img", "fitAOA_subpNAD.png")
plt.savefig(ruta)

# Gráfica 4 - Subpaths adaptados

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')
plt.plot(tx[0],tx[1],'^g',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^r',label='UE', linewidth='4.5')

plt.plot(xs_A,ys_A,'x',label='Rebotes subpaths')
for i in range(0,AOD_sA.size):
    plt.plot([tx[0],tx[0]+liTX_sA[i]*np.cos(AOD_sA[i])],[tx[1],tx[1]+liTX_sA[i]*np.sin(AOD_sA[i])],color=cm.jet(i/(nSubp-1)),linewidth = '0.5') 
    plt.plot([rx[0],rx[0]+liRX_sA[i]*np.cos(AOA_sfA[i])],[rx[1],rx[1]+liRX_sA[i]*np.sin(AOA_sfA[i])],color=cm.jet(i/(nSubp-1)),linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')

ruta = os.path.join("img", "fitAOA_nbsubpAD.png")
plt.savefig(ruta)

#subpaths con backlobe correction
fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')
plt.plot(tx[0],tx[1],'^g',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^r',label='UE', linewidth='4.5')

plt.plot(xs_Ab,ys_Ab,'x',label='Rebotes subpaths')
for i in range(0,AOD_sAb.size):
    plt.plot([tx[0],tx[0]+liTX_sAb[i]*np.cos(AOD_sAb[i])],[tx[1],tx[1]+liTX_sAb[i]*np.sin(AOD_sAb[i])],color='red',linewidth = '0.5') 
    plt.plot([rx[0],rx[0]+liRX_sAb[i]*np.cos(AOA_sfAb[i])],[rx[1],rx[1]+liRX_sAb[i]*np.sin(AOA_sfAb[i])],color='blue',linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')

ruta = os.path.join("img", "fitAOA_bsubpAD.png")
plt.savefig(ruta)


# Gráfica 5: Deck de subpaths AOD, AOA e delay non correxido

fig_ctr+=1
fig = plt.figure(fig_ctr)
nClus = clustersNAD['tau'].size
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
    markerline, stemlines, baseline = plt.stem( subpathsNAD.loc[n,:].tau.to_numpy() ,10*np.log10( subpathsAD.loc[n,:].P.to_numpy() ),bottom=np.min(10*np.log10(subpathsAD.P.to_numpy())))
    plt.setp(stemlines, color=cm.jet(n/(nClus-1)))
    plt.setp(markerline, color=cm.jet(n/(nClus-1))) 
plt.grid()

ruta = os.path.join("img", "fitAOA_decknoAD.png")
plt.savefig(ruta)


# Gráfica 6: Deck de subpaths AOD, AOA e delay correxido

fig_ctr+=1
fig = plt.figure(fig_ctr)
nSubpb = subpathsADb['tau'].size
plt.subplot(2,2,1, projection='polar',title="AoD")

AOD_1c = subpathsADb['AOD'].to_numpy() *np.pi/180
pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpathsADb['P'].to_numpy()  ),-45)
Nsp=len(AOD_1c)
plt.polar(AOD_1c,pathAmplitudesdBtrunc25_1c,':',color='red' )
plt.scatter(AOD_1c,pathAmplitudesdBtrunc25_1c,color='red',marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize = 7)
plt.subplot(2,2,2, projection='polar')


AOA_1cf = subpathsADb['AOA'].to_numpy() *np.pi/180
pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpathsADb['P'].to_numpy()  ),-45)
Nsp=len(AOA_1cf)
plt.polar(AOA_1cf,pathAmplitudesdBtrunc25_1c,':',color='blue' )
plt.scatter(AOA_1cf,pathAmplitudesdBtrunc25_1c,color='blue',marker='+')

plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize=7)
plt.subplot(2,1,2)
plt.ylabel("power [dB]")
plt.xlabel("TDoA (s)")

markerline, stemlines, baseline = plt.stem( subpathsADb['tau'].to_numpy() ,10*np.log10( subpathsADb['P'].to_numpy() ),bottom=np.min(10*np.log10(subpathsADb['P'].to_numpy())))
plt.setp(stemlines, color='green')
plt.setp(markerline, color='orange') 
plt.grid()

ruta = os.path.join("img", "fitAOA_bdeckAD.png")
plt.savefig(ruta)

#Gráfica 7: deck de subpaths con backlobe correction

fig_ctr+=1
fig = plt.figure(fig_ctr)
nClus = clustersAD['tau'].size
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
plt.xlabel("TDoA (s)")
for n in range(nClus):   
    markerline, stemlines, baseline = plt.stem( subpathsAD.loc[n,:].tau.to_numpy() ,10*np.log10( subpathsAD.loc[n,:].P.to_numpy() ),bottom=np.min(10*np.log10(subpathsAD.P.to_numpy())))
    plt.setp(stemlines, color=cm.jet(n/(nClus-1)))
    plt.setp(markerline, color=cm.jet(n/(nClus-1))) 
plt.grid()

ruta = os.path.join("img", "fitAOA_nbdeckAD.png")
plt.savefig(ruta)

# %%
