#%% 
import numpy as np
import matplotlib.pyplot as plt
import threeGPPMultipathGenerator as mpg
import multipathChannel as mc
from matplotlib import cm
import pandas as pd

plt.close('all')
fig_ctr = 0

""" Banco de gráficas e tests exclusivo da función fitAOA (e correción de backlobes se corresponde)"""

# Posicións transmisor e receptor

tx = (0,0,10)
rx = (40,30,1.5)
phi0 = np.random.uniform(0,2*np.pi)

# Selección de escenario - UMi, UMa, RMa, InH-Office-Mixed, InH-Office-Open

sce = "RMa"

# ----------------------------
# ---- Canle A, largeBW = true
# ----------------------------

modelA = mpg.ThreeGPPMultipathChannelModel(scenario = sce, bLargeBandwidthOption=True)
plinfoA,macroA,clustersA,subpathsA = modelA.create_channel(tx,rx)

#AOAs non correxidos
AOA_cA = clustersA['AOA'].T.to_numpy() * (np.pi/(180.0))
AOA_sA = subpathsA['AOA'].T.to_numpy() * (np.pi/(180.0))


# Adaptación canle 1:

ad_clustersA  = modelA.fitAOA(tx,rx,clustersA)
ad_subpathsA = modelA.fitAOA(tx,rx,subpathsA)

# Se queremos ademais correxir backlobes:

ad_clustersA = modelA.deleteBacklobes(ad_clustersA,phi0)

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
AOA_cfA = ad_clustersA['AOA'].T.to_numpy() * (np.pi/(180.0))
AOA_sfA = ad_subpathsA['AOA'].T.to_numpy() * (np.pi/(180.0))

#AODs e tau non varían
AOD_cA = clustersA['AOD'].T.to_numpy() * (np.pi/(180.0))
AOD_sA = subpathsA['AOD'].T.to_numpy() * (np.pi/(180.0))

tau_cA = clustersA['tau'].T.to_numpy() * (np.pi/(180.0))
tau_sA = subpathsA['tau'].T.to_numpy() * (np.pi/(180.0))


# -----------------------------
# ---- Canle B, largeBW = false
# -----------------------------

modelB = mpg.ThreeGPPMultipathChannelModel(scenario = sce, bLargeBandwidthOption=False)
plinfoB,macroB,clustersB,subpathsB = modelB.create_channel(tx,rx)

# Adaptación canle 2:

ad_clustersB = modelB.fitAOA(tx,rx,clustersB)
ad_subpathsB = modelB.fitAOA(tx,rx,subpathsB)

#Posición dos rebotes:
xc_B,yc_B = [ad_clustersB['xloc'].T.to_numpy(),ad_clustersB['yloc'].T.to_numpy()]
xs_B,ys_B = [ad_subpathsB['xloc'].T.to_numpy(),ad_subpathsB['yloc'].T.to_numpy()]

#Distancia entre receptor e posición do rebote
liRX_cB = np.sqrt((xc_B-rx[0])**2+(yc_B - rx[1])**2)
liRX_sB = np.sqrt((xs_B-rx[0])**2+(xs_B - rx[1])**2)
#Distancia entre transmisor e receptor do rebote
liTX_cB = np.sqrt(xc_B**2+yc_B**2) 
liTX_sB = np.sqrt(xs_B**2+ys_B**2) 

AOA_cfB = ad_clustersB['AOA'].T.to_numpy() * (np.pi/(180.0))
AOA_sfB = ad_subpathsB['AOA'].T.to_numpy() * (np.pi/(180.0))

AOD_cB = ad_clustersB['AOD'].T.to_numpy() * (np.pi/(180.0))
AOD_sB = ad_subpathsB['AOD'].T.to_numpy() * (np.pi/(180.0))



# ---- Gráfica 1, camiños non adaptados:

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.title("AOA sen correxir (clusters)")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

nClus = tau_cA.size
nSubp = tau_sA.size

plt.plot(tx[0],tx[1],'^g',color='r',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^',color='g',label='UE', linewidth='4.5')
plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
plt.plot(xc_A,yc_A,'x',label='Rebotes')
for i in range(0,AOD_cA.size): 
    plt.plot([tx[0],tx[0]+liTX_cA[i]*np.cos(AOD_cA[i])],[tx[1],tx[1]+liTX_cA[i]*np.sin(AOD_cA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.5') 
    plt.plot([rx[0],rx[0]+liRX_cA[i]*np.cos(AOA_cA[i])],[rx[1],rx[1]+liRX_cA[i]*np.sin(AOA_cA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')

# Gráfica 2 - Camiños clusters adaptados 

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.title("AOA correxidos (clusters)")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')

plt.plot(tx[0],tx[1],'^g',color='r',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^',color='g',label='UE', linewidth='4.5')
plt.plot([tx[0],rx[0]],[tx[1],rx[1]],'--')
plt.plot(xc_A,yc_A,'x',label='Rebotes')
for i in range(0,AOD_cA.size): 
    plt.plot([tx[0],tx[0]+liTX_cA[i]*np.cos(AOD_cA[i])],[tx[1],tx[1]+liTX_cA[i]*np.sin(AOD_cA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.5') 
    plt.plot([rx[0],rx[0]+liRX_cA[i]*np.cos(AOA_cfA[i])],[rx[1],rx[1]+liRX_cA[i]*np.sin(AOA_cfA[i])],color=cm.jet(i/(nClus-1)),linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')

# Gráfica 3 - camiños subpaths adaptados

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.title("AOA correxidos (subpaths)")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')
plt.plot(tx[0],tx[1],'^g',color='r',label='BS',linewidth = '4.5')
plt.plot(rx[0],rx[1],'^',color='g',label='UE', linewidth='4.5')

plt.plot(xs_A,ys_A,'x',label='Rebotes subpaths')
for i in range(0,AOD_sA.size):
    plt.plot([tx[0],tx[0]+liTX_sA[i]*np.cos(AOD_sA[i])],[tx[1],tx[1]+liTX_sA[i]*np.sin(AOD_sA[i])],color=cm.jet(i/(nSubp-1)),linewidth = '0.5') 
    plt.plot([rx[0],rx[0]+liRX_sA[i]*np.cos(AOA_sfA[i])],[rx[1],rx[1]+liRX_sA[i]*np.sin(AOA_sfA[i])],color=cm.jet(i/(nSubp-1)),linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')

# Gráfica 4: Potencia subpaths:
fig_ctr+=1
fig = plt.figure(fig_ctr)
nSubp = tau_sA.size

for n in range(nClus):   
    AOD_1c = subpathsA.loc[n,:].AOD.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpathsA.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOD_1c)
    plt.polar(AOD_1c*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClus-1)) )
    plt.scatter(AOD_1c,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClus-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'],fontsize = 7)

# %%
