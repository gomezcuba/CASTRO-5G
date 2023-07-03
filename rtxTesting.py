#%% 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpi
import threeGPPMultipathGenerator as mpg
import multipathChannel as mc

from matplotlib import cm

# -------- Datos iniciais ------- #
fig_ctr=0
txPos = (0,0,10)
rxPos = (25,25,1.5)
model = mpg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=False)
plinfo,macro,clusters,subpaths = model.create_channel(txPos,rxPos)
tau,powC,AOA,AOD,ZOA,ZOD = clusters.T.to_numpy()
los, PLfree, SF = plinfo
tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths.T.to_numpy()

# 1. Con AOAs correxidos:
AOA_spFix = model.fitAOA(txPos,rxPos,AOD_sp,tau_sp)
AOA_fix = model.fitAOA(txPos,rxPos,AOD,tau)
# 2. Con AODs correxidos:
AOD_fix = model.fitAOD(txPos,rxPos,)
AOD_spFix = model.fitAOD(txPos,rxPos,)
# 3. Con delays consistentes con AOA e AOD non modificados:
tauFix = model.fitDelay(txPos,rxPos,)
tau_spFix = model.fitDelay(txPos,rxPos,)
# 4. Canle con xeración aleatoria de tau, AOD ou AOA
prob = (0.5,0.2,0.3)
#TODO mais tarde

# ------------------------------- #

#%%

#1. Plot ubicación da BS e do user (UE) 
#1.1 - Plot dataset completo non correxido - subplot1
#1.2 - Plot dataset completo correxido - subplot2

fig_ctr+=1

txPos2D = txPos[0:-1]
rxPos2D = rxPos[0:-1]
Npoints = 1001

# Representación
fig = plt.figure(fig_ctr)
plt.title("Ubicación de UE y BS")
plt.plot(txPos2D,'b',color='r',label='BS')
plt.plot(rxPos2D,'b')
plt.grid(linestyle = '--')
plt.xlim(-50,50)
plt.xlabel('x-location (m)')
plt.xlabel('y-location (m)')
plt.ylim(-50,50)


# %%
# --- ArrayPolar ---
# 2.1 - Representación da orientación dos AOAs - non correxidos
# 2.2 - Representación AOAs_fixed

# 3.1 - Diagrama de antena con backlobes non descartados
# 3.2 - Diagrama de antena con correción de backlobes

model = mpg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=False)
plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
tau,powC,AOA,AOD,ZOA,ZOD = clusters.T.to_numpy()
los, PLfree, SF = plinfo
nClusters = tau.size
if los:
    M=max(subpaths.loc[0,:].index)
    (tau_los,pow_los,losAoA,losAoD,losZoA,losZoD)=subpaths.loc[(0,M),:]
    tau_sp,powC_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths.drop((0,M)).T.to_numpy()
else:    
    M=max(subpaths.loc[0,:].index)+1
    tau_sp,powC_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths.T.to_numpy()
    AOA_spFix = model.fitAOA((0,0,10),(40,0,1.5),AOD_sp,tau_sp)
    #zsubpaths = model.fixAOAConsistency(subpaths, AOA_spFix)
    #tau_sp,powC_sp,AOA_spFix,AOD_sp,ZOA_sp,ZOD_sp = subpaths.T.to_numpy()
    
tau_sp=tau_sp.reshape(nClusters,-1)
powC_sp=powC_sp.reshape(nClusters,-1)
AOA_sp=AOA_sp.reshape(nClusters,-1)
AOA_spFix=AOA_spFix.reshape(nClusters,-1)
AOD_sp=AOD_sp.reshape(nClusters,-1)
ZOA_sp=ZOA_sp.reshape(nClusters,-1)
ZOD_sp=ZOD_sp.reshape(nClusters,-1)
plt.close('all')
fig_ctr=0

#2D polar plots of AoA
AoAs = AOA_sp.reshape(-1)*np.pi/180#radians
AoAs_fix = AOA_spFix.reshape(-1)*np.pi/180
Npath=np.size(AoAs)
pathAmplitudes = np.sqrt( powC_sp.reshape(-1) )*np.exp(2j*np.pi*np.random.rand(Npath))

#%%
#plot of rx AoAs and channel gains - Comparison between fixed and not
fig_ctr+=1
fig = plt.figure(fig_ctr)
pathAmplitudesdBtrunc25 = np.maximum(10*np.log10(np.abs(pathAmplitudes)**2),-45)

plt.polar(AoAs*np.ones((2,1)),np.vstack([-40*np.ones((1,Npath)),pathAmplitudesdBtrunc25]),':')
plt.scatter(AoAs,pathAmplitudesdBtrunc25,color='k',marker='x')
Nsp=AOA_sp.shape[1]

if los:
    plt.polar(losAoA*np.pi/180*np.ones((2,1)),np.vstack([[-40],10*np.log10(pow_los)]),':',color=cm.jet(0))
    plt.scatter(losAoA*np.pi/180*np.ones((2,1)),np.vstack([[-40],10*np.log10(pow_los)]),color=cm.jet(0),marker='<')
for n in range(nClusters):   
    pathAmplitudes_sp = np.sqrt( powC_sp[n,:] )*np.exp(2j*np.pi*np.random.rand(Nsp))
    pathAmplitudesdBtrunc25_sp = np.maximum(10*np.log10(np.abs(pathAmplitudes_sp)**2),-45)
    plt.polar(AOA_spFix[n,:]*np.pi/180*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_sp]),':',color=cm.jet(n/(nClusters-1)) )
    plt.scatter(AOA_spFix[n,:]*np.pi/180,pathAmplitudesdBtrunc25_sp,color=cm.jet(n/(nClusters-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'])
#%%
# compute the response of the antenna array with Nant antennas
Nant = 16
AntennaResponses =mc.fULA(AoAs_fix,Nant)
Npointsplot=1001
# compute the "beamforming vector". This vector is multiplied by the "response" when we want to receive from the desired angle
angles_plot = np.linspace(0,2*np.pi,Npointsplot)
BeamformingVectors =mc.fULA(angles_plot,Nant)

#plot of receive array response of first path
fig_ctr+=1
fig = plt.figure(fig_ctr)
arrayGain1Path=(BeamformingVectors.transpose([0,2,1]).conj()@AntennaResponses[0,:,:]).reshape(-1)
arrayGain1PathdBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(arrayGain1Path)**2),-25)

plt.polar(angles_plot,arrayGain1PathdBtrunc25)
plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])

Nsp=AOA_sp.shape[1]

#%%
#plot of receive array response of ALL paths in SEPARATE LINES, WITHOUT the effect of power
fig_ctr+=1
fig = plt.figure(fig_ctr)
arrayGainAllPaths=(AntennaResponses.transpose([0,2,1]).conj()@BeamformingVectors[:,None,:,:]).reshape((Npointsplot,Npath))

arrayGainAllPathsdBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(arrayGainAllPaths)**2),-25)

plt.polar(angles_plot,arrayGainAllPathsdBtrunc25)
plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])

#%%
#plot of receive array response of ALL paths in SEPARATE LINES, WITH the effect of power
fig_ctr+=1
fig = plt.figure(fig_ctr)

channelArrayGainAllPaths =  arrayGainAllPaths*pathAmplitudes 
channelArrayGainAllPathsdBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(channelArrayGainAllPaths)**2),-25)

plt.polar(angles_plot,channelArrayGainAllPathsdBtrunc25)
plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])

#plot of receive array response of 5 STRONGEST PATHS ONLY, in SEPARATE LINES, WITH the effect of power
fig_ctr+=1
fig = plt.figure(fig_ctr)
Nbig = 5 # can also use something like np.sum(np.abs(pathAmplitudes)**2>1e-2) to calculate the number of paths greater than 0.01
sortIndices = np.argsort(np.abs(pathAmplitudes)**2)

channelArrayGainBigPaths =  arrayGainAllPaths[:,sortIndices[-Nbig:]]*pathAmplitudes [sortIndices[-Nbig:]]
channelArrayGainBigPathsdBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(channelArrayGainBigPaths)**2),-35)

plt.polar(angles_plot,channelArrayGainBigPathsdBtrunc25)
plt.polar(AoAs[sortIndices[-Nbig:]]*np.ones((2,1)),np.vstack([-35*np.ones((1,Nbig)),10*np.log10(np.abs(pathAmplitudes [sortIndices[-Nbig:]])**2)]),':')
plt.scatter(AoAs[sortIndices[-Nbig:]],10*np.log10(np.abs(pathAmplitudes [sortIndices[-Nbig:]])**2),color='k',marker='x')
plt.yticks(ticks=[-30,-20,-10,0],labels=['-20dB','-30dB','-10dB','0dB'])


#plot of receive array response of ALL paths COMBINED
fig_ctr+=1
fig = plt.figure(fig_ctr)

arrayResponseCombined = np.sum( arrayGainAllPaths*pathAmplitudes , axis=1)
arrayResCondBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(arrayResponseCombined)**2),-25)

plt.polar(angles_plot,arrayResCondBtrunc25)
plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])




# %%
