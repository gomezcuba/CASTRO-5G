#%% 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpi
import threeGPPMultipathGenerator as mpg
import multipathChannel as mc

from matplotlib import cm

plt.close('all')

# -------- Datos iniciais ------- #
fig_ctr=0
txPos = (0,0,10)
rxPos = (50,0,1.5)
model = mpg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=True)
plinfo,macro,clusters,subpaths = model.create_channel(txPos,rxPos)
tau,powC,AOA,AOD,ZOA,ZOD = clusters.T.to_numpy()
los, PLfree, SF = plinfo
tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths.T.to_numpy()

# 1. Con AOAs correxidos:
AOA_fixsp, xPathLoc_sp, yPathLoc_sp = model.fitAOA(txPos,rxPos,AOD_sp,tau_sp)
AOA_fix, xPathLoc, yPathLoc = model.fitAOA(txPos,rxPos,AOD,tau)
# 2. Con AODs correxidos:
#AOD_fix = model.fitAOD(txPos,rxPos,tau,AOA)
#AOD_spFix = model.fitAOD(txPos,rxPos,tau,AOA_sp)
# 3. Con delays consistentes con AOA e AOD non modificados:
tauFix = model.fitDelay(txPos,rxPos,AOD,AOA)
tau_spFix = model.fitDelay(txPos,rxPos,AOD_sp,AOA_sp)
# 4. Canle con xeración aleatoria de tau, AOD ou AOA
prob = (0.5,0.2,0.3)
#TODO mais tarde

# ------------------------------- #

#1. Plot ubicación da BS e do user (UE) 
#1.1 - Plot dataset completo non correxido - subplot1
#1.2 - Plot dataset completo correxido - subplot2



txPos2D = txPos[0:-1]
rxPos2D = rxPos[0:-1]

AOA_r = AOA*(np.pi/180)
AOA_rF = AOA_fix*(np.pi/180)
AOD_r = AOD*(np.pi/180)
liRX = np.sqrt((xPathLoc-rxPos[0])**2+(yPathLoc - rxPos[1])**2)
liTX = np.sqrt(xPathLoc**2+yPathLoc**2)
li = liTX + liRX

scaleguide = np.max(np.abs(np.concatenate([yPathLoc,xPathLoc],0)))

Npoints = 1001
rg = np.linspace(0,1,200)

tau0=np.linalg.norm(rxPos2D)/3e8
PathLoc=np.vstack([xPathLoc,yPathLoc])
tau_fromloc = (np.linalg.norm(PathLoc,axis=0)+np.linalg.norm(PathLoc.T-rxPos2D,axis=1))/3e8 - tau0
print("dif of taus (ns). :\n %s"%(1e9 * (tau_fromloc-tau)))

# Representación
fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.title("AOA sen correxir")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')
plt.plot(txPos2D[0],txPos2D[1],'^g',color='b',label='BS',linewidth = '4.5')
plt.plot(rxPos2D[0],rxPos2D[1],'^',color='r',label='UE', linewidth='4.5')
plt.plot([txPos2D[0],rxPos2D[0]],[txPos2D[1],rxPos2D[1]],'--',color='g',label='LOS')
plt.plot(xPathLoc,yPathLoc,'x',color='y',label='Rebotes')

for i in range(0,AOD.size): 

    plt.plot([txPos2D[0],xPathLoc[i]],[txPos2D[1],yPathLoc[i]],'k',color = 'blue',linewidth = '0.5') 
    plt.plot([rxPos2D[0],rxPos2D[0]+liRX[i]*np.cos(AOA_r[i])],[rxPos2D[1],rxPos2D[1]+liRX[i]*np.sin(AOA_r[i])],'k',linewidth = '0.5')


legend = plt.legend(shadow=True, fontsize='10')

xlim = (1,2)

fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.title("AOA correxidos")
plt.grid(linestyle = '--')
plt.xlabel('x-location (m)')
plt.ylabel('y-location (m)')
plt.plot(txPos2D[0],txPos2D[1],'^g',color='b',label='BS',linewidth = '4.5')
plt.plot(rxPos2D[0],rxPos2D[1],'^',color='r',label='UE', linewidth='4.5')
plt.plot([txPos2D[0],rxPos2D[0]],[txPos2D[1],rxPos2D[1]],'--',color='g',label='LOS')
plt.plot(xPathLoc,yPathLoc,'x',color='y',label='Rebotes')

for i in range(0,AOD.size): 
    plt.plot([txPos2D[0],xPathLoc[i]],[txPos2D[1],yPathLoc[i]],'k',color = 'blue',linewidth = '0.5') 
    plt.plot([rxPos2D[0],rxPos2D[0]+liRX[i]*np.cos(AOA_rF[i])],[rxPos2D[1],rxPos2D[1]+liRX[i]*np.sin(AOA_rF[i])],'k',linewidth = '0.5')
legend = plt.legend(shadow=True, fontsize='10')


fig_ctr+=1
fig = plt.figure(fig_ctr)
nClusters = tau.size
plt.subplot(2,2,1, projection='polar',title="AoD")
for n in range(nClusters):   
    AOD_1c = subpaths.loc[n,:].AOD.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpaths.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOD_1c)
    plt.polar(AOD_1c*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClusters-1)) )
    plt.scatter(AOD_1c,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClusters-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'])
plt.subplot(2,2,2, projection='polar',title="AoA sen corrixir")
for n in range(nClusters):  
    AOA_1c = subpaths.loc[n,:].AOA.to_numpy() *np.pi/180
    pathAmplitudesdBtrunc25_1c = np.maximum(10*np.log10( subpaths.loc[n,:].P.to_numpy()  ),-45)
    Nsp=len(AOA_1c)
    plt.polar(AOA_1c*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_1c]),':',color=cm.jet(n/(nClusters-1)) )
    plt.scatter(AOA_1c,pathAmplitudesdBtrunc25_1c,color=cm.jet(n/(nClusters-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'])
plt.subplot(2,1,2)
for n in range(nClusters):   
    markerline, stemlines, baseline = plt.stem( subpaths.loc[n,:].tau.to_numpy() ,10*np.log10( subpaths.loc[n,:].P.to_numpy() ),bottom=np.min(10*np.log10(subpaths.P.to_numpy())))
    plt.setp(stemlines, color=cm.jet(n/(nClusters-1)))
    plt.setp(markerline, color=cm.jet(n/(nClusters-1))) 

# --- ArrayPolar ---
# 2.1 - Representación da orientación dos AOAs - non correxidos
# 2.2 - Representación AOAs_fixed

# 3.1 - Diagrama de antena con backlobes non descartados
# 3.2 - Diagrama de antena con correción de backlobes
#%%

fig_ctr+=1
fig = plt.figure(fig_ctr)
nClusters = tau.size
if los:
    M=max(subpaths.loc[0,:].index)
    (tau_los,pow_los,losAoA,losAoD,losZoA,losZoD)=subpaths.loc[(0,M),:]
    tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths.drop((0,M)).T.to_numpy()
else:    
    M=max(subpaths.loc[0,:].index)+1
    tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths.T.to_numpy()

plt.subplot(2,2,1, projection='polar',title="AoD")
Nsp=AOD_sp.shape[1]
if los:
    plt.polar(losAoD*np.pi/180*np.ones((2,1)),np.vstack([[-40],10*np.log10(pow_los)]),':',color=cm.jet(0))
    plt.scatter(losAoD*np.pi/180*np.ones((2,1)),np.vstack([[-40],10*np.log10(pow_los)]),color=cm.jet(0),marker='<')
for n in range(nClusters):   
    pathAmplitudes_sp = np.sqrt( pow_sp[n,:] )*np.exp(2j*np.pi*np.random.rand(Nsp))
    pathAmplitudesdBtrunc25_sp = np.maximum(10*np.log10(np.abs(pathAmplitudes_sp)**2),-45)
    plt.polar(AOD_sp[n,:]*np.pi/180*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_sp]),':',color=cm.jet(n/(nClusters-1)) )
    plt.scatter(AOD_sp[n,:]*np.pi/180,pathAmplitudesdBtrunc25_sp,color=cm.jet(n/(nClusters-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'])
plt.subplot(2,2,2, projection='polar',title="AoA sen corrixir")
if los:
    plt.polar(losAoA*np.pi/180*np.ones((2,1)),np.vstack([[-40],10*np.log10(pow_los)]),':',color=cm.jet(0))
    plt.scatter(losAoA*np.pi/180*np.ones((2,1)),np.vstack([[-40],10*np.log10(pow_los)]),color=cm.jet(0),marker='<')
for n in range(nClusters):   
    pathAmplitudes_sp = np.sqrt( pow_sp[n,:] )*np.exp(2j*np.pi*np.random.rand(Nsp))
    pathAmplitudesdBtrunc25_sp = np.maximum(10*np.log10(np.abs(pathAmplitudes_sp)**2),-45)
    plt.polar(AOA_sp[n,:]*np.pi/180*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_sp]),':',color=cm.jet(n/(nClusters-1)) )
    plt.scatter(AOA_sp[n,:]*np.pi/180,pathAmplitudesdBtrunc25_sp,color=cm.jet(n/(nClusters-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'])
plt.subplot(2,1,2)
if los:
    markerline, stemlines, baseline = plt.stem(0,10*np.log10(pow_los),bottom=np.min(10*np.log10(pow_sp)))
    plt.setp(stemlines, color=cm.jet(0))
    plt.setp(markerline, color=cm.jet(0)) 
for n in range(nClusters):   
    markerline, stemlines, baseline = plt.stem(tau_sp[n,:],10*np.log10(pow_sp[n,:]),bottom=np.min(10*np.log10(pow_sp)))
    plt.setp(stemlines, color=cm.jet(n/(nClusters-1)))
    plt.setp(markerline, color=cm.jet(n/(nClusters-1))) 
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
