#%% 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import threeGPPMultipathGenerator as mpg
import multipathChannel as mc

from matplotlib import cm

""" Probas integración plotchan e plotArrayChan """
#TODO - poñer título as gráficas, integrar fitAOA unha vez resolto sympy


""" DelayChan """

fig_ctr=0

# Creamos canal básico - BS en (0,0) e user en (25,25)
txPos = (0,0,10)
rxPos = (25,25,1.5)
model = mpg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=False)
plinfo,macro,clusters,subpaths = model.create_channel(txPos,rxPos)
tau,powC,AOA,AOD,ZOA,ZOD = clusters.T.to_numpy()
los, PLfree, SF = plinfo
tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths.T.to_numpy()

#Plot ubicación da BS e do user (UE)


#2D plots of power vs delay

#plots of continuous time channels (deltas)
delays = tau_sp*1e9 #ns
Npath=np.size(delays)
pathAmplitudes = np.sqrt(pow_sp )*np.exp(2j*np.pi*np.random.rand(Npath))

# Figure 1
fig_ctr+=1
fig = plt.figure(fig_ctr)

plt.subplot(2,1,1)
plt.title("Complex Channel - Real")
plt.stem(delays,np.real(pathAmplitudes))
plt.xlabel('t (ns)')
plt.ylabel('$\Re\{h(t)\}$')
plt.subplot(2,1,2)
plt.title("Complex Channel - Imaginary")
plt.stem(delays,np.imag(pathAmplitudes))
plt.xlabel('t (ns)')
plt.ylabel('$\Im\{h(t)\}$')

# %%
clusters, subpathsFix = model.fitAOA(txPos,rxPos,clusters,subpaths)
#%%
# O mesmo pero agora probamos as novas funcións:
# Datos iniciais - l0, tau0 e aod0 (losAOD)
clusters2 = clusters
subpaths2 = subpaths
# Datos iniciais - l0, tau0 e aod0 (losAOD)
vLOS = np.array(rxPos) - np.array(txPos)
l0 = np.linalg.norm(vLOS[0:-1])
tau0 = l0 / 3e8
losAOD =(np.mod( np.arctan( vLOS[1] / vLOS[0] )+np.pi*(vLOS[0]<0),2*np.pi))*(180.0/np.pi) # en graos

# Extraemos index. de clusters
#nClusters = clusters.shape[0]

#for i in range(0,nClusters -1):
    # de aquí sacamos aod e tau
    #TODO organizar para q procese ben valores do df
    #clusterValues = subpaths.loc()
aod = subpaths2['AOD'].astype(float)
tau = subpaths2['tau'].astype(float)
#TODO end

li = l0 + tau * 3e8
dAOD = (aod-losAOD)*(np.pi/180)

cosdAOD = np.cos(dAOD)
sindAOD = np.sin(dAOD)
nu = tau/tau0

# Resolvemos:
xsolA = (sindAOD*(1-nu))/(nu**2+1-(2*nu*cosdAOD))
xsolB = (sindAOD*(1+nu-(2*nu*cosdAOD)))/(nu**2+1-(2*nu*cosdAOD))

#Posibles solucions:
sols = np.zeros((4,aod.size)) 
sols[0,:] = np.transpose(np.arcsin(xsolA))
sols[1,:] = np.transpose(np.arcsin(xsolB))
sols[2,:] = np.transpose(np.pi - np.arcsin(xsolA))
sols[3,:] = np.transpose(np.pi - np.arcsin(xsolB))

#Avaliamos consistencia e distancia:
dist = np.zeros((4,aod.size))
for i in range(0,3):
    numNu= sindAOD + np.sin(sols[i,:])
    denomNu= sindAOD*np.cos(sols[i,:]) + cosdAOD*np.sin(sols[i,:])
    dist[i]= (abs(numNu/denomNu)-nu)

distMod = np.sum(dist,axis=1)    
solIndx=np.argmin(distMod,0)
sol = sols[solIndx,range(li.size)]
# Norm., convertimos de novo a graos e achamos o aoaReal - non o aux.:
aoaDeg = np.mod(sol,2*np.pi)
aoaDeg = aoaDeg*(180/np.pi)
subpaths['AOA'] = aoaDeg

# Eliminamos valores de AOA dos backlobes
# Creo función aparte para poder chamala dende calquer lado



#%% 
#Figure 2
fig_ctr+=1
fig = plt.figure(fig_ctr)
pathAmpdBtrunc25 = 10*np.log10(np.abs(pathAmplitudes)**2)
mindB = np.min(pathAmpdBtrunc25)

plt.plot((delays*np.ones((2,1))),np.vstack([mindB*np.ones_like(pathAmpdBtrunc25),pathAmpdBtrunc25]),'b-o')
plt.title("Power of each Delta")
plt.xlabel('t (ns)')
plt.ylabel('$|h(t)|^2$ [dB]')

#plots of discrete equivalent channels
#the DEC h[n] is a digital transmission pulse convolved with the channel and sampled p(t)*h(t)|_{t=nTs}
Ts=5 #ns
#while the DEC is an IIR filter, we approximate it as a FIR filter with a length that can cover the delays
Ds=np.max(delays)
Ntaps = int(np.ceil(Ds/Ts))


#1 path becomes 1 pulse (we choose sinc in this example)
#without multiplying by the complex gain, the pluse itself is real
fig_ctr+=1
fig = plt.figure(fig_ctr)
t=np.linspace(0,Ds,10*Ntaps)
pulsesOversampling = np.sinc((t-delays[:,None])/Ts)
plt.plot(t/Ts,pulsesOversampling[0,:],'k-.')

n=np.linspace(0,Ntaps-1,Ntaps)
pulses = np.sinc(n-delays[:,None]/Ts)
plt.stem(n,pulses[0,:])
plt.xlabel('n=t/Ts')
plt.ylabel('$|h[n]|^2$ [dB]')

#all paths together become a sum of complex-coefficients x sinc pulses
hn=np.sum(pulses*pathAmplitudes[:,None],axis=0)
hnOversampling=np.sum(pulsesOversampling*pathAmplitudes[:,None],axis=0)
#real and imaginary complex DEC
fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.subplot(2,1,1)
plt.plot(t/Ts,np.real(hnOversampling),'k-.')
plt.stem(n,np.real(hn))
plt.xlabel('t (ns)')
plt.ylabel('$\Re\{h[n]\}$')
plt.subplot(2,1,2)
plt.plot(t/Ts,np.imag(hnOversampling),'k-.')
plt.stem(n,np.imag(hn))
plt.xlabel('n=t/Ts')
plt.ylabel('$\Im\{h[n]\}$')

#power of each tap
fig_ctr+=1
fig = plt.figure(fig_ctr)


DECdBtrunc25Oversampling = 10*np.log10(np.abs(hnOversampling)**2)
plt.plot(t/Ts,DECdBtrunc25Oversampling,'k-.')
DECdBtrunc25 = 10*np.log10(np.abs(hn)**2)
mindB = np.min(DECdBtrunc25Oversampling)
plt.plot((n*np.ones((2,1))),np.vstack([mindB*np.ones_like(DECdBtrunc25),DECdBtrunc25]),'b-o')
plt.xlabel('n=t/Ts')
plt.ylabel('$|h[n]|^2$ [dB]')

# %%
# ArrayPolar:

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
tau_sp=tau_sp.reshape(nClusters,-1)
powC_sp=powC_sp.reshape(nClusters,-1)
AOA_sp=AOA_sp.reshape(nClusters,-1)
AOD_sp=AOD_sp.reshape(nClusters,-1)
ZOA_sp=ZOA_sp.reshape(nClusters,-1)
ZOD_sp=ZOD_sp.reshape(nClusters,-1)
plt.close('all')
fig_ctr=0

#2D polar plots of AoA
AoAs = AOA_sp.reshape(-1)*np.pi/180#radians
Npath=np.size(AoAs)
pathAmplitudes = np.sqrt( powC_sp.reshape(-1) )*np.exp(2j*np.pi*np.random.rand(Npath))

#plot of rx AoAs and channel gains
fig_ctr+=1
fig = plt.figure(fig_ctr)
pathAmplitudesdBtrunc25 = np.maximum(10*np.log10(np.abs(pathAmplitudes)**2),-45)

# plt.polar(AoAs*np.ones((2,1)),np.vstack([-40*np.ones((1,Npath)),pathAmplitudesdBtrunc25]),':')
# plt.scatter(AoAs,pathAmplitudesdBtrunc25,color='k',marker='x')
Nsp=AOA_sp.shape[1]

if los:
    plt.polar(losAoA*np.pi/180*np.ones((2,1)),np.vstack([[-40],10*np.log10(pow_los)]),':',color=cm.jet(0))
    plt.scatter(losAoA*np.pi/180*np.ones((2,1)),np.vstack([[-40],10*np.log10(pow_los)]),color=cm.jet(0),marker='<')
for n in range(nClusters):   
    pathAmplitudes_sp = np.sqrt( powC_sp[n,:] )*np.exp(2j*np.pi*np.random.rand(Nsp))
    pathAmplitudesdBtrunc25_sp = np.maximum(10*np.log10(np.abs(pathAmplitudes_sp)**2),-45)
    plt.polar(AOA_sp[n,:]*np.pi/180*np.ones((2,1)),np.vstack([-40*np.ones((1,Nsp)),pathAmplitudesdBtrunc25_sp]),':',color=cm.jet(n/(nClusters-1)) )
    plt.scatter(AOA_sp[n,:]*np.pi/180,pathAmplitudesdBtrunc25_sp,color=cm.jet(n/(nClusters-1)),marker='<')
plt.yticks(ticks=[-40,-30,-20,-10],labels=['-40dB','-30dB','-20dB','-10dB'])

# compute the response of the antenna array with Nant antennas
Nant = 16
AntennaResponses =mc.fULA(AoAs,Nant)
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


#plot of receive array response of ALL paths in SEPARATE LINES, WITHOUT the effect of power
fig_ctr+=1
fig = plt.figure(fig_ctr)
arrayGainAllPaths=(AntennaResponses.transpose([0,2,1]).conj()@BeamformingVectors[:,None,:,:]).reshape((Npointsplot,Npath))

arrayGainAllPathsdBtrunc25 = np.maximum(10*np.log10(Nant*np.abs(arrayGainAllPaths)**2),-25)

plt.polar(angles_plot,arrayGainAllPathsdBtrunc25)
plt.yticks(ticks=[-20,-10,0,10],labels=['-20dB','-10dB','0dB','10dB'])


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
