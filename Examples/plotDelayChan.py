#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as pg


plt.close('all')
fig_ctr=0

model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=False)
plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
nClusters = clusters.shape[0]
nNLOSsp=subpaths.loc[1,:].shape[0]
# TDOA,powC,AOA,AOD,ZOA,ZOD = clusters.T.to_numpy()
los, PLfree, SF = plinfo
# TDOA_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp,XPR_sp,phase00,phase01,phase10,phase11 = subpaths.T.to_numpy()

#2D plots of power vs delay

#plots of continuous time channels (deltas)

#real and imaginary complex channel
fig_ctr+=1
fig = plt.figure(fig_ctr)
for n in range(nClusters):
    pathAmplitudes = np.sqrt( subpaths.loc[n,:].P )*np.exp(1j* subpaths.loc[n,:].phase00)
    plt.subplot(2,1,1)
    markerline, stemlines, baseline = plt.stem(subpaths.loc[n,:].TDoA *1e9,np.real(pathAmplitudes))
    plt.setp(stemlines, color=cm.jet(n/(nClusters-1)))
    plt.setp(markerline, color=cm.jet(n/(nClusters-1))) 
    plt.subplot(2,1,2)
    markerline, stemlines, baseline = plt.stem(subpaths.loc[n,:].TDoA *1e9,np.imag(pathAmplitudes))
    plt.setp(stemlines, color=cm.jet(n/(nClusters-1)))
    plt.setp(markerline, color=cm.jet(n/(nClusters-1))) 
plt.subplot(2,1,1)
plt.title("CIR for all clusters and subpaths")
plt.xlabel('t (ns)')
plt.ylabel('$\Re\{h(t)\}$')
plt.subplot(2,1,2)
plt.xlabel('t (ns)')
plt.ylabel('$\Im\{h(t)\}$')

#power of each delta
fig_ctr+=1
fig = plt.figure(fig_ctr)

mindB = np.min(10*np.log10( subpaths.P ))
for n in range(nClusters):
    pathAmpdBtrunc25 = 10*np.log10( subpaths.loc[n,:].P )
    markerline, stemlines, baseline = plt.stem(subpaths.loc[n,:].TDoA *1e9, pathAmpdBtrunc25, bottom=mindB)
    plt.setp(stemlines, color=cm.jet(n/(nClusters-1)))
    plt.setp(markerline, color=cm.jet(n/(nClusters-1))) 

plt.title("PDP for all clusters and subpaths")
plt.xlabel('t (ns)')
plt.ylabel('$|h(t)|^2$ [dB]')

#plots of discree equivalent channels
#the DEC h[n] is a digital transmission pulse convolved with the channel and sampled p(t)*h(t)|_{t=nTs}
Ts=2 #ns
#while the DEC is an IIR filter, we approximate it as a FIR filter with a length that can cover the delays
Ds=np.max(subpaths.TDoA *1e9)
Ntaps = int(np.ceil(Ds/Ts))
#1 path becomes 1 pulse (we choose sinc in this example)
PrecedingSincLobes = 5
#without multiplying by the complex gain, the pluse itself is real
fig_ctr+=1
fig = plt.figure(fig_ctr)
t=np.linspace(-PrecedingSincLobes*Ts,(Ntaps-1)*Ts,10*(Ntaps+PrecedingSincLobes)+1).reshape(-1,1)
taps=np.linspace(-PrecedingSincLobes,Ntaps-1,Ntaps+PrecedingSincLobes).reshape(-1,1)

Ntop = 8
topNpaths = subpaths.sort_values(by=['P'],ascending=False).index[0:Ntop]
for p in range(Ntop):
    n,m = topNpaths[p]    
    pathCoefAbs = np.sqrt( subpaths.loc[n,m].P )
    pulsesOversampling = np.sinc((t-subpaths.loc[n,m].TDoA *1e9)/Ts)
    plt.plot(t/Ts,pathCoefAbs*pulsesOversampling,':',color=cm.jet(p/(Ntop-1)))
    
    pulses = np.sinc(taps-subpaths.loc[n,m].TDoA *1e9/Ts)
    markerline, stemlines, baseline =  plt.stem(taps,pathCoefAbs*pulses,label="CIR subpath %d,%d"%(n,m))
    plt.setp(stemlines, color=cm.jet(p/(Ntop-1)))
    plt.setp(markerline, color=cm.jet(p/(Ntop-1))) 
plt.xlabel('n=t/Ts')
plt.ylabel('$h[n]$')
plt.title('%d strongest subpaths, sampled by sinc pulses with Ts=%.1f'%(Ntop,Ts))
plt.legend()

#all paths together become a sum of complex-coefficients x sinc pulses
pulsesOversampling = np.sinc((t-subpaths.TDoA.to_numpy() *1e9)/Ts)
pulses = np.sinc(taps -subpaths.TDoA.to_numpy() *1e9/Ts)
pathAmplitudes = ( np.sqrt( subpaths.P )*np.exp(1j* subpaths.phase00) ).to_numpy()
hn=np.sum(pulses*pathAmplitudes,axis=1)
hnOversampling=np.sum(pulsesOversampling*pathAmplitudes,axis=1)
#real and imaginary complex DEC
fig_ctr+=1
fig = plt.figure(fig_ctr)
plt.subplot(2,1,1)
plt.plot(t/Ts,np.real(hnOversampling),'k-.')
plt.stem(taps,np.real(hn))
plt.xlabel('t (ns)')
plt.ylabel('$\Re\{h[n]\}$')
plt.title('DEC sum of all subpaths, sampled by sinc pulses with Ts=%.1f'%(Ts))
plt.subplot(2,1,2)
plt.plot(t/Ts,np.imag(hnOversampling),'k-.')
plt.stem(taps,np.imag(hn))
plt.xlabel('n=t/Ts')
plt.ylabel('$\Im\{h[n]\}$')

#power of each tap
fig_ctr+=1
fig = plt.figure(fig_ctr)

DECdBtrunc25Oversampling = 10*np.log10(np.abs(hnOversampling)**2)
plt.plot(t/Ts,DECdBtrunc25Oversampling,'k-.')
DECdBtrunc25 = 10*np.log10(np.abs(hn)**2)
mindB = np.min(DECdBtrunc25)
plt.stem(taps,DECdBtrunc25,bottom=mindB)
plt.xlabel('n=t/Ts')
plt.ylabel('$|h[n]|^2$ [dB]')
plt.title('PDP sum of all subpaths, sampled by sinc pulses with Ts=%.1f'%(Ts))
