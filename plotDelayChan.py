#!/usr/bin/python

import threeGPPMultipathGenerator as pg

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import animation, rc
rc('animation', html='html5')


plt.close('all')
fig_ctr=0

model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=True)
plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
tau,powC,AOA,AOD,ZOA,ZOD = clusters.T.to_numpy()
los, PLfree, SF = plinfo
tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp,XPR_sp,phase00,phase01,phase10,phase11 = subpaths.T.to_numpy()

#2D plots of power vs delay

#plots of continuous time channels (deltas)
delays = tau_sp*1e9#nanoseconds
Npath=np.size(delays)
pathAmplitudes = np.sqrt(pow_sp )*np.exp(1j*phase00)

#real and imaginary complex channel
fig_ctr+=1
fig = plt.figure(fig_ctr)

plt.subplot(2,1,1)
plt.stem(delays,np.real(pathAmplitudes))
plt.xlabel('t (ns)')
plt.ylabel('$\Re\{h(t)\}$')
plt.subplot(2,1,2)
plt.stem(delays,np.imag(pathAmplitudes))
plt.xlabel('t (ns)')
plt.ylabel('$\Im\{h(t)\}$')

#power of each delta
fig_ctr+=1
fig = plt.figure(fig_ctr)

pathAmpdBtrunc25 = 10*np.log10(np.abs(pathAmplitudes)**2)

mindB = np.min(pathAmpdBtrunc25)

plt.plot((delays*np.ones((2,1))),np.vstack([mindB*np.ones_like(pathAmpdBtrunc25),pathAmpdBtrunc25]),'b-o')
plt.xlabel('t (ns)')
plt.ylabel('$|h(t)|^2$ [dB]')

#plots of discree equivalent channels
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
