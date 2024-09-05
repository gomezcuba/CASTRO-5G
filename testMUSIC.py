#!/usr/bin/python

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

import sys
sys.path.append('../')
from CASTRO5G import multipathChannel as mc

# any function that receives a 1D array with anges[n], and returns a 3D array with ret[n,:,:] a column vector with the array phases at each angle
# lambda_fArray = mc.fULA
lambda_fArray = mc.fUCA

Nant=5
angle = np.pi/5
# angle = np.random.uniform(0,2*np.pi)
garray = lambda_fArray(angle,Nant,.5)

fs=10e6
fc=10e3
Nsamples=100
Px=1
t=np.arange(Nsamples)*1/fs
xo=np.exp(2j*np.pi*fc*t)
# xo=mc.AWGN((1,Nsamples),Px)


sigmaZ=1e-1
z=mc.AWGN((Nant,Nsamples),sigmaZ)
yarray=garray[:,None]*xo+z

COVy=np.cov(yarray)
eigvals,eigvecs=np.linalg.eigh(COVy)
noiseEigenvectors=eigvecs[:,0:-1]
signalEigenvector=eigvecs[:,-1:]

theta=np.linspace(0,2*np.pi,1001)
fMUSIC=np.sum(np.abs(noiseEigenvectors.T.conj()@lambda_fArray(theta,Nant).T)**2,axis=0)
# fMUSIC=1/np.sum(np.abs(signalEigenvector.T.conj()@lambda_fArray(theta,Nant).T)**2,axis=0)

yphase = np.angle(yarray)
def modangdif(a,b):
    return( np.mod(a-b+np.pi,2*np.pi)-np.pi )
yphasedif = modangdif( yphase[1:5], yphase[0:4] )
yphase_expected=np.angle(lambda_fArray(theta,Nant))
yphasedif_expected = modangdif( yphase_expected[:,1:5], yphase_expected[:,0:4] )
fQ=np.sum(np.abs(modangdif( yphasedif, yphasedif_expected[:,:,None] ) )**2,axis=(1,2))

fBeamforming=(np.sum(np.abs(yarray)**2)-np.sum(np.abs(yarray.T.conj()@lambda_fArray(theta,Nant).T )**2,axis=0))

#inspired by MUSIC, we estimate the covariance between antenna n-1 and n, but directly in the phases 
yphaseOfCovariance_diff = np.mean(modangdif( yphasedif, yphasedif_expected[:,:,None] ),axis=2)
fQcovariance=np.sum(np.abs( yphaseOfCovariance_diff )**2,axis=1)

plt.figure(1)

plt.plot(theta,fMUSIC/np.max(fMUSIC))
plt.plot(theta,fQ/np.max(fQ))
plt.plot(theta,fBeamforming/np.max(fBeamforming))
plt.plot(theta,fQcovariance/np.max(fQcovariance))
plt.legend(['MUSIC','Mod Squared Distance','Beamforming','Covariance Phase SQDist'])
plt.savefig('anglemetrics.png')


plt.figure(2)

fMUSICi=1/fMUSIC
fQi=1/fQ
fBeamformingi=1/fBeamforming
fQcovariancei=1/fQcovariance
plt.plot(theta,fMUSICi/np.max(fMUSICi))
plt.plot(theta,fQi/np.max(fQi))
plt.plot(theta,fBeamformingi/np.max(fBeamformingi))
plt.plot(theta,fQcovariancei/np.max(fQcovariancei))
plt.legend(['MUSIC','Mod Squared Distance','Beamforming','Covariance Phase SQDist'])
plt.savefig('anglemetrics.png')