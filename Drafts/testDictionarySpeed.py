#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import os
from tqdm import tqdm

import sys
sys.path.append('../')
from CASTRO5G import multipathChannel as mc
from CASTRO5G import compressedSensingTools as cs

#values for 8GB memory usage in the biggest dictionary
K=512
Ncp=32
Na=8
Nd=8
Nframe=5
Nrfr=1
Nrft=1
Tcp=570e-9 #mu=2
Ts=Tcp/Ncp
SNR=1e-2
dimH=(K,Ncp,Na,Nd)
dimPhi=(2*Ncp,2*Na,2*Nd)
dimY=(Nframe,K,Nrfr)

dicBase=cs.CSCachedDictionary()
dicFFT=cs.CSBasicFFTDictionary()
dicMult=cs.CSMultiDictionary()
dicFast=cs.CSMultiFFTDictionary()

bytesPerFloat = np.array([0],dtype=np.complex128).itemsize
# %timeit dicBase.setHDic(dimH,dimPhi)
tini=time.time()
dicBase.setHDic(dimH,dimPhi)
print(f'Create HDic base: {time.time()-tini:.2f} seconds {bytesPerFloat/1024/1024*dicBase.currHDic.mPhiH.size:.2f} MB memory')
# %timeit dicFFT.setHDic(dimH,dimPhi)
tini=time.time()
dicFFT.setHDic(dimH,dimPhi)
print(f'Create HDic fft: {time.time()-tini:.2f} seconds {bytesPerFloat/1024/1024*dicFFT.currHDic.mPhiH.size:.2f} MB memory')
# %timeit dicMult.setHDic(dimH,dimPhi)
tini=time.time()
dicMult.setHDic(dimH,dimPhi)
print(f'Create HDic mult: {time.time()-tini:.2f} seconds {bytesPerFloat/1024/1024*np.sum([x.size for x in dicMult.currHDic.mPhiH]):.2f} MB memory')
# %timeit dicFast.setHDic(dimH,dimPhi)
tini=time.time()
dicFast.setHDic(dimH,dimPhi)
print(f'Create HDic fast: {time.time()-tini:.2f} seconds {0} MB memory')

pilgen = mc.MIMOPilotChannel("IDUV")
wp,vp=pilgen.generatePilots(Nframe*K*Nrft,Na,Nd,Npr=Nframe*K*Nrfr,rShape=(Nframe,K,Nrfr,Na),tShape=(Nframe,K,Nd,Nrft))

# %timeit dicBase.setYDic("someID",(wp,vp))
tini=time.time()
dicBase.setYDic("someID",(wp,vp))
print(f'Create YDic base: {time.time()-tini:.2f} seconds {bytesPerFloat/1024/1024*dicBase.currYDic.mPhiY.size:.2f} MB memory')
tini=time.time()
# %timeit dicFFT.setYDic("someID",(wp,vp))
dicFFT.setYDic("someID",(wp,vp))
print(f'Create YDic fft: {time.time()-tini:.2f} seconds {bytesPerFloat/1024/1024*dicFFT.currYDic.mPhiY.size:.2f} MB memory')
tini=time.time()
# %timeit dicMult.setYDic("someID",(wp,vp))
dicMult.setYDic("someID",(wp,vp))
print(f'Create YDic mult: {time.time()-tini:.2f} seconds {bytesPerFloat/1024/1024*np.sum([x.size for x in dicMult.currYDic.mPhiY ]):.2f} MB memory')
tini=time.time()
# %timeit dicFast.setYDic("someID",(wp,vp))
dicFast.setYDic("someID",(wp,vp))
print(f'Create YDic fast: {time.time()-tini:.2f} seconds {0} MB memory')

chgen=mc.UniformMultipathChannelModel(Npath=1,Ds=.9*Tcp,mode3D=False)
allPathsData=chgen.create_channel()
print(allPathsData)

mpch = mc.MultipathChannel((0,0,10),(40,0,1.5),allPathsData.loc[0,:])
ht=mpch.getDEC(Na,Nd,Ncp,Ts)*np.sqrt(Nd*Na)#mpch uses normalized matrices of gain 1
hk=np.fft.fft(ht,K,axis=0)
zp=mc.AWGN((Nframe,K,Na,1))
zp_bb=np.matmul(wp,zp)
yp_noiseless=pilgen.applyPilotChannel( hk,wp,vp,None)

yp=yp_noiseless+np.sqrt(1/SNR)*zp_bb

%timeit dicBase.projY(yp.reshape(-1,1))
%timeit dicFFT.projY(yp.reshape(-1,1))
%timeit dicMult.projY(yp.reshape(-1,1))
%timeit dicFast.projY(yp.reshape(-1,1))

cBase= dicBase.projY(yp.reshape(-1,1))
cFFT= dicFFT.projY(yp.reshape(-1,1))
cMult= dicMult.projY(yp.reshape(-1,1))
cFast= dicFast.projY(yp.reshape(-1,1))

print(f'Mult dictioary is equal to base: {np.all(np.isclose(cBase,cMult))}')
print(f'1D-FFT dictioary is equal to multiFFT: {np.all(np.isclose(cFFT,np.fft.fftshift(cFast.reshape(dimPhi),axes=(1,2)).reshape(-1,1)))}')

plt.plot(np.abs(cBase),'b',label='Basic Dictionary')
plt.plot(np.abs(cFFT),'r-.',label='1D delay FFT Dictionary')
plt.plot(np.abs(cMult),'c:o',label='Multi Dictionary')
plt.plot(np.abs(cFFT),'m:s',label='3D FFT Multi Dictionary')
plt.xlabel("Dictionary item")
plt.ylabel("Projection metric")
plt.legend()

tau,aoa,aod=dicBase.ind2Param(np.argmax(cBase))
tau=tau*Ts
#NOTE the aoa and aod estimates are from -pi/2 to pi/2. The array suffers simetry sin(phi)=sin(pi-phi) and the "backlobe" angles are mirrored
print(f'estimated multipath delay {tau} (º) aoa {aoa:.2f} (rad) aod {aod:.2f}  (rad)')
