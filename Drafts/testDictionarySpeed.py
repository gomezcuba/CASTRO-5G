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
K=64
Ncp=32
Na=8
Nd=8
Nframe=3
Nrfr=1
Nrft=1
Tcp=570e-9 #mu=2
Ts=Tcp/Ncp
SNR=1e-2
dimH=(K,Ncp,Na,Nd)
dimPhi=(4*Ncp,4*Na,4*Nd)
dimY=(Nframe,K,Nrfr)

bTestBase = True #disable for large sizes
bTestFFT = True
bTestMult = True
bTestFast = True

if bTestBase:
    dicBase=cs.CSCachedDictionary()
if bTestFFT:
    dicFFT=cs.CSBasicFFTDictionary()
if bTestMult:
    dicMult=cs.CSMultiDictionary()
if bTestFast:
    dicFast=cs.CSMultiFFTDictionary()

bytesPerFloat = np.array([0],dtype=np.complex128).itemsize
if bTestBase:
    # %timeit dicBase.setHDic(dimH,dimPhi)
    tini=time.time()
    dicBase.setHDic(dimH,dimPhi)
    print(f'Create HDic base: {time.time()-tini:.2f} seconds {bytesPerFloat/1024/1024*dicBase.currHDic.mPhiH.size:.2f} MB memory')
if bTestFFT:
    # %timeit dicFFT.setHDic(dimH,dimPhi)
    tini=time.time()
    dicFFT.setHDic(dimH,dimPhi)
    print(f'Create HDic fft: {time.time()-tini:.2f} seconds {bytesPerFloat/1024/1024*dicFFT.currHDic.mPhiH.size:.2f} MB memory')
if bTestMult:
    # %timeit dicMult.setHDic(dimH,dimPhi)
    tini=time.time()
    dicMult.setHDic(dimH,dimPhi)
    print(f'Create HDic mult: {time.time()-tini:.2f} seconds {bytesPerFloat/1024/1024*np.sum([x.size for x in dicMult.currHDic.mPhiH]):.2f} MB memory')
if bTestFast:
    # %timeit dicFast.setHDic(dimH,dimPhi)
    tini=time.time()
    dicFast.setHDic(dimH,dimPhi)
    print(f'Create HDic fast: {time.time()-tini:.2f} seconds {0} MB memory')

pilgen = mc.MIMOPilotChannel("IDUV")
wp,vp=pilgen.generatePilots(Nframe*K*Nrft,Na,Nd,Npr=Nframe*K*Nrfr,rShape=(Nframe,K,Nrfr,Na),tShape=(Nframe,K,Nd,Nrft))

if bTestBase:
    # %timeit dicBase.setYDic("someID",(wp,vp))
    tini=time.time()
    dicBase.setYDic("someID",(wp,vp))
    print(f'Create YDic base: {time.time()-tini:.2f} seconds {bytesPerFloat/1024/1024*dicBase.currYDic.mPhiY.size:.2f} MB memory')
if bTestFFT:
    tini=time.time()
    # %timeit dicFFT.setYDic("someID",(wp,vp))
    dicFFT.setYDic("someID",(wp,vp))
    print(f'Create YDic fft: {time.time()-tini:.2f} seconds {bytesPerFloat/1024/1024*dicFFT.currYDic.mPhiY.size:.2f} MB memory')
if bTestMult:
    tini=time.time()
    # %timeit dicMult.setYDic("someID",(wp,vp))
    dicMult.setYDic("someID",(wp,vp))
    print(f'Create YDic mult: {time.time()-tini:.2f} seconds {bytesPerFloat/1024/1024*np.sum([x.size for x in dicMult.currYDic.mPhiY ]):.2f} MB memory')
if bTestFast:
    tini=time.time()
    # %timeit dicFast.setYDic("someID",(wp,vp))
    dicFast.setYDic("someID",(wp,vp))
    print(f'Create YDic fast: {time.time()-tini:.2f} seconds {0} MB memory')

chgen=mc.UniformMultipathChannelModel(Npath=1,Ds=.9*Tcp,mode3D=False)
allPathsData=chgen.create_channel()
print(allPathsData)

mpch = mc.MultipathDEC((0,0,10),(40,0,1.5),allPathsData.loc[0,:])
ht=mpch.getDEC(Na,Nd,Ncp,Ts)*np.sqrt(Nd*Na)#mpch uses normalized matrices of gain 1
hk=np.fft.fft(ht,K,axis=0)
zp=mc.AWGN((Nframe,K,Na,1))
zp_bb=np.matmul(wp,zp)
yp_noiseless=pilgen.applyPilotChannel( hk,wp,vp,None)

yp=yp_noiseless+np.sqrt(1/SNR)*zp_bb

if bTestBase:
    %timeit dicBase.projY(yp.reshape(-1,1))
if bTestFFT:
    %timeit dicFFT.projY(yp.reshape(-1,1))
if bTestMult:
    %timeit dicMult.projY(yp.reshape(-1,1))
if bTestFast:
    %timeit dicFast.projY(yp.reshape(-1,1))

if bTestBase:
    cBase= dicBase.projY(yp.reshape(-1,1))
if bTestFFT:
    cFFT= dicFFT.projY(yp.reshape(-1,1))
if bTestMult:
    cMult= dicMult.projY(yp.reshape(-1,1))
if bTestFast:
    cFast= dicFast.projY(yp.reshape(-1,1))

if bTestBase and bTestMult:
    print(f'Mult dictioary is equal to base: {np.all(np.isclose(cBase,cMult))}')
if bTestFFT and bTestFast:
    print(f'1D-FFT dictioary is equal to multiFFT: {np.all(np.isclose(cFFT,np.fft.fftshift(cFast.reshape(dimPhi),axes=(1,2)).reshape(-1,1)))}')

if bTestBase:
    plt.plot(np.abs(cBase),'b',label='Basic Dictionary')
if bTestFFT:
    plt.plot(np.abs(cFFT),'r-.',label='1D delay FFT Dictionary')
if bTestMult:
    plt.plot(np.abs(cMult),'c:o',label='Multi Dictionary')
if bTestFast:
    plt.plot(np.abs(cFast),'m:s',label='3D FFT Multi Dictionary')
plt.xlabel("Dictionary item")
plt.ylabel("Projection metric")
plt.legend()

if bTestMult:
    tau,aoa,aod=dicMult.ind2Param(np.argmax(cMult))
    tau=tau*Ts
    #NOTE the aoa and aod estimates are from -pi/2 to pi/2. The array suffers simetry sin(phi)=sin(pi-phi) and the "backlobe" angles are mirrored
    print(f"""Estimated multipath values:
          TDoA: true {allPathsData.TDoA.to_numpy()[0]*1e9:.2f} ns, estimated {tau*1e9:.2f} ns, error {(allPathsData.TDoA.to_numpy()[0]-tau)*1e9:.2f} ns
          AoD: true {np.mod(allPathsData.AoD.to_numpy()[0]*180/np.pi,360):.2f} º, estimated  {np.mod(aod*180/np.pi,360):.2f} º, error {np.mod(allPathsData.AoD.to_numpy()[0]*180/np.pi,360)-np.mod(aod*180/np.pi,360):.2f} º
          AoA: true {np.mod(allPathsData.AoA.to_numpy()[0]*180/np.pi,360):.2f} º, estimated  {np.mod(aoa*180/np.pi,360):.2f} º, error {np.mod(allPathsData.AoA.to_numpy()[0]*180/np.pi,360)-np.mod(aoa*180/np.pi,360):.2f} º
          """)
