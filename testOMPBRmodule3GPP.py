#!/usr/bin/python

import threeGPPMultipathGenerator as mp3g
import multipathChannel as ch
import OMPCachedRunner as oc
import MIMOPilotChannel as pil
import csProblemGenerator as prb
#import testRLmp as rl

import matplotlib.pyplot as plt
import numpy as np

import time
from progress.bar import Bar

plt.close('all')

Nchan=10
Nd=4
Na=4
Nt=128
Nxp=2
Nrft=1
Nrfr=2
K=128
#Ts=2.5
Ts=320/Nt
Ds=Ts*Nt
SNRs=10**(np.arange(-1,2.01,1.0))
#SNRs=10**(np.arange(1,1.01,1.0))
MSE=[]
Npaths=[]
totTimes=[]

omprunner = oc.OMPCachedRunner()
pilgen = pil.MIMOPilotChannel("IDUV")
chgen = mp3g.ThreeGPPMultipathChannelModel()
chgen.bLargeBandwidthOption=True
probgen = prb.csProblemGenerator(Nt,Nd,Na,Nrft,Nrfr,Nxp,Ts,"IDUV")
probgen.pregenerate(Nchan)


x0=np.random.rand(Nchan)*100-50
y0=np.random.rand(Nchan)*100-50

legStrAlgs=[
        'OMPx1',
        'OMPx2',
        'OMPBR',
        ]

bar = Bar("CS sims", max=Nchan)
bar.check_tty = False
for ichan in range(0,Nchan):
    
    plinfo,macro,small = chgen.create_channel((0,0,10),(40,0,1.5))
    clusters,subpaths = small
    tau_sp,powC_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths
    mpch = ch.MultipathChannel((0,0,10),(40,0,1.5),[])
    tau_sp = tau_sp.reshape(-1)
    Npath = tau_sp.size
    AOA_sp = AOA_sp.reshape(-1)
    AOD_sp = AOD_sp.reshape(-1)
    ZOA_sp = ZOA_sp.reshape(-1)
    ZOD_sp = ZOD_sp.reshape(-1)
    pathAmplitudes = np.sqrt( powC_sp.reshape(-1) )*np.exp(2j*np.pi*np.random.rand(Npath))
    mpch.insertPathsFromListParameters(pathAmplitudes,tau_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp,np.zeros_like(pathAmplitudes))
    ht=mpch.getDEC(Na,Nd,Nt,Ts)*np.sqrt(Nd*Na)#mpch uses normalized matrices of gain 1
    hk=np.fft.fft(ht.transpose([2,0,1]),K,axis=0)
        
    (wp,vp)=pilgen.generatePilots((K,Nxp,Nrfr,Na,Nd,Nrft),"IDUV")       
    zp=probgen.zAWGN((K,Nxp,Na,1))
#    h=np.fft.ifft(hk,axis=0)
#    hsparse=np.fft.ifft(h,axis=1)*np.sqrt(Nd)
#    hsparse=np.fft.ifft(hsparse,axis=2)*np.sqrt(Na)
    # print(hk)
    MSE.append([])
    totTimes.append([])
    Npaths.append([])
    
    for isnr in range(0,len(SNRs)):
        sigma2=1.0/SNRs[isnr]
        
        yp=pilgen.applyPilotChannel(hk,wp,vp,zp*np.sqrt(sigma2))
        t0 = time.time()
        hest3,paths=omprunner.OMPBR(yp,sigma2*K*Nxp*Nrfr,ichan,vp,wp,1.0,1.0,1.0,1.0, accelDel = True)
        MSE[-1].append([])
        MSE[-1][-1].append(np.mean(np.abs(hk-hest3)**2)/np.mean(np.abs(hk)**2))
        totTimes[-1].append([])
        totTimes[-1][-1].append(time.time()-t0)
        Npaths[-1].append([])        
        Npaths[-1][-1].append(len(paths.delays))

        t0 = time.time()
        hest4,paths=omprunner.OMPBR(yp,sigma2*K*Nxp*Nrfr,ichan,vp,wp,4.0,4.0,4.0,1.0, accelDel = True)
        MSE[-1][-1].append(np.mean(np.abs(hk-hest4)**2)/np.mean(np.abs(hk)**2))
        totTimes[-1][-1].append(time.time()-t0)
        Npaths[-1][-1].append(len(paths.delays))

        t0 = time.time()
        hest5,paths=omprunner.OMPBR(yp,sigma2*K*Nxp*Nrfr,ichan,vp,wp,1.0,1.0,1.0,10.0, accelDel = True)
        MSE[-1][-1].append(np.mean(np.abs(hk-hest5)**2)/np.mean(np.abs(hk)**2))
        totTimes[-1][-1].append(time.time()-t0)
        Npaths[-1][-1].append(len(paths.delays))
    bar.next()
bar.finish()

MSE=np.array(MSE)
totTimes=np.array(totTimes)
print(np.mean(MSE,axis=0))
print(np.mean(totTimes,axis=0))
plt.semilogy(10*np.log10(SNRs),np.mean(MSE,axis=0))
plt.legend(legStrAlgs)
plt.xlabel('SNR(dB)')
plt.ylabel('MSE')
plt.figure()
plt.bar(range(np.size(totTimes)//Nchan),np.mean(totTimes,axis=0).reshape(np.size(totTimes)//Nchan,))
plt.xlabel('alg')
plt.ylabel('runtime')
plt.figure()
plt.bar(range(np.size(totTimes)//Nchan),np.mean(Npaths,axis=0).reshape(np.size(Npaths)//Nchan,))
plt.xlabel('alg')
plt.ylabel('N paths')
plt.show()
