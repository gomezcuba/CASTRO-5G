#!/usr/bin/python

from CASTRO5G import threeGPPMultipathGenerator as mp3g
from CASTRO5G import multipathChannel as ch
from CASTRO5G import OMPCachedRunner as oc
#import testRLmp as rl

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import time
from progress.bar import Bar

plt.close('all')

Nchan=5
#()
Nd=16 #Nt (Ntv*Nth) con Ntv=1
Na=16 #Nr (Nrv*Nrh) con Ntv=1
Nt=128
Nsym=1
Nrft=1
Nrfr=2
K=512
#Ts=2.5
Ts=320/Nt
Ds=Ts*Nt
SNRs=10**(np.arange(-1,2.01,1.0))
#SNRs=10**(np.arange(1,1.01,1.0))

omprunner = oc.OMPCachedRunner()
dicBasic=oc.CSCachedDictionary()
dicAcc=oc.CSAccelDictionary()
pilgen = ch.MIMOPilotChannel("IDUV")
chgen = mp3g.ThreeGPPMultipathChannelModel()
chgen.bLargeBandwidthOption=True

x0=np.random.rand(Nchan)*100-50
y0=np.random.rand(Nchan)*100-50

confAlgs=[#Xt Xd Xa Xmu accel legend string name
    # (1.0,1.0,1.0,1.0,dicBasic,'OMPx1'),
    (1.0,1.0,1.0,1.0,dicAcc,'OMPx1a'),
    # (4.0,1.0,1.0,1.0,dicBasic,'OMPx4T'),
    # (4.0,1.0,1.0,1.0,dicAcc,'OMPx4Ta'),
    # (4.0,4.0,4.0,1.0,dicBasic,'OMPx4'),
    (4.0,4.0,4.0,1.0,dicAcc,'OMPx4a'),
    # (1.0,1.0,1.0,100.0,dicBasic,'OMPBR'),
    (1.0,1.0,1.0,10.0,dicAcc,'OMPBRa'),
    ]

legStrAlgs=[x[-1] for x in confAlgs]
Nalgs=len(confAlgs)
Nsnr=len(SNRs)
MSE=np.zeros((Nchan,Nsnr,Nalgs))
Npaths=np.zeros((Nchan,Nsnr,Nalgs))
dicPrepTime=np.zeros((Nchan,Nalgs))
runTimes=np.zeros((Nchan,Nsnr,Nalgs))
bar = Bar("CS sims", max=Nchan)
bar.check_tty = False
for ichan in range(0,Nchan):
    
    model = mp3g.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=True)
    plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
    tau,powC,AOA,AOD,ZOA,ZOD = clusters.T.to_numpy()
    los, PLfree, SF = plinfo
    tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp,XPR_sp,phase00,phase01,phase10,phase11 = subpaths.T.to_numpy()

    mpch = ch.MultipathChannel((0,0,10),(40,0,1.5),[])
    Npath = tau_sp.size
    pathAmplitudes = np.sqrt( pow_sp )*np.exp(-1j*phase00)
    mpch.insertPathsFromListParameters(pathAmplitudes,tau_sp,AOA_sp*np.pi/180,AOD_sp*np.pi/180,ZOA_sp*np.pi/180,ZOD_sp*np.pi/180,np.zeros_like(pathAmplitudes))
    ht=mpch.getDEC(Na,Nd,Nt,Ts)*np.sqrt(Nd*Na)#mpch uses normalized matrices of gain 1
    hk=np.fft.fft(ht.transpose([2,0,1]),K,axis=0)
        
    (wp,vp)=pilgen.generatePilots(Nsym*K*Nrft,Na,Nd,Npr=Nsym*K*Nrfr,rShape=(Nsym,K,Nrfr,Na),tShape=(Nsym,K,Nd,Nrft))

    zp=ch.AWGN((Nsym,K,Na,1))
    zp_bb=np.matmul(wp,zp)
    yp_noiseless=pilgen.applyPilotChannel(hk,wp,vp,None)
    
    
    for nalg in range(0,Nalgs):
        t0 = time.time()
        Xt,Xd,Xa,Xmu,confDic,label = confAlgs[nalg]        
        confDic.setHDic((K,Nt,Na,Nd),(int(Nt*Xt),int(Nd*Xd),int(Na*Xa))) 
        confDic.setYDic(ichan,(wp,vp))
        dicPrepTime[ichan,nalg] = time.time()-t0
    for isnr in range(0,Nsnr):
        sigma2=1.0/SNRs[isnr]
        yp=yp_noiseless+zp_bb*np.sqrt(sigma2)
        for nalg in range(0,Nalgs):
            Xt,Xd,Xa,Xmu,confDic,label = confAlgs[nalg]
            t0 = time.time()
            omprunner.setDictionary(confDic)
            hest,paths=omprunner.OMPBR(yp,sigma2*K*Nsym*Nrfr,ichan,vp,wp, Xt,Xa,Xd,Xmu,Nt)
            MSE[ichan,isnr,nalg] = np.mean(np.abs(hk-hest)**2)/np.mean(np.abs(hk)**2)
            runTimes[ichan,isnr,nalg] = time.time()-t0
            Npaths[ichan,isnr,nalg] = len(paths.delays)
    #for large Nsims the pilot cache grows too much so we free the memory when not needed
    for nalg in range(0,Nalgs):
        Xt,Xd,Xa,Xmu,confDic,label = confAlgs[nalg]
        confDic.freeCacheOfPilot(ichan,(Nt,Na,Nd),(int(Nt*Xt),int(Na*Xa),int(Nd*Xd)))
    bar.next()
bar.finish()
print(np.mean(MSE,axis=0))
print(np.mean(runTimes,axis=0))
plt.semilogy(10*np.log10(SNRs),np.mean(MSE,axis=0))
plt.legend(legStrAlgs)
plt.xlabel('SNR(dB)')
plt.ylabel('MSE')
plt.figure()
barwidth=0.9/Nalgs * np.mean(np.diff(10*np.log10(SNRs)))
for ialg in range(Nalgs):
    offset=(ialg-(Nalgs-1)/2)*barwidth
    plt.bar(10*np.log10(SNRs)+offset,np.mean(runTimes[:,:,ialg],axis=0),width=barwidth,label=legStrAlgs[ialg])
plt.xlabel('SNR(dB)')
plt.ylabel('runtime')
plt.legend(legStrAlgs)
plt.figure()
barwidth=0.9/Nalgs * np.mean(np.diff(10*np.log10(SNRs)))
for ialg in range(Nalgs):
    offset=(ialg-(Nalgs-1)/2)*barwidth
    plt.bar(10*np.log10(SNRs)+offset,np.mean(Npaths[:,:,ialg],axis=0),width=barwidth,label=legStrAlgs[ialg])
plt.xlabel('SNR(dB)')
plt.ylabel('N paths')
plt.legend(legStrAlgs)
plt.show()