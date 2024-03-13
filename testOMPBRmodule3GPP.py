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
Nd=4 #Nt (Ntv*Nth) con Ntv=1
Na=4 #Nr (Nrv*Nrh) con Ntv=1
Nt=32
Nxp=2
Nrft=1
Nrfr=2
K=32
#Ts=2.5
Ts=320/Nt
Ds=Ts*Nt
SNRs=10**(np.arange(-1,2.01,1.0))
#SNRs=10**(np.arange(1,1.01,1.0))

omprunner = oc.OMPCachedRunner()
pilgen = ch.MIMOPilotChannel("IDUV")
chgen = mp3g.ThreeGPPMultipathChannelModel()
chgen.bLargeBandwidthOption=True

x0=np.random.rand(Nchan)*100-50
y0=np.random.rand(Nchan)*100-50

legStrAlgs=[
        'OMPx1',
        'OMPx1acc',
        'OMPx4',
        'OMPx4acc',
        'OMPBRx10',
        'OMPBRx10acc',
        ]
confAlgs=[#Xt Xd Xa Xmu accel
    # (1.0,1.0,1.0,1.0,False),
    (1.0,1.0,1.0,1.0,True),
    # (4.0,4.0,4.0,1.0,False),
    (8.0,8.0,8.0,1.0,True),
    (1.0,1.0,1.0,10.0,False),
    # (1.0,1.0,1.0,10.0,True),
    ]

Nalgs=len(confAlgs)
Nsnr=len(SNRs)
MSE=np.zeros((Nchan,Nsnr,Nalgs))
Npaths=np.zeros((Nchan,Nsnr,Nalgs))
totTimes=np.zeros((Nchan,Nsnr,Nalgs))
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
        
    (wp,vp)=pilgen.generatePilots(K*Nxp,(Nrfr,Na),(Nd,Nrft),"IDUV")
    wp=wp.reshape(K,Nxp,Nrfr,Na)
    vp=vp.reshape(K,Nxp,Nd,Nrft)

    zp=ch.AWGN((K,Nxp,Na,1))
#    h=np.fft.ifft(hk,axis=0)
#    hsparse=np.fft.ifft(h,axis=1)*np.sqrt(Nd)
#    hsparse=np.fft.ifft(hsparse,axis=2)*np.sqrt(Na)
    # print(hk)
    for isnr in range(0,Nsnr):
        sigma2=1.0/SNRs[isnr]
        
        yp=pilgen.applyPilotChannel(hk,wp,vp,zp*np.sqrt(sigma2))
        for nalg in range(0,Nalgs):
            Xt,Xd,Xa,Xmu,confAccel = confAlgs[nalg]
            t0 = time.time()
            hest,paths=omprunner.OMPBR(yp,sigma2*K*Nxp*Nrfr,ichan,vp,wp, Xt,Xd,Xa,Xmu, accelDel = confAccel)
            MSE[ichan,isnr,nalg] = np.mean(np.abs(hk-hest)**2)/np.mean(np.abs(hk)**2)
            totTimes[ichan,isnr,nalg] = time.time()-t0
            Npaths[ichan,isnr,nalg] = len(paths.delays)
    #for large Nsims the pilot cache grows too much so we free the memory when not needed
    for nalg in range(0,Nalgs):
        Xt,Xd,Xa,Xmu,confAccel = confAlgs[nalg]
        if confAccel:
            omprunner.freeCacheOfPilot(ichan,1,Nd,Na,Xt,Xd,Xa)
        else:
            omprunner.freeCacheOfPilot(ichan,Nt,Nd,Na,Xt,Xd,Xa)
    bar.next()
bar.finish()

print(np.mean(MSE,axis=0))
print(np.mean(totTimes,axis=0))
plt.semilogy(10*np.log10(SNRs),np.mean(MSE,axis=0))
plt.legend(legStrAlgs)
plt.xlabel('SNR(dB)')
plt.ylabel('MSE')
plt.figure()
NBarGroups=len(legStrAlgs)
barwidth=0.9/Nalgs
for nbar in range(Nalgs):
    plt.bar(np.arange(len(SNRs))+(nbar-(NBarGroups-1)/2)*barwidth,np.mean(totTimes[:,:,nbar],axis=0),width=barwidth,color=cm.jet(nbar/(NBarGroups-1)))
plt.xlabel('alg')
plt.ylabel('runtime')
plt.figure()
barwidth=0.9/Nalgs
for nbar in range(Nalgs):
    plt.bar(np.arange(len(SNRs))+(nbar-(NBarGroups-1)/2)*barwidth,np.mean(np.array(Npaths)[:,:,nbar],axis=0),width=barwidth,color=cm.jet(nbar/(NBarGroups-1)))
plt.xlabel('alg')
plt.ylabel('N paths')
# plt.show()
