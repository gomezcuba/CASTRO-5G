#!/usr/bin/python

from CASTRO5G import OMPCachedRunner as oc
from CASTRO5G import multipathChannel as mc

import matplotlib.pyplot as plt
import numpy as np

import time
from progress.bar import Bar

plt.close('all')

Nchan=1
Nd=4
Na=4
Nt=16
Nxp=3
Nrft=1
Nrfr=2
K=32
Ts=2.5
Ds=Ts*Nt
SNRs=10**(np.arange(-1,2.01,1.0))
#SNRs=10**(np.arange(1,1.01,1.0))
# SNRs=np.array([100])

omprunner = oc.OMPCachedRunner()
dicBasic=oc.CSCachedDictionary()
dicAcc=oc.CSAccelDictionary()
pilgen = mc.MIMOPilotChannel("IDUV")
model=mc.DiscreteMultipathChannelModel(dims=(Nt,Na,Nd),fftaxes=(1,2))
listPreparedProblems = []
for ichan in range(Nchan):    
    h=model.getDEC(3)
    hk=np.fft.fft(h,K,axis=0)
    zp=mc.AWGN((Nxp,K,Na,1))
    (wp,vp)=pilgen.generatePilots(Nxp*K*Nrft,Na,Nd,Npr=Nxp*K*Nrfr,rShape=(Nxp,K,Nrfr,Na),tShape=(Nxp,K,Nd,Nrft))
    listPreparedProblems.append( (hk,zp,wp,vp ) )

# legStrAlgs=[
# #        'LS',
# #        'simplifiedOMP',
# #        'OMPx1',
# #        'OMPx2',
#         'OMPBR',
#         'OMPBR accel',
# #        'simplifiedISTA',
# #        'ISTAx1',
# #        'simplifiedFISTA',
# #        'FISTAx1',
# #        'simplifiedAMP',
# #        'AMP',
# #        'VAMP'
#         ]

algList = [
        "dir",
        "sOMP",
        # "sISTA",
        # "sFISTA",
        # "sAMP",
        "OMPx1",
        # "OMPx2",
        # "OMPx4",
        "OMPBR",
        "OMPx4a",
        # "ISTAx1",
        # "ISTAx2",
        # "FISTAx1",
        # "AMPx1",
        # "VAMPx1",
        ]
bar = Bar("CS sims", max=Nchan)
bar.check_tty = False
Nalg=len(algList)
Nsnr=len(SNRs)
MSE=np.zeros((Nchan,Nsnr,Nalg))
runTime=np.zeros((Nchan,Nsnr,Nalg))
Npaths=np.zeros((Nchan,Nsnr,Nalg))
for ichan in range(Nchan):
    (hk,zp,wp,vp)=listPreparedProblems[ichan]
    zp_bb=np.matmul(wp,zp)
    yp_noiseless=pilgen.applyPilotChannel(hk,wp,vp,None)
    zh=mc.AWGN((Nt,Na,Nd))
        
    hsparse=np.fft.ifft(hk,axis=0)[0:Nt,:,:]
    hsparse=np.fft.ifft(hsparse,axis=1)*np.sqrt(Nd)
    hsparse=np.fft.ifft(hsparse,axis=2)*np.sqrt(Na)

    for isnr in range(Nsnr):
        sigma2=1.0/SNRs[isnr]
        yp=yp_noiseless+zp_bb*np.sqrt(sigma2)
        hnoised=hsparse*np.sqrt(Nt)+zh*np.sqrt(sigma2)
        for ialg in range(Nalg):
            alg=algList[ialg]            
            t0 = time.time()
            if alg in ["dir","sOMP","sISTA","sFISTA","sAMP"]:                
                horig=hsparse*np.sqrt(Nt)
            else:
                horig=hk
            if alg=="dir":
                #direct observation with noise
                hest=hnoised
            elif alg=="sOMP":
                #simplifiedOMP sparse component selection
                hest=oc.simplifiedOMP(hnoised,sigma2*Nt*Na*Nd)
            elif alg=="sISTA":
                hest=oc.simplifiedISTA(hnoised,.5*np.sqrt(sigma2),15,hsparse*np.sqrt(Nt))
            elif alg=="sFISTA":
                hest=oc.simplifiedFISTA(hnoised,.5*np.sqrt(sigma2),15,hsparse*np.sqrt(Nt))
            elif alg=="sAMP":
                hest=oc.simplifiedAMP(hnoised,.5*np.sqrt(sigma2),15,hsparse*np.sqrt(Nt))                
            elif alg=="OMPx1":
                omprunner.setDictionary(dicBasic)
                hest,paths=omprunner.OMPBR(yp,sigma2*K*Nxp*Nrfr,ichan,vp,wp,1.0,1.0,1.0,1.0,Nt)
            elif alg=="OMPx2":  
                omprunner.setDictionary(dicBasic)
                hest,paths=omprunner.OMPBR(yp,sigma2*K*Nxp*Nrfr,ichan,vp,wp,2.0,2.0,2.0,1.0,Nt)    
            elif alg=="OMPx4":  
                omprunner.setDictionary(dicBasic)
                hest,paths=omprunner.OMPBR(yp,sigma2*K*Nxp*Nrfr,ichan,vp,wp,4.0,1.0,1.0,1.0,Nt)
            elif alg=="OMPBR":
                omprunner.setDictionary(dicBasic)
                hest,paths=omprunner.OMPBR(yp,sigma2*K*Nxp*Nrfr,ichan,vp,wp,1.0,1.0,1.0,100.0,Nt)
            elif alg=="OMPx4a": 
                omprunner.setDictionary(dicAcc)
                hest,paths=omprunner.OMPBR(yp,sigma2*K*Nxp*Nrfr,ichan,vp,wp,4.0,1.0,1.0,1.0,Nt)
            elif alg=="ISTAx1":
                omprunner.setDictionary(dicBasic)
                hest,paths=omprunner.Shrinkage(yp, (.5*np.sqrt(sigma2), .5) ,15,ichan,vp,wp,1.0,1.0,1.0,'ISTA')
            elif alg=="ISTAx2":
                omprunner.setDictionary(dicBasic)
                hest,paths=omprunner.Shrinkage(yp, (.5*np.sqrt(sigma2), .5) ,15,ichan,vp,wp,2.0,2.0,2.0,'ISTA')
            elif alg=="FISTAx1":
                omprunner.setDictionary(dicBasic)
                hest,paths=omprunner.Shrinkage(yp, (.5*np.sqrt(sigma2), .5) ,15,ichan,vp,wp,1.0,1.0,1.0,'FISTA')
            elif alg=="AMPx1":
                omprunner.setDictionary(dicBasic)
                hest,paths=omprunner.Shrinkage(yp, (.5*np.sqrt(sigma2), .5) ,15,ichan,vp,wp,1.0,1.0,1.0,'AMP')
            elif alg=="VAMPx1":
                omprunner.setDictionary(dicBasic)
                hest,paths=omprunner.Shrinkage(yp, (.5*np.sqrt(sigma2), .5) ,15,ichan,vp,wp,1.0,1.0,1.0,'VAMP')
            else:
                print("Unrecognized algorithm, doing nothing")
            MSE[ichan,isnr,ialg] = np.mean(np.abs(horig-hest)**2)/np.mean(np.abs(horig)**2)
            runTime[ichan,isnr,ialg] = time.time()-t0            
            if alg in ["dir","sOMP","sISTA","sFISTA","sAMP"]:                
                Npaths[ichan,isnr,ialg] = len(np.where(np.abs(hest)>0)[0])
            else:          
                Npaths[ichan,isnr,ialg] = len(paths.delays)
       
    for ialg in range(Nalg):    
        alg=algList[ialg]                
        if alg=="OMPx1":
            dicAcc.freeCacheOfPilot(ichan,(Nt,Na,Nd),(Nt,Na,Nd))
        elif alg=="OMPx2":  
            dicAcc.freeCacheOfPilot(ichan,(Nt,Na,Nd),(2*Nt,2*Na,2*Nd))
        elif alg=="OMPx4":  
            dicAcc.freeCacheOfPilot(ichan,(Nt,Na,Nd),(4*Nt,Na,Nd))
        elif alg=="OMPBR":
            dicAcc.freeCacheOfPilot(ichan,(Nt,Na,Nd),(Nt,Na,Nd))
        elif alg=="OMPx4a": 
            dicAcc.freeCacheOfPilot(ichan,(Nt,Na,Nd),(4*Nt,Na,Nd))
    bar.next()
bar.finish()

print(np.mean(MSE,axis=0))
print(np.mean(runTime,axis=0))
plt.semilogy(10*np.log10(SNRs),np.mean(MSE,axis=0))
plt.legend(algList)
plt.xlabel('SNR(dB)')
plt.ylabel('MSE')
plt.figure()
barwidth= 0.9/Nalg * (np.mean(np.diff(10*np.log10(SNRs))) if len(SNRs)>1 else 1)
for ialg in range(Nalg):
    offset=(ialg-(Nalg-1)/2)*barwidth
    plt.bar(10*np.log10(SNRs)+offset,np.mean(runTime[:,:,ialg],axis=0),width=barwidth,label=algList[ialg])
plt.xlabel('SNR(dB)')
plt.ylabel('runtime')
plt.legend(algList)
plt.figure()
barwidth=0.9/Nalg * np.mean(np.diff(10*np.log10(SNRs)))
for ialg in range(1,Nalg):
    offset=(ialg-(Nalg-1)/2)*barwidth
    plt.bar(10*np.log10(SNRs)+offset,np.mean(Npaths[:,:,ialg],axis=0),width=barwidth,label=algList[ialg])
plt.xlabel('SNR(dB)')
plt.ylabel('N paths')
plt.legend(algList[1:])
plt.show()
