#!/usr/bin/python

from CASTRO5G import OMPCachedRunner as oc
from CASTRO5G import multipathChannel as mc

import matplotlib.pyplot as plt
import numpy as np

import time
from tqdm import tqdm

plt.close('all')

Nchan=10
Nd=8
Na=8
Nt=32
Nsym=3
Nrft=1
Nrfr=2
K=64
Ts=2.5
Ds=Ts*Nt
SNRs=10**(np.arange(-1,2.01,1.0))
#SNRs=10**(np.arange(1,1.01,1.0))
# SNRs=np.array([100])

omprunner = oc.OMPCachedRunner()
dicBase=oc.CSCachedDictionary()
dicMult=oc.CSMultiDictionary()
dicAcc=oc.CSAccelDictionary()
pilgen = mc.MIMOPilotChannel("IDUV")
model=mc.DiscreteMultipathChannelModel(dims=(Nt,Na,Nd),fftaxes=(1,2))
listPreparedProblems = []
for ichan in range(Nchan):    
    h=model.getDEC(3)
    hk=np.fft.fft(h,K,axis=0)
    zp=mc.AWGN((Nsym,K,Na,1))
    (wp,vp)=pilgen.generatePilots(Nsym*K*Nrft,Na,Nd,Npr=Nsym*K*Nrfr,rShape=(Nsym,K,Nrfr,Na),tShape=(Nsym,K,Nd,Nrft))
    listPreparedProblems.append( (hk,zp,wp,vp ) )

confAlgs = [
        ("dir" ,'AWGN',None),
        ("sOMP",'callF',lambda v,xi: oc.simplifiedOMP(v,xi*Nt*Na*Nd)),
        # ("sISTA",'callF',lambda v,xi: oc.simplifiedISTA(v,.5*np.sqrt(xi),15)),
        # ("sFISTA",'callF',lambda v,xi: oc.simplifiedFISTA(v,.5*np.sqrt(xi),15)),
        # ("sAMP",'callF',lambda v,xi: oc.simplifiedAMP(v,.5*np.sqrt(xi),15)),
        ("OMPx1",'runGreedy',1.0,1.0,1.0,1.0,dicBase),
        ("OMPx1m",'runGreedy',1.0,1.0,1.0,1.0,dicMult),
        # ("OMPx2",'runGreedy',2.0,2.0,2.0,1.0,dicBase),
        ("OMPx4",'runGreedy',4.0,4.0,4.0,1.0,dicBase),
        ("OMPx4m",'runGreedy',4.0,4.0,4.0,1.0,dicMult),
        ("OMPBR",'runGreedy',1.0,1.0,1.0,10.0,dicBase),
        ("OMPx4a",'runGreedy',4.0,4.0,4.0,1.0,dicAcc),
        # ("ISTAx1",'runShrink',1.0,1.0,1.0,'ISTA',dicBase),
        # ("ISTAx2",'runShrink',2.0,2.0,2.0,'ISTA',dicBase),
        # ("FISTAx1",'runShrink',1.0,1.0,1.0,'FISTA',dicBase),
        # ("AMPx1",'runShrink',1.0,1.0,1.0,'AMP',dicBase),
        # ("VAMPx1",'runShrink',1.0,1.0,1.0,'VAMP',dicBase),
    ]

Nalg=len(confAlgs)
Nsnr=len(SNRs)
MSE=np.zeros((Nchan,Nsnr,Nalg))
prepYTime=np.zeros((Nchan,Nalg))
prepHTime=np.zeros((Nalg))
sizeYDic=np.zeros((Nalg))
sizeHDic=np.zeros((Nalg))
runTime=np.zeros((Nchan,Nsnr,Nalg))
Npaths=np.zeros((Nchan,Nsnr,Nalg))

#-------------------------------------------------------------------------------
#pregenerate the H dics (pilot independen)
for ialg in range(Nalg):
    t0 = time.time()
    algParam = confAlgs[ialg]
    behavior = algParam[1]
    if behavior in ["runGreedy","runShrink"]:     
        Xt,Xa,Xd,_,dicObj = algParam[2:]
        Lt,La,Ld=(int(Nt*Xt),int(Nd*Xd),int(Na*Xa))
        dicObj.setHDic((K,Nt,Na,Nd),(Lt,La,Ld))# duplicates handled by cache
        if isinstance(dicObj.currHDic.mPhiH,np.ndarray):
            sizeHDic[ialg] = dicObj.currHDic.mPhiH.size
        else:
            sizeHDic[ialg] = np.sum([x.size for x in dicObj.currHDic.mPhiH])
    prepHTime[ialg] = time.time()-t0            
    
#-------------------------------------------------------------------------------
for ichan in tqdm(range(Nchan),desc="CS Sims: "):
    (hk,zp,wp,vp)=listPreparedProblems[ichan]
    zp_bb=np.matmul(wp,zp)
    yp_noiseless=pilgen.applyPilotChannel(hk,wp,vp,None)
    zh=mc.AWGN((Nt,Na,Nd))
        
    hsparse=np.fft.ifft(hk,axis=0)[0:Nt,:,:]
    hsparse=np.fft.ifft(hsparse,axis=1)*np.sqrt(Nd)
    hsparse=np.fft.ifft(hsparse,axis=2)*np.sqrt(Na)    
    #--------------------------------------------------------------------------
    #pregenerate the Y dics (SINR independen)
    for ialg in range(Nalg):     
        t0 = time.time()
        algParam = confAlgs[ialg]  
        behavior = algParam[1]          
        if behavior in ["runGreedy","runShrink"]:        
            Xt,Xa,Xd,_,dicObj = algParam[2:]
            Lt,La,Ld=(int(Nt*Xt),int(Nd*Xd),int(Na*Xa))
            dicObj.setHDic((K,Nt,Na,Nd),(Lt,La,Ld))# should be cached here
            dicObj.setYDic(ichan,(wp,vp))
            if isinstance(dicObj.currYDic.mPhiY,np.ndarray):
                sizeYDic[ialg] = dicObj.currYDic.mPhiY.size
            else:
                sizeYDic[ialg] = np.sum([x.size for x in dicObj.currYDic.mPhiY])
        prepYTime[ichan,ialg] = time.time()-t0            
    #--------------------------------------------------------------------------
    for isnr in range(Nsnr):
        sigma2=1.0/SNRs[isnr]
        yp=yp_noiseless+zp_bb*np.sqrt(sigma2)
        hnoised=hsparse*np.sqrt(Nt)+zh*np.sqrt(sigma2)
        ialg=0
        for ialg in range(Nalg):
            t0 = time.time()
            algParam = confAlgs[ialg]
            behavior = algParam[1]
            if behavior in ["AWGN","callF"]:                
                horig=hsparse*np.sqrt(Nt)
            else:
                horig=hk
            if behavior == 'AWGN':
                hest=hnoised
            elif behavior == 'callF':
                funHandle = algParam[2]
                hest=funHandle(hnoised,sigma2)
            elif behavior == 'runGreedy':
                Xt,Xa,Xd,Xr,dicObj = algParam[2:]
                omprunner.setDictionary(dicObj)
                hest,paths=omprunner.OMPBR(yp,sigma2*K*Nsym*Nrfr,ichan,vp,wp,Xt,Xa,Xd,Xr,Nt)
            elif behavior == 'runShrink':
                Xt,Xa,Xd,modeName,dicObj = algParam[2:]
                omprunner.setDictionary(dicObj)
                hest,paths=omprunner.Shrinkage(yp, (.5*np.sqrt(sigma2), .5) ,15,ichan,vp,wp,Xt,Xa,Xd,modeName)                
                    
            MSE[ichan,isnr,ialg] = np.mean(np.abs(horig-hest)**2)/np.mean(np.abs(horig)**2)
            runTime[ichan,isnr,ialg] = time.time()-t0            
            if behavior in ["AWGN","callF"]:               
                Npaths[ichan,isnr,ialg] = len(np.where(np.abs(hest)>0)[0])
            else:          
                Npaths[ichan,isnr,ialg] = len(paths.delays)                
            ialg+=1
       
    for ialg in range(Nalg):
        algParam = confAlgs[ialg]  
        behavior = algParam[1]             
        if behavior in ["runGreedy","runShrink"]:
            Xt,Xa,Xd,_,dicObj = algParam[2:]
            Lt,La,Ld=(int(Nt*Xt),int(Nd*Xd),int(Na*Xa))
            dicObj.freeCacheOfPilot(ichan,(Nt,Na,Nd),(Xt*Nt,Xa*Na,Xd*Nd))

outputFileTag=f'{Nsym}-{K}-{Nt}-{Nrfr}-{Na}-{Nd}-{Nrfr}'
bytesPerFloat = dicBase.currHDic.mPhiH[0].itemsize
algLegendList = [x[0] for x in confAlgs]

plt.figure()
plt.yscale("log")
barwidth= 0.9/2
offset=(-1/2)*barwidth
plt.bar(np.arange(len(algLegendList[2:]))+offset,bytesPerFloat*sizeHDic[2:]*(2.0**-20),width=barwidth,label='H dict')
offset=(+1/2)*barwidth
plt.bar(np.arange(len(algLegendList[2:]))+offset,bytesPerFloat*sizeYDic[2:]*(2.0**-20),width=barwidth,label='Y dict')
plt.xticks(ticks=np.arange(len(algLegendList[2:])),labels=algLegendList[2:])
plt.xlabel('Algoritm')
plt.ylabel('Dictionary size MByte')
plt.legend()
plt.savefig(f'DicMBvsAlg-{outputFileTag}.svg')
plt.figure()
plt.yscale("log")
barwidth= 0.9/2
offset=(-1/2)*barwidth
plt.bar(np.arange(len(algLegendList[2:]))+offset,prepHTime[2:],width=barwidth,label='H dict')
offset=(+1/2)*barwidth
plt.bar(np.arange(len(algLegendList[2:]))+offset,np.mean(prepYTime[:,2:],axis=0),width=barwidth,label='Y dict')
plt.xticks(ticks=np.arange(len(algLegendList[2:])),labels=algLegendList[2:])
plt.xlabel('Algoritm')
plt.ylabel('precomputation time')
plt.legend()
plt.savefig(f'DicCompvsAlg-{outputFileTag}.svg')
plt.figure()
plt.semilogy(10*np.log10(SNRs),np.mean(MSE,axis=0))
plt.legend(algLegendList)
plt.xlabel('SNR(dB)')
plt.ylabel('MSE')
plt.savefig(f'MSEvsSNR-{outputFileTag}.svg')
plt.figure()
plt.yscale("log")
barwidth= 0.9/Nalg * (np.mean(np.diff(10*np.log10(SNRs))) if len(SNRs)>1 else 1)
for ialg in range(Nalg):
    offset=(ialg-(Nalg-1)/2)*barwidth
    plt.bar(10*np.log10(SNRs)+offset,np.mean(runTime[:,:,ialg],axis=0),width=barwidth,label=algLegendList[ialg])
plt.xlabel('SNR(dB)')
plt.ylabel('runtime')
plt.legend(algLegendList)
plt.savefig(f'CSCompvsSNR-{outputFileTag}.svg')
plt.figure()
barwidth=0.9/Nalg * np.mean(np.diff(10*np.log10(SNRs)))
for ialg in range(1,Nalg):
    offset=(ialg-(Nalg-1)/2)*barwidth
    plt.bar(10*np.log10(SNRs)+offset,np.mean(Npaths[:,:,ialg],axis=0),width=barwidth,label=algLegendList[ialg])
plt.xlabel('SNR(dB)')
plt.ylabel('N paths')
plt.legend(algLegendList[1:])
plt.savefig(f'NpathvsSNR-{outputFileTag}.svg')
plt.show()
