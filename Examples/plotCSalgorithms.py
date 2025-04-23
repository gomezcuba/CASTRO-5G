#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import time
from tqdm import tqdm

import sys
sys.path.append('../')
from CASTRO5G import compressedSensingTools as cs
from CASTRO5G import multipathChannel as mc

plt.close('all')

Nchan=10
Nd=4
Na=4
Ncp=16
Nsym=5
Nrft=1
Nrfr=2
K=64
Ts=2.5
Ds=Ts*Ncp
SNRs=10**(np.arange(-1,2.01,1.0))
#SNRs=10**(np.arange(1,1.01,1.0))
# SNRs=np.array([100])

omprunner = cs.CSDictionaryRunner()
dicBase=cs.CSCachedDictionary()
dicMult=cs.CSMultiDictionary()
dicFFT=cs.CSBasicFFTDictionary()
dicFast=cs.CSMultiFFTDictionary()
pilgen = mc.MIMOPilotChannel("IDUV")
model=mc.DiscreteMultipathChannelModel(dims=(Ncp,Na,Nd),fftaxes=())
listPreparedChannels = []
for ichan in range(Nchan):    
    hsparse=model.getDEC(10)
    itdoa,iaoa,iaod=np.where(hsparse!=0)
    pathsparse=pd.DataFrame({        
        "coefs" : hsparse[itdoa,iaoa,iaod],
        "TDoA"  : itdoa,
        "AoA"   : np.arcsin(2*iaoa/Na-2*(iaoa>=Na/2)),
        "AoD"   : np.arcsin(2*iaod/Nd-2*(iaod>=Nd/2))
        })
    hk=np.fft.fft(hsparse,K,axis=0)
    hk=np.fft.fft(hk,Na,axis=1,norm="ortho")
    hk=np.fft.fft(hk,Nd,axis=2,norm="ortho")
    zp=mc.AWGN((Nsym,K,Na,1))
    (wp,vp)=pilgen.generatePilots(Nsym*K*Nrft,Na,Nd,Npr=Nsym*K*Nrfr,rShape=(Nsym,K,Nrfr,Na),tShape=(Nsym,K,Nd,Nrft))
    listPreparedChannels.append( (pathsparse,hsparse,hk,zp,wp,vp ) )

confAlgs = [
        ("dir" ,'AWGN',None,'-.','p','k'),
        ("sOMP",'callF',lambda v,xi: cs.simplifiedOMP(v,xi*Ncp*Na*Nd),':','p','k'),
        # ("sISTA",'callF',lambda v,xi: cs.simplifiedISTA(v,.5*np.sqrt(xi),15),'-','o','b'),
        # ("sFISTA",'callF',lambda v,xi: cs.simplifiedFISTA(v,.5*np.sqrt(xi),15),'-','o','b'),
        # ("sAMP",'callF',lambda v,xi: cs.simplifiedAMP(v,.5*np.sqrt(xi),15),'-','o','b'),
        # ("OMPx1",'runGreedy',1.0,1.0,1.0,1.0,dicBase,':','o','b'),
        # ("OMPx2",'runGreedy',2.0,2.0,2.0,1.0,dicBase,':','s','c'),
        # ("OMPx4",'runGreedy',4.0,4.0,4.0,1.0,dicBase,':','*','r'),
        # ("OMPBR",'runGreedy',1.0,1.0,1.0,10.0,dicBase,':','^','g'),
        # ("OMPx1a",'runGreedy',1.0,1.0,1.0,1.0,dicFFT,'-.','o','b'),
        # ("OMPx2a",'runGreedy',2.0,2.0,2.0,1.0,dicFFT,'-.','s','c'),
        # ("OMPx4a",'runGreedy',4.0,4.0,4.0,1.0,dicFFT,'-.','*','r'),
        # ("OMPBRa",'runGreedy',1.0,1.0,1.0,10.0,dicFFT,'-.','^','g'),
        ("OMPx1m",'runGreedy',1.0,1.0,1.0,1.0,dicMult,'--','o','b'),
        ("OMPx2m",'runGreedy',2.0,2.0,2.0,1.0,dicMult,'--','s','c'),
        # ("OMPx4m",'runGreedy',4.0,4.0,4.0,1.0,dicMult,'--','*','r'),
        # ("OMPBRm",'runGreedy',1.0,1.0,1.0,10.0,dicMult,'--','^','g'),
        # ("OMPx1f",'runGreedy',1.0,1.0,1.0,1.0,dicFast,'-','o','b'),
        # ("OMPx2f",'runGreedy',2.0,2.0,2.0,1.0,dicFast,'-','s','c'),
        # ("OMPx4f",'runGreedy',4.0,4.0,4.0,1.0,dicFast,'-','*','r'),
        # ("OMPBRf",'runGreedy',1.0,1.0,1.0,10.0,dicFast,'-','^','g'),
        # ("ISTAx1",'runShrink',1.0,1.0,1.0,'ISTA',dicBase,':','o','r'),
        # ("ISTAx2",'runShrink',2.0,2.0,2.0,'ISTA',dicBase,':','o','r'),
        # ("FISTAx1",'runShrink',1.0,1.0,1.0,'FISTA',dicBase,':','o','r'),
        # ("AMPx1",'runShrink',1.0,1.0,1.0,'AMP',dicBase,':','o','r'),
        # ("VAMPx1",'runShrink',1.0,1.0,1.0,'VAMP',dicBase,':','o','r'),
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
pathResults={}

#-------------------------------------------------------------------------------
#pregenerate the H dics (pilot independen)
for ialg in tqdm(range(Nalg),desc="Dictionary Preconfig: "):
    t0 = time.time()
    algParam = confAlgs[ialg]
    behavior = algParam[1]
    if behavior in ["runGreedy","runShrink"]:     
        Xt,Xa,Xd,_,dicObj,_,_,_ = algParam[2:]
        Lt,La,Ld=(int(Ncp*Xt),int(Na*Xa),int(Nd*Xd))
        dicObj.setHDic((K,Ncp,Na,Nd),(Lt,La,Ld))# duplicates handled by cache
        if isinstance(dicObj.currHDic.mPhiH,np.ndarray):
            sizeHDic[ialg] = dicObj.currHDic.mPhiH.size
        elif isinstance(dicObj.currHDic.mPhiH,tuple):
            sizeHDic[ialg] = np.sum([x.size for x in dicObj.currHDic.mPhiH])
        else:
            sizeHDic[ialg] = 0
    prepHTime[ialg] = time.time()-t0            
    
#-------------------------------------------------------------------------------
for ichan in tqdm(range(Nchan),desc="CS Sims: "):
    (pathsparse,hsparse,hk,zp,wp,vp)=listPreparedChannels[ichan]
    zp_bb=np.matmul(wp,zp)
    yp_noiseless=pilgen.applyPilotChannel(hk,wp,vp,None)
    zh=mc.AWGN((Ncp,Na,Nd))
    #--------------------------------------------------------------------------
    #pregenerate the Y dics (SINR independen)
    for ialg in range(Nalg):     
        t0 = time.time()
        algParam = confAlgs[ialg]  
        behavior = algParam[1]          
        if behavior in ["runGreedy","runShrink"]:        
            Xt,Xa,Xd,_,dicObj,_,_,_ = algParam[2:]
            Lt,La,Ld=(int(Ncp*Xt),int(Na*Xa),int(Nd*Xd))
            dicObj.setHDic((K,Ncp,Na,Nd),(Lt,La,Ld))# should be cached here
            dicObj.setYDic(ichan,(wp,vp))
            if isinstance(dicObj.currYDic.mPhiY,np.ndarray):
                sizeYDic[ialg] = dicObj.currYDic.mPhiY.size
            elif isinstance(dicObj.currYDic.mPhiY,tuple):
                sizeYDic[ialg] = np.sum([x.size for x in dicObj.currYDic.mPhiY])
            else:
                sizeYDic[ialg] = 0    
        prepYTime[ichan,ialg] = time.time()-t0            
    #--------------------------------------------------------------------------
    for isnr in range(Nsnr):
        sigma2=1.0/SNRs[isnr]
        yp=yp_noiseless+zp_bb*np.sqrt(sigma2)
        hnoised=hsparse*np.sqrt(Ncp)+zh*np.sqrt(sigma2)
        ialg=0
        for ialg in range(Nalg):
            t0 = time.time()
            algParam = confAlgs[ialg]
            behavior = algParam[1]
            if behavior == 'AWGN':
                hest=hnoised
                paths=None
                Npaths[ichan,isnr,ialg] = 0
            elif behavior == 'callF':
                funHandle = algParam[2]
                hest=funHandle(hnoised,sigma2)
                paths=np.where(np.abs(hest)>0)
                Npaths[ichan,isnr,ialg] = len(paths[0])
            elif behavior == 'runGreedy':
                Xt,Xa,Xd,Xr,dicObj,_,_,_ = algParam[2:]
                omprunner.setDictionary(dicObj)
                hest,paths,_,_=omprunner.OMP(yp,sigma2*K*Nsym*Nrfr,ichan,vp,wp,Xt,Xa,Xd,Xr,Ncp)
                Npaths[ichan,isnr,ialg] = len(paths.TDoA)
            elif behavior == 'runShrink':
                Xt,Xa,Xd,modeName,dicObj,_,_,_ = algParam[2:]
                omprunner.setDictionary(dicObj)
                hest,paths,_,_=omprunner.Shrinkage(yp, (.05*np.sqrt(sigma2), .5) ,15,ichan,vp,wp,Xt,Xa,Xd,modeName)                
                Npaths[ichan,isnr,ialg] = len(paths.TDoA)
            runTime[ichan,isnr,ialg] = time.time()-t0
            pathResults[(ichan,isnr,ialg)]=(hest,paths)
            if behavior in ["AWGN","callF"]:                
                horig=hsparse*np.sqrt(Ncp)
            else:
                horig=hk
            MSE[ichan,isnr,ialg] = np.mean(np.abs(horig-hest)**2)/np.mean(np.abs(horig)**2)
            ialg+=1
       
    for ialg in range(Nalg):
        algParam = confAlgs[ialg]  
        behavior = algParam[1]             
        if behavior in ["runGreedy","runShrink"]:
            Xt,Xa,Xd,_,dicObj,_,_,_ = algParam[2:]
            Lt,La,Ld=(int(Ncp*Xt),int(Nd*Xd),int(Na*Xa))
            dicObj.freeCacheOfPilot(ichan,(Ncp,Na,Nd),(Xt*Ncp,Xa*Na,Xd*Nd))

outputFileTag=f'{Nsym}-{K}-{Ncp}-{Nrfr}-{Na}-{Nd}-{Nrfr}'
bytesPerFloat = np.array([0],dtype=np.complex128).itemsize
algLegendList = [x[0] for x in confAlgs]

plt.figure()
plt.yscale("log")
barwidth= 0.9/2
offset=(-1/2)*barwidth
plt.bar(np.arange(len(algLegendList[:]))+offset,bytesPerFloat*sizeHDic[:]*(2.0**-20),width=barwidth,label='H dict')
offset=(+1/2)*barwidth
plt.bar(np.arange(len(algLegendList[:]))+offset,bytesPerFloat*sizeYDic[:]*(2.0**-20),width=barwidth,label='Y dict')
plt.xticks(ticks=np.arange(len(algLegendList[:])),labels=algLegendList[:])
plt.xlabel('Algoritm')
plt.ylabel('Dictionary size MByte')
plt.legend()
plt.savefig(f'../Figures/basic_DicMBvsAlg-{outputFileTag}.svg')
plt.figure()
plt.yscale("log")
barwidth= 0.9/2
offset=(-1/2)*barwidth
plt.bar(np.arange(len(algLegendList[:]))+offset,prepHTime[:],width=barwidth,label='H dict')
offset=(+1/2)*barwidth
plt.bar(np.arange(len(algLegendList[:]))+offset,np.mean(prepYTime[:,:],axis=0),width=barwidth,label='Y dict')
plt.xticks(ticks=np.arange(len(algLegendList[:])),labels=algLegendList[:])
plt.xlabel('Algoritm')
plt.ylabel('precomputation time')
plt.legend()
plt.savefig(f'../Figures/basic_DicCompvsAlg-{outputFileTag}.svg')
plt.figure()
for ialg in range(Nalg):
    lin,mrk,clr = confAlgs[ialg][-3:]
    plt.semilogy(10*np.log10(SNRs),np.mean(MSE[:,:,ialg],axis=0),color=clr,marker=mrk,linestyle=lin,label=algLegendList[ialg])
plt.legend()
plt.xlabel('SNR(dB)')
plt.ylabel('MSE')
plt.savefig(f'../Figures/basic_MSEvsSNR-{outputFileTag}.svg')
plt.figure()
plt.yscale("log")
barwidth= 0.9/Nalg * (np.mean(np.diff(10*np.log10(SNRs))) if len(SNRs)>1 else 1)
for ialg in range(Nalg):
    offset=(ialg-(Nalg-1)/2)*barwidth
    plt.bar(10*np.log10(SNRs)+offset,np.mean(runTime[:,:,ialg],axis=0),width=barwidth,label=algLegendList[ialg],color=cm.tab20(ialg/20))
plt.xlabel('SNR(dB)')
plt.ylabel('runtime')
plt.legend(algLegendList)
plt.savefig(f'../Figures/basic_CSCompvsSNR-{outputFileTag}.svg')
plt.figure()
barwidth=0.9/Nalg * np.mean(np.diff(10*np.log10(SNRs)))
for ialg in range(Nalg):
    offset=(ialg-(Nalg-1)/2)*barwidth
    plt.bar(10*np.log10(SNRs)+offset,np.mean(Npaths[:,:,ialg],axis=0),width=barwidth,label=algLegendList[ialg],color=cm.tab20(ialg/20))
plt.xlabel('SNR(dB)')
plt.ylabel('N paths')
plt.legend()
plt.savefig(f'../Figures/basic_NpathvsSNR-{outputFileTag}.svg')
# plt.show()
