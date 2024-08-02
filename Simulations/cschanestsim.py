#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

import time
import os
from tqdm import tqdm
import argparse

import sys
sys.path.append('../')
from CASTRO5G import compressedSensingTools as cs
from CASTRO5G import multipathChannel as mc
from CASTRO5G import threeGPPMultipathGenerator as mp3g
parser = argparse.ArgumentParser(description='MIMO OFDM CS Multipath Channel Estimation Simulator')
#parameters that affect number of simulations
parser.add_argument('-N', type=int,help='No. simulated channels')
parser.add_argument('--z3D', action='store_true', help='Activate 3d simulation mode')

#parameters that affect frame
parser.add_argument('-F', type=str,help='comma-separated list of frame dimensions')

#parameters that affect multipath generator
parser.add_argument('-G', type=str,help='Type of generator. "3gpp" or "Geo:N" for N scatterers contained in Map')

#parameters that affect estimation algorithms
parser.add_argument('-A', type=str,help='comma-separated list of algorithms')

#parameters that affect plots

#parameters that affect workflow
parser.add_argument('--label', type=str,help='str label appended to storage files')
parser.add_argument('--nosave', help='Do not save simulation data to new results file', action='store_true')
parser.add_argument('--nompg',help='Do not perform multipath generation, load existing file', action='store_true')
parser.add_argument('--noest',help='Do not perform location estimation, load existing file', action='store_true')
parser.add_argument('--show', help='Open plot figures during execution', action='store_true')
parser.add_argument('--print', help='Save plot files in svg to results folder', action='store_true')

plt.close('all')

#TODO: implement the above command line API and generate plots for publication

Nchan=10
#()
Nd=8 #Nt (Ntv*Nth) con Ntv=1
Na=8 #Nr (Nrv*Nrh) con Ntv=1
Nt=32
Nframe=3
Nrft=1
Nrfr=2
K=64
#Ts=2.5
Ts=320/Nt
Ds=Ts*Nt
SNRs=10**(np.arange(-1,2.01,1.0))
#SNRs=10**(np.arange(1,1.01,1.0))

omprunner = cs.CSDictionaryRunner()
dicBase=cs.CSCachedDictionary()
dicMult=cs.CSMultiDictionary()
dicFFT=cs.CSBasicFFTDictionary()
dicFast=cs.CSMultiFFTDictionary()
pilgen = mc.MIMOPilotChannel("IDUV")
chgen = mp3g.ThreeGPPMultipathChannelModel()
chgen.bLargeBandwidthOption=True

x0=np.random.rand(Nchan)*100-50
y0=np.random.rand(Nchan)*100-50

confAlgs=[#Xt Xd Xa Xmu accel legend string name
    # (1.0,1.0,1.0,1.0,dicBase,'OMPx1',':','o','b'),
    # (4.0,1.0,1.0,1.0,dicBase,'OMPx4T',':','p','y'),
    # (4.0,1.0,1.0,1.0,dicFFT,'OMPx4Ta',':','d','kb'),
    # (4.0,4.0,4.0,1.0,dicBase,'OMPx4',':','*','r'),
    # (1.0,1.0,1.0,100.0,dicBase,'OMPBR',':','^','g'),
    # (1.0,1.0,1.0,1.0,dicFFT,'OMPx1a','-.','o','b'),
    # (4.0,4.0,4.0,1.0,dicFFT,'OMPx4a','-.','*','r'),
    # (1.0,1.0,1.0,10.0,dicFFT,'OMPBRa','-.','^','g'),
    (1.0,1.0,1.0,1.0,dicMult,'OMPx1m','--','o','b'),
    # (4.0,4.0,4.0,1.0,dicMult,'OMPx4m','--','*','r'),
    # (8.0,8.0,8.0,1.0,dicMult,'OMPx8m','--','*','r'),
    # (1.0,1.0,1.0,10.0,dicMult,'OMPBRm','--','^','g'),
    (1.0,1.0,1.0,1.0,dicFast,'OMPx1f','-','o','b'),
    (4.0,4.0,4.0,1.0,dicFast,'OMPx4f','-','*','r'),
    # (8.0,8.0,8.0,1.0,dicFast,'OMPx8f','-','*','r'),
    (1.0,1.0,1.0,10.0,dicFast,'OMPBRf','-','^','g'),
    ]

legStrAlgs=[x[-1] for x in confAlgs]
Nalgs=len(confAlgs)
Nsnr=len(SNRs)
MSE=np.zeros((Nchan,Nsnr,Nalgs))
Npaths=np.zeros((Nchan,Nsnr,Nalgs))
prepYTime=np.zeros((Nchan,Nalgs))
prepHTime=np.zeros((Nalgs))
sizeYDic=np.zeros((Nalgs))
sizeHDic=np.zeros((Nalgs))
runTimes=np.zeros((Nchan,Nsnr,Nalgs))


#-------------------------------------------------------------------------------
#pregenerate the H dics (pilot independen)
for ialg in tqdm(range(Nalgs),desc="Dictionary Preconfig: "):
    t0 = time.time()
    Xt,Xa,Xd,_,dicObj,_,_,_,_ = confAlgs[ialg]
    Lt,La,Ld=(int(Nt*Xt),int(Na*Xa),int(Nd*Xd))
    dicObj.setHDic((K,Nt,Na,Nd),(Lt,La,Ld))# duplicates handled by cache
    if isinstance(dicObj.currHDic.mPhiH,np.ndarray):
        sizeHDic[ialg] = dicObj.currHDic.mPhiH.size
    elif isinstance(dicObj.currHDic.mPhiH,tuple):
        sizeHDic[ialg] = np.sum([x.size for x in dicObj.currHDic.mPhiH])
    else:
        sizeHDic[ialg] = 0
    prepHTime[ialg] = time.time()-t0            
    
#-------------------------------------------------------------------------------

for ichan in  tqdm(range(Nchan),desc="CS Sims: "):
    
    model = mp3g.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=True)
    plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
    tau,powC,AOA,AOD,ZOA,ZOD = clusters.T.to_numpy()
    los, PLfree, SF = plinfo
    if los:
        #disregard los channels in this case
        tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp,XPR_sp,phase00,phase01,phase10,phase11 = subpaths.drop(index=(0,40)).T.to_numpy()
        pow_sp=pow_sp/np.sum(pow_sp)
    else:
        tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp,XPR_sp,phase00,phase01,phase10,phase11 = subpaths.T.to_numpy()

    mpch = mc.MultipathChannel((0,0,10),(40,0,1.5),[])
    Npath = tau_sp.size
    pathAmplitudes = np.sqrt( pow_sp )*np.exp(-1j*phase00)
    mpch.insertPathsFromListParameters(pathAmplitudes,tau_sp,AOA_sp*np.pi/180,AOD_sp*np.pi/180,ZOA_sp*np.pi/180,ZOD_sp*np.pi/180,np.zeros_like(pathAmplitudes))
    ht=mpch.getDEC(Na,Nd,Nt,Ts)*np.sqrt(Nd*Na)#mpch uses normalized matrices of gain 1
    hk=np.fft.fft(ht.transpose([2,0,1]),K,axis=0)
        
    (wp,vp)=pilgen.generatePilots(Nframe*K*Nrft,Na,Nd,Npr=Nframe*K*Nrfr,rShape=(Nframe,K,Nrfr,Na),tShape=(Nframe,K,Nd,Nrft))

    zp=mc.AWGN((Nframe,K,Na,1))
    zp_bb=np.matmul(wp,zp)
    yp_noiseless=pilgen.applyPilotChannel(hk,wp,vp,None)
    
    
    for ialg in range(0,Nalgs):
        t0 = time.time()
        Xt,Xa,Xd,Xmu,confDic,label,_,_,_ = confAlgs[ialg]
        Lt,La,Ld=(int(Nt*Xt),int(Na*Xa),int(Nd*Xd))
        confDic.setHDic((K,Nt,Na,Nd),(Lt,La,Ld))# should be cached at this point
        confDic.setYDic(ichan,(wp,vp))
        if isinstance(confDic.currYDic.mPhiY,np.ndarray):
            sizeYDic[ialg] = confDic.currYDic.mPhiY.size
        elif isinstance(confDic.currYDic.mPhiY,tuple):
            sizeYDic[ialg] = np.sum([x.size for x in confDic.currYDic.mPhiY])
        else:
            sizeYDic[ialg] = 0    
        prepYTime[ichan,ialg] = time.time()-t0         
    for isnr in range(0,Nsnr):
        sigma2=1.0/SNRs[isnr]
        yp=yp_noiseless+zp_bb*np.sqrt(sigma2)
        for nalg in range(0,Nalgs):
            Xt,Xa,Xd,Xmu,confDic,label,_,_,_ = confAlgs[nalg]
            t0 = time.time()
            omprunner.setDictionary(confDic)
            hest,paths,_,_=omprunner.OMP(yp,sigma2*K*Nframe*Nrfr,ichan,vp,wp, Xt,Xa,Xd,Xmu,Nt)
            MSE[ichan,isnr,nalg] = np.mean(np.abs(hk-hest)**2)/np.mean(np.abs(hk)**2)
            runTimes[ichan,isnr,nalg] = time.time()-t0
            Npaths[ichan,isnr,nalg] = len(paths.TDoA)
    #for large Nsims the pilot cache grows too much so we free the memory when not needed
    for nalg in range(0,Nalgs):
        Xt,Xa,Xd,Xmu,confDic,label,_,_,_ = confAlgs[nalg]
        confDic.freeCacheOfPilot(ichan,(Nt,Na,Nd),(int(Nt*Xt),int(Na*Xa),int(Nd*Xd)))
        
outputFileTag=f'{Nframe}-{K}-{Nt}-{Nrfr}-{Na}-{Nd}-{Nrfr}'
bytesPerFloat = np.array([0],dtype=np.complex128).itemsize
algLegendList = [x[5] for x in confAlgs]

plt.figure()
plt.yscale("log")
barwidth= 0.9/2
offset=(-1/2)*barwidth
plt.bar(np.arange(len(algLegendList))+offset,bytesPerFloat*sizeHDic*(2.0**-20),width=barwidth,label='H dict')
offset=(+1/2)*barwidth
plt.bar(np.arange(len(algLegendList))+offset,bytesPerFloat*sizeYDic*(2.0**-20),width=barwidth,label='Y dict')
plt.xticks(ticks=np.arange(len(algLegendList)),labels=algLegendList)
plt.xlabel('Algoritm')
plt.ylabel('Dictionary size MByte')
plt.legend()
plt.savefig(f'../Figures/3gpp_DicMBvsAlg-{outputFileTag}.svg')
plt.figure()
plt.yscale("log")
barwidth= 0.9/2
offset=(-1/2)*barwidth
plt.bar(np.arange(len(algLegendList))+offset,prepHTime,width=barwidth,label='H dict')
offset=(+1/2)*barwidth
plt.bar(np.arange(len(algLegendList))+offset,np.mean(prepYTime[:,:],axis=0),width=barwidth,label='Y dict')
plt.xticks(ticks=np.arange(len(algLegendList)),labels=algLegendList)
plt.xlabel('Algoritm')
plt.ylabel('precomputation time')
plt.legend()
plt.savefig(f'../Figures/3gpp_DicCompvsAlg-{outputFileTag}.svg')
plt.figure()
for ialg in range(Nalgs):
    Xt,Xd,Xa,Xmu,confDic,label,lin,mrk,clr = confAlgs[ialg][:]
    plt.semilogy(10*np.log10(SNRs),np.mean(MSE[:,:,ialg],axis=0),color=clr,marker=mrk,linestyle=lin,label=label)
plt.legend()
plt.xlabel('SNR(dB)')
plt.ylabel('MSE')
plt.savefig(f'../Figures/3gpp_MSEvsSNR-{outputFileTag}.svg')
plt.figure()
plt.yscale("log")
barwidth= 0.9/Nalgs * (np.mean(np.diff(10*np.log10(SNRs))) if len(SNRs)>1 else 1)
for ialg in range(Nalgs):
    offset=(ialg-(Nalgs-1)/2)*barwidth
    plt.bar(10*np.log10(SNRs)+offset,np.mean(runTimes[:,:,ialg],axis=0),width=barwidth,label=algLegendList[ialg])
plt.xlabel('SNR(dB)')
plt.ylabel('runtime')
plt.legend(algLegendList)
plt.savefig(f'../Figures/3gpp_CSCompvsSNR-{outputFileTag}.svg')
plt.figure(5)
barwidth=0.9/Nalgs * np.mean(np.diff(10*np.log10(SNRs)))
for ialg in range(0,Nalgs):
    offset=(ialg-(Nalgs-1)/2)*barwidth
    plt.bar(10*np.log10(SNRs)+offset,np.mean(Npaths[:,:,ialg],axis=0),width=barwidth,label=algLegendList[ialg])
plt.xlabel('SNR(dB)')
plt.ylabel('N paths')
plt.legend()
plt.savefig(f'../Figures/3gpp_NpathvsSNR-{outputFileTag}.svg')
plt.show()
