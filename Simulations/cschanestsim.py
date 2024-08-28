#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import os
from tqdm import tqdm
import argparse

import sys
sys.path.append('../')
from CASTRO5G import compressedSensingTools as cs
from CASTRO5G import multipathChannel as mc
from CASTRO5G import threeGPPMultipathGenerator as mpg
parser = argparse.ArgumentParser(description='MIMO OFDM CS Multipath Channel Estimation Simulator')
#parameters that affect number of simulations
parser.add_argument('-N', type=int,help='No. simulated channels')
#TODO
# parser.add_argument('--z3D', action='store_true', help='Activate 3d simulation mode')

#parameters that affect frame
parser.add_argument('-F', type=str,help='comma-separated list of frames dimension cases Nframe:K:Ncp,')

#parameters that affect multipath generator
parser.add_argument('-G', type=str,help='Type of generator. "3gpp" or "Uni:N" for N uniform IID paths')

#parameters that affect estimation algorithms
#TODO
# parser.add_argument('-A', type=str,help='comma-separated list of algorithms')
#TODO
# parser.add_argument('-P', type=str,help='comma-separated list of pilot schemes')

parser.add_argument('-S', type=str,help='SNR values in dB min:max:step')


#parameters that affect plots

#parameters that affect workflow
parser.add_argument('--label', type=str,help='str label appended to storage files')
parser.add_argument('--nosave', help='Do not save simulation data to new results file', action='store_true')
parser.add_argument('--nompg',help='Do not perform multipath generation, load existing file', action='store_true')
parser.add_argument('--noest',help='Do not perform channel estimation, load existing file', action='store_true')
parser.add_argument('--show', help='Open plot figures during execution', action='store_true')
parser.add_argument('--print', help='Save plot files in svg to results folder', action='store_true')

# args = parser.parse_args("-N 10 -G Uni:10 -F=3:32:16:2:8:8:1 --label testUni --show --print".split(' '))
args = parser.parse_args("-N 10 -G Uni:10 -F=5:128:32:4:8:8:1 --label testUni --show --print".split(' '))
# there are TOO MANH PATHS in 3gpp channel. this config does not have enough observations for good CS
# args = parser.parse_args("-N 10 -G 3gpp -F=3:64:32:2:8:8:1 --label test3GPP --show --print".split(' '))
# this config is a bit slow but is the minimal working one
# args = parser.parse_args("-N 10 -G 3gpp -F=5:128:32:4:8:8:1 --label test3GPP --show --print".split(' '))

plt.close('all')

#TODO: implement the above command line API and generate plots for publication

Nchan=args.N if args.N else 10
# bMode3D = args.z3D
if args.F:
    #TODO support multiple frame shapes in a single run
    frameDims = [int(x) for x in args.F.split(':')]
else:
    frameDims = [3,32,16,2,4,4,1]
Nframe,K,Ncp,Nrfr,Na,Nd,Nrft = frameDims

# Ds=390e-9 #Ts=2.5ns with Rs=400MHz in NYUwim
Tcp=570e-9 #mu=2
Ts=Tcp/Ncp
if args.S:
    minSNRdB,maxSNRdB,stepSNRdB = args.S.split(':')
else:
    minSNRdB,maxSNRdB,stepSNRdB = (-10,20,10)
SNRs=10**(np.arange(minSNRdB,maxSNRdB+0.01,stepSNRdB)/10)

# multipath generator
if args.G:
    mpgenInfo = args.G.split(':')
else:        
    mpgenInfo = ['Uni','20']
mpgen=mpgenInfo[0]

if args.label:
    outfoldername=f'../Results/CSChanEstresults{args.label}'
else:
    outfoldername=f'../Results/CSChanEstresults-{":".join([str(x) for x in frameDims])}'
if not os.path.isdir(outfoldername):
    os.mkdir(outfoldername)

omprunner = cs.CSDictionaryRunner()
dicBase=cs.CSCachedDictionary()
dicMult=cs.CSMultiDictionary()
dicFFT=cs.CSBasicFFTDictionary()
dicFast=cs.CSMultiFFTDictionary()
pilgen = mc.MIMOPilotChannel("IDUV")

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
    (4.0,4.0,4.0,1.0,dicMult,'OMPx4m','--','*','r'),
    # (8.0,8.0,8.0,1.0,dicMult,'OMPx8m','--','d','k'),
    (1.0,1.0,1.0,10.0,dicMult,'OMPBRm','--','^','g'),
    # (1.0,1.0,1.0,1.0,dicFast,'OMPx1f','-','o','b'),
    # (4.0,4.0,4.0,1.0,dicFast,'OMPx4f','-','*','r'),
    # # (8.0,8.0,8.0,1.0,dicFast,'OMPx8f','-','*','r'),
    # (1.0,1.0,1.0,10.0,dicFast,'OMPBRf','-','^','g'),
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

if args.nompg:    
    # allUserData=pd.read_csv(outfoldername+'/userGenData.csv',index_col=['ue']) 
    allPathsData=pd.read_csv(outfoldername+'/chanGenData.csv',index_col=['ue','n']) 
else:
    
    if mpgen=='3gpp':
        lMultipathParameters = []
        chgen = mpg.ThreeGPPMultipathChannelModel(scenario="UMi",bLargeBandwidthOption=True)
        for ichan in  tqdm(range(Nchan),desc="Channel Generation: "):
            chgen.initCache()#generate i.i.d. realizations of channel on the same location
            plinfo,macro,clusters,subpaths = chgen.create_channel((0,0,10),(40,0,1.5))
            los, PLfree, SF = plinfo
            if los:#we will use only NLOS channels as the SNR is normalized
                Plos = subpaths.loc[(0,40)].P
                Ptot = subpaths.P.sum()
                subpaths.drop(index=(0,40),inplace=True)
                subpaths.P=subpaths.P*Ptot/(Ptot-Plos)
            subpaths.reset_index(inplace=True)#moves cluster-subpath pairs n,m to normal columns
            subpaths.TDoA=subpaths.TDoA*.9*Tcp/subpaths.TDoA.max()
            subpaths.AoD=subpaths.AoD*np.pi/180
            subpaths.AoA=subpaths.AoA*np.pi/180
            subpaths.ZoD=subpaths.ZoD*np.pi/180
            subpaths.ZoA=subpaths.ZoA*np.pi/180
            lMultipathParameters.append(subpaths)
        allPathsData=pd.concat(lMultipathParameters,keys=np.arange(Nchan),names=["ue",'n'])    
    elif mpgen=='Uni':        
        chgen=mc.UniformMultipathChannelModel(Npath=int(mpgenInfo[1]),Ds=.9*Tcp,mode3D=False)
        allPathsData=chgen.create_channel(Nchan)
    else:
        print("Unknown multipath generator")
    if not args.nosave: 
        # allUserData.to_csv(outfoldername+'/userGenData.csv') 
        allPathsData.to_csv(outfoldername+'/chanGenData.csv') 
#-------------------------------------------------------------------------------
#pregenerate the H dics (pilot independent)
for ialg in tqdm(range(Nalgs),desc="Dictionary Preconfig: "):
    t0 = time.time()
    Xt,Xa,Xd,_,dicObj,_,_,_,_ = confAlgs[ialg]
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

for ichan in  tqdm(range(Nchan),desc="CS Sims: "):
    mpch = mc.MultipathChannel((0,0,10),(40,0,1.5),allPathsData.loc[ichan,:])
    ht=mpch.getDEC(Na,Nd,Ncp,Ts)*np.sqrt(Nd*Na)#mpch uses normalized matrices of gain 1
    hk=np.fft.fft(ht,K,axis=0)
        
    (wp,vp)=pilgen.generatePilots(Nframe*K*Nrft,Na,Nd,Npr=Nframe*K*Nrfr,rShape=(Nframe,K,Nrfr,Na),tShape=(Nframe,K,Nd,Nrft))

    zp=mc.AWGN((Nframe,K,Na,1))
    zp_bb=np.matmul(wp,zp)
    yp_noiseless=pilgen.applyPilotChannel(hk,wp,vp,None)
    
    
    for ialg in range(0,Nalgs):
        t0 = time.time()
        Xt,Xa,Xd,Xmu,confDic,label,_,_,_ = confAlgs[ialg]
        Lt,La,Ld=(int(Ncp*Xt),int(Na*Xa),int(Nd*Xd))
        confDic.setHDic((K,Ncp,Na,Nd),(Lt,La,Ld))# should be cached at this point
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
            hest,paths,_,_=omprunner.OMP(yp,sigma2*K*Nframe*Nrfr,ichan,vp,wp, Xt,Xa,Xd,Xmu,Ncp)
            MSE[ichan,isnr,nalg] = np.mean(np.abs(hk-hest)**2)/np.mean(np.abs(hk)**2)
            runTimes[ichan,isnr,nalg] = time.time()-t0
            Npaths[ichan,isnr,nalg] = len(paths.TDoA)
    #for large Nsims the pilot cache grows too much so we free the memory when not needed
    for nalg in range(0,Nalgs):
        Xt,Xa,Xd,Xmu,confDic,label,_,_,_ = confAlgs[nalg]
        confDic.freeCacheOfPilot(ichan,(Ncp,Na,Nd),(int(Ncp*Xt),int(Na*Xa),int(Nd*Xd)))
        
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
plt.savefig(outfoldername+'/DicMBvsAlg.svg')
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
plt.savefig(outfoldername+'/DicCompvsAlg.svg')
plt.figure()
for ialg in range(Nalgs):
    Xt,Xd,Xa,Xmu,confDic,label,lin,mrk,clr = confAlgs[ialg][:]
    plt.semilogy(10*np.log10(SNRs),np.mean(MSE[:,:,ialg],axis=0),color=clr,marker=mrk,linestyle=lin,label=label)
plt.legend()
plt.xlabel('SNR(dB)')
plt.ylabel('MSE')
plt.savefig(outfoldername+'/MSEvsSNR.svg')
plt.figure()
plt.yscale("log")
barwidth= 0.9/Nalgs * (np.mean(np.diff(10*np.log10(SNRs))) if len(SNRs)>1 else 1)
for ialg in range(Nalgs):
    offset=(ialg-(Nalgs-1)/2)*barwidth
    plt.bar(10*np.log10(SNRs)+offset,np.mean(runTimes[:,:,ialg],axis=0),width=barwidth,label=algLegendList[ialg])
plt.xlabel('SNR(dB)')
plt.ylabel('runtime')
plt.legend(algLegendList)
plt.savefig(outfoldername+'/CSCompvsSNR.svg')
plt.figure(5)
barwidth=0.9/Nalgs * np.mean(np.diff(10*np.log10(SNRs)))
for ialg in range(0,Nalgs):
    offset=(ialg-(Nalgs-1)/2)*barwidth
    plt.bar(10*np.log10(SNRs)+offset,np.mean(Npaths[:,:,ialg],axis=0),width=barwidth,label=algLegendList[ialg])
plt.xlabel('SNR(dB)')
plt.ylabel('N paths')
plt.legend()
plt.savefig(outfoldername+'/NpathvsSNR.svg')
plt.show()
