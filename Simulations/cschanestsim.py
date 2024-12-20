#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mplin
import matplotlib.cm as cm
plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "Helvetica"
})
import time
import os
from tqdm import tqdm
import argparse
import pickle

import sys
sys.path.append('../')
from CASTRO5G import compressedSensingTools as cs
from CASTRO5G import multipathChannel as mc
from CASTRO5G import threeGPPMultipathGenerator as mpg
parser = argparse.ArgumentParser(description='MIMO OFDM CS Multipath Channel Estimation Simulator')

# parser.add_argument('--z3D', action='store_true', help='Activate 3d simulation mode')

parser.add_argument('--mpg',help='Generate multipath channels, (existing csv file must exist otherwise)', action='store_true')
parser.add_argument('-N', type=int,help='No. simulated channels')
parser.add_argument('-G', type=str,help='Type of generator. "3gpp" or "Uni:N" for N uniform IID paths')

parser.add_argument('--est',help='Perform channel estimation, (existing npz and csv files must exist otherwise)', action='store_true')
parser.add_argument('-F', type=str,help='comma-separated list of frames dimension cases Nframe:K:Ncp,')
#TODO
# parser.add_argument('-P', type=str,help='comma-separated list of pilot schemes')
parser.add_argument('-A', type=str,help='comma-separated list of algorithms')
parser.add_argument('-S', type=str,help='SNR values in dB min:max:step')

#parameters that affect plots

#parameters that affect workflow
parser.add_argument('--label', type=str,help='str label appended to result storage folder and figures')
parser.add_argument('--nosave', help='Do not save simulation data to new results file', action='store_true')
parser.add_argument('--show', help='Open plot figures during execution', action='store_true')
parser.add_argument('--print', help='Save plot files in svg to results folder', action='store_true')

# args = parser.parse_args("-N 4 -G Uni:5 -F=3:64:32:3:8:8:1 -A dicMult:2:2:2:1,dicMult:4:4:4:1,dicFast:2:2:2:1,dicFast:4:4:4:1 --label test --show --print".split(' '))

####
# Basic uni tests
# args = parser.parse_args("-N 100 -G Uni:10 -F=3:32:16:2:4:4:1,3:64:16:2:4:4:1 --label compareBaseDic --show --print".split(' '))
# args = parser.parse_args("-N 2 -G Uni:5 -F=1:2048:256:4:16:16:1 --label compareResolution --show --print".split(' '))

####
# Simulation uni results

args = parser.parse_args("-N 4 -G Uni:5 -F=5:64:32:2:8:8:1 --label SmallDicSize --show --print".split(' '))
# args = parser.parse_args("-N 100 -G Uni:5 -F=5:64:32:2:8:8:1 --label SmallDicSizeX100 --show --print".split(' '))
# args = parser.parse_args("-N 4 -G Uni:5 -F=5:64:32:2:8:8:1 --label SmallDicSizePFP --show --print".split(' '))

# args = parser.parse_args("--mpg -N 4 -G Uni:5 --est -F=1:512:16:1:4:4:1,1:1024:32:2:8:8:1,1:2048:64:4:16:16:1 --label BigDicSize --show --print".split(' '))
# args = parser.parse_args("-N 100 -G Uni:5 -F=1:512:16:1:4:4:1,1:1024:32:2:8:8:1,1:2048:64:4:16:16:1 --label BigDicSizeX100 --show --print".split(' '))
# args = parser.parse_args("-N 4 -G Uni:5 -F=31:1024:64:1:16:16:1 --label BigDicSizePFP --show --print".split(' '))

####
# Basic 3gpp tests
# there are TOO MANY PATHS in 3gpp channel. this config does not have enough observations for good CS
# args = parser.parse_args("--mpg -N 10 -G 3gpp --est -F=3:64:32:2:8:8:1 --label test3GPPinsuficient --show --print".split(' '))

# this config does not work due to insufficient number of combs and symbols
# args = parser.parse_args("-N 10 -G 3gpp -F=2:32:16:1:4:4:1 --label test3GPPtiny --show --print".split(' '))
# this config works but delay is very coarse
# args = parser.parse_args("-N 10 -G 3gpp -F=2:64:4:1:4:4:1 --label test3GPPtiny --show --print".split(' '))#10it/s
# this config is a bit slow but is the minimal working one with large FIR channel
# args = parser.parse_args("-N 100 -G 3gpp -F=2:256:16:1:4:4:1 --label test3GPPsmall --show --print".split(' '))#1.4s/it
# args = parser.parse_args("-N 10 -G 3gpp -F=1:512:32:2:8:8:1 --label test3GPPsmallplus --show --print".split(' '))#21s iter
# args = parser.parse_args("-N 10 -G 3gpp -F=1:1024:64:2:8:8:1 --label test3GPPmedium --show --print".split(' '))#130s iter
# args = parser.parse_args("-N 10 -G 3gpp -F=1:1024:64:4:16:16:1 --label test3GPPmedium16 --show --print".split(' '))#750s iter
# args = parser.parse_args("-N 10 -G 3gpp -F=1:2048:128:1:4:4:1 --label test3GPPbig4 --show --print".split(' '))#66s iter
# args = parser.parse_args("-N 2 -G 3gpp -F=1:2048:128:2:8:8:1 --label test3GPPbig8 --show --print".split(' '))#?350s iter
# args = parser.parse_args("--mpg -N 2 -G 3gpp --est -F=1:2048:128:4:16:16:1 -A=dicFast:2:2:2:1,dicFast:4:4:4:1 --label test3GPPbig --show --print".split(' '))#?s iter
# args = parser.parse_args("-N 10 -G 3gpp -F=1:4096:256:1:2:2:1 --label test3GPPhuge2 --show --print".split(' '))# 50s iter
# args = parser.parse_args("-N 10 -G 3gpp -F=1:4096:256:1:4:4:1 --label test3GPPhuge4 --show --print".split(' '))# 130s iter
# args = parser.parse_args("-N 2 -G 3gpp -F=1:4096:256:2:8:8:1 --label test3GPPhuge8 --show --print".split(' '))#1500s iter
# args = parser.parse_args("--mpg -N 2 -G 3gpp --est -F=1:4096:256:4:16:16:1 -A=dicFast:4:4:4:1 -S=-10:0:10 --label test3GPPhuge --show --print".split(' '))#950?s iter
# args = parser.parse_args("-N 10 -G 3gpp -F=3:512:16:1:4:4:1,2:1024:64:1:4:4:1,1:2048:128:1:4:4:1,1:4096:256:1:4:4:1 --label test3GPPframe --show --print".split(' '))
# args = parser.parse_args("-N 10 -G 3gpp -F=1:4096:256:1:4:4:1,1:4096:256:2:8:8:1,1:4096:256:4:16:16:1 --label test3GPPant --show --print".split(' '))
####
# Simulation 3gpp results
# args = parser.parse_args("-N 10 -G 3gpp -F=1:1024:64:2:8:8:1 --label test3GPPalg --show --print".split(' '))

plt.close('all')

#TODO: implement the above command line API and generate plots for publication

Nchan=args.N if args.N else 10
# bMode3D = args.z3D
if args.F:
    #TODO support multiple frame shapes in a single run
    frameDims = np.array([ [int(x) for x in line.split(':')] for line in args.F.split(',')],dtype=int)
else:
    frameDims = np.array([[3,32,16,2,4,4,1]])
NframeDims = len(frameDims)

# Ds=390e-9 #Ts=2.5ns with Rs=400MHz in NYUwim
# Tcp=520e-9 #mu=3
# Tcp=285e-9 #mu=4
Tcp=4166e-9#

if args.S:
    minSNRdB,maxSNRdB,stepSNRdB = [float(x) for x in args.S.split(':')]
else:
    minSNRdB,maxSNRdB,stepSNRdB = (-10,20,10)
SNRs=10**(np.arange(minSNRdB,maxSNRdB+0.01,stepSNRdB)/10)
Nsnr=len(SNRs)

# multipath generator
if args.G:
    mpgenInfo = args.G.split(':')
else:        
    mpgenInfo = ['Uni','20']
mpgen=mpgenInfo[0]

if args.label:
    outfoldername=f'../Results/CSChanEstresults{args.label}'
else:
    outfoldername=f'../Results/CSChanEstresults-{time.ctime()}'
if not os.path.isdir(outfoldername) and not args.nosave:
    os.mkdir(outfoldername)

# def stopModifierPfpOMP(Pfp,N):
    
#     return(-np.log(1-(1-Pfp)**(1/N)))

PFP = .25

def toppowAWGN(f):
    # Calculates the average sum power of the (1-f)N smallest coefficients of
    # a CAWGN vector of N coefficients. Result normalized by 1/N*σ².
    return (1+(1-f)*(1-np.log(1-f)))

def stopModifierPfpOMP(Pfp):
    #when SNR->0 the 'f' largest noise coefficients are selected, this upper-
    # -bounds the Pfp1
    return toppowAWGN(Pfp)

if args.A:
    confAlgsStr = [l.split(':') for l in args.A.split(',')]
    confAlgs = [(l[0],tuple(map(lambda x:float(x),l[1:]))) for l in confAlgsStr]
else:
    confAlgs=[# dicName, oversampling = (Xt Xd Xa Xmu)
        ("dicBase",(1.0,1.0,1.0,1.0)),
        # ("dicBase",(2.0,2.0,2.0,1.0)),
        ("dicBase",(4.0,4.0,4.0,1.0)),
        # ("dicBase",(8.0,8.0,8.0,1.0)),
        # ("dicBase",(1.0,1.0,1.0,10.0)),
        ("dicFFT",(1.0,1.0,1.0,1.0)),
        # ("dicFFT",(2.0,2.0,2.0,1.0)),
        ("dicFFT",(4.0,4.0,4.0,1.0)),
        # ("dicFFT",(8.0,8.0,8.0,1.0)),
        # ("dicFFT",(1.0,1.0,1.0,10.0)),stopModifierPfpOMP
        ("dicMult",(1.0,1.0,1.0,1.0)),
        # ("dicMult",(2.0,2.0,2.0,1.0)),
        ("dicMult",(4.0,4.0,4.0,1.0)),
        ("dicMult",(8.0,8.0,8.0,1.0)),
        # ("dicMult",(1.0,1.0,1.0,10.0)),
        ("dicFast",(1.0,1.0,1.0,1.0)),
        # ("dicFast",(2.0,2.0,2.0,1.0)),
        ("dicFast",(4.0,4.0,4.0,1.0)),
        ("dicFast",(8.0,8.0,8.0,1.0)),
        # ("dicFast",(1.0,1.0,1.0,10.0)),
        ]

# confAlgs=[#Xt Xd Xa Xmu accel legend string name
#     # (1.0,1.0,1.0,1.0,"dicBase",'Full Dic. X=1','-','o','r'),
#     # # (2.0,2.0,2.0,1.0,"dicBase",'OMPx2','-','s','r'),
#     # (4.0,4.0,4.0,1.0,"dicBase",'Full Dic. X=4','-','D','r'),
#     # # (1.0,1.0,1.0,100.0,"dicBase",'OMPBR','-','v','r'),
#     # (1.0,1.0,1.0,1.0,"dicFFT",'NB+FFT X=1','-.','*','k'),
#     # # (2.0,2.0,2.0,1.0,"dicFFT",'OMPx2a','-.','x','k'),
#     # (4.0,4.0,4.0,1.0,"dicFFT",'NB+FFT X=4','-.','+','k'),
#     # # (8.0,8.0,8.0,1.0,"dicFFT",'NB+FFT X=8','-.','1','k'),
#     # # (1.0,1.0,1.0,10.0,"dicFFT",'OMPBRa','-.','1','k'),
#     # (1.0,1.0,1.0,1.0,"dicMult",'MultiDic. X=1',':','o','b'),
#     (2.0,2.0,2.0,1.0,"dicMult",'MultiDic. X=2',':','s','b'),
#     (4.0,4.0,4.0,1.0,"dicMult",'MultiDic. X=4',':','D','b'),
#     # (8.0,8.0,8.0,1.0,"dicMult",'MultiDic. X=8','-.','v','b'),
#     # # (1.0,1.0,1.0,10.0,"dicMult",'OMPBRm',':','^','b'),
#     # (1.0,1.0,1.0,1.0,"dicFast",'3D-FFT X=1','--','*','g'),
#     (2.0,2.0,2.0,1.0,"dicFast",'3D-FFT X=2','--','x','g'),
#     (4.0,4.0,4.0,1.0,"dicFast",'3D-FFT X=4','--','+','g'),
#     # (8.0,8.0,8.0,1.0,"dicFast",'3D-FFT X=8','--','1','g'),
#     # (2.0,2.0,2.0,1.0,"dicSphr",'3D-FFT sphere X=2','--','x','g'),
#     ]

Nalg=len(confAlgs)

omprunner = cs.CSDictionaryRunner()
csDictionaries={
    "dicBase" :cs.CSCachedDictionary(),
    "dicMult" :cs.CSMultiDictionary(),
    "dicFFT"  :cs.CSBasicFFTDictionary(),
    "dicFast" :cs.CSMultiFFTDictionary(),
    "dicSphr" :cs.CSSphereFFTDictionary()
    }
pilgen = mc.MIMOPilotChannel("IDUV")
channelResponseFunctions = {
    # "TDoA" : mc.pSinc,
    "TDoA" : lambda t,M: np.fft.ifft(mc.pCExp(t,M)),  
    "AoA" : mc.fULA,
    "AoD" : mc.fULA,
    }
mpch = mc.MultipathDEC((0,0,10),(40,0,1.5),customResponse=channelResponseFunctions)
#-------------------------------------------------------------------------------
if args.mpg:        
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
else:
    # allUserData=pd.read_csv(outfoldername+'/userGenData.csv',index_col=['ue']) 
    allPathsData=pd.read_csv(outfoldername+'/chanGenData.csv',index_col=['ue','n']) 
#-------------------------------------------------------------------------------

if args.est:    
    MSE=np.zeros((NframeDims,Nalg,Nsnr,Nchan))
    Npaths=np.zeros((NframeDims,Nalg,Nsnr,Nchan))
    prepYTime=np.zeros((NframeDims,Nalg,Nchan))
    prepHTime=np.zeros((NframeDims,Nalg))
    sizeYDic=np.zeros((NframeDims,Nalg))
    sizeHDic=np.zeros((NframeDims,Nalg))
    runTimes=np.zeros((NframeDims,Nalg,Nsnr,Nchan))

    for ifdim in range(NframeDims):
        Nframe,K,Ncp,Nrfr,Na,Nd,Nrft = frameDims[ifdim,:]
        Ts=Tcp/Ncp
    
        hk_all = np.zeros((Nchan,K,Na,Nd),dtype=complex)
        wp_all = np.zeros((Nchan,Nframe,K,Nrfr,Na),dtype=complex)
        vp_all = np.zeros((Nchan,Nframe,K,Nd,Nrft),dtype=complex)
        zp_bb_all = np.zeros((Nchan,Nframe,K,Nrfr,1),dtype=complex)
        yp_noiseless_all = np.zeros((Nchan,Nframe,K,Nrfr,1),dtype=complex)
        for ichan in  tqdm(range(Nchan),desc=f"DEC with shape {frameDims[ifdim,:]}: "):
            mpch.insertPathsFromDF(allPathsData.loc[ichan,:])
            ht=mpch.getDEC(Na,Nd,Ncp,Ts)*np.sqrt(Nd*Na)#mpch uses normalized matrices of gain 1
            hk_all[ichan,:,:,:]=np.fft.fft(ht,K,axis=0)
            wp,vp=pilgen.generatePilots(Nframe*K*Nrft,Na,Nd,Npr=Nframe*K*Nrfr,rShape=(Nframe,K,Nrfr,Na),tShape=(Nframe,K,Nd,Nrft))
            wp_all[ichan,:,:,:,:] = wp
            vp_all[ichan,:,:,:,:] = vp
            zp=mc.AWGN((Nframe,K,Na,1))
            zp_bb_all[ichan,:,:,:,:]=np.matmul(wp,zp)
            yp_noiseless_all[ichan,:,:,:,:]=pilgen.applyPilotChannel( hk_all[ichan,:,:,:] ,wp,vp,None)
            
        #pregenerate the H dics (pilot independent)
        for ialg in range(Nalg):           
        
    #-------------------------------------------------------------------------------
            t0 = time.time()
            dicName,superresolution = confAlgs[ialg]
            Xt,Xa,Xd,Xmu=superresolution
            dicObj=csDictionaries[dicName]
            Lt,La,Ld=(int(Ncp*Xt),int(Na*Xa),int(Nd*Xd))
            
            # print(f'Paramos OMP con coeficiente {stopModifierPfpOMP(PFP,Lt*La*Ld)} x sigma')
            OMPstopmodifier = stopModifierPfpOMP(1e-1)
            # OMPstopmodifier = 1
            Nobserv=(Nrfr*K*Nframe)
            Nsearch=Lt*La*Ld
            print(f"Pregen CS dict alg={confAlgs[ialg]} shape={frameDims[ifdim]} th. max paths {int(np.floor(Nobserv/np.log2(Nsearch)))}")
            dicObj.setHDic((K,Ncp,Na,Nd),(Lt,La,Ld))# duplicates handled by cache
            if isinstance(dicObj.currHDic.mPhiH,np.ndarray):
                sizeHDic[ifdim,ialg] = dicObj.currHDic.mPhiH.size
            elif isinstance(dicObj.currHDic.mPhiH,tuple):
                sizeHDic[ifdim,ialg] = np.sum([x.size for x in dicObj.currHDic.mPhiH])
            else:
                sizeHDic[ifdim,ialg] = 0
            prepHTime[ifdim,ialg] = time.time()-t0         
            print(f"Finished in {prepHTime[ifdim,ialg]:3f} seconds")
            
            for ichan in  tqdm(range(Nchan),desc=f"CS Sim {confAlgs[ialg]} - {frameDims[ifdim,:]} =>", position=0):
                #load the DEC values once for all SNRS
                hk = hk_all[ichan,:,:,:]
                vp = vp_all[ichan,:,:,:,:]
                wp = wp_all[ichan,:,:,:,:]
                zp_bb = zp_bb_all[ichan,:,:,:]
                yp_noiseless = yp_noiseless_all[ichan,:,:,:,:]     
                
                #preconfigure the pilot dictionary once for all SNRs
                t0 = time.time()
                dicObj.setYDic(ichan,(wp,vp))
                if isinstance(dicObj.currYDic.mPhiY,np.ndarray):
                    sizeYDic[ifdim,ialg] = dicObj.currYDic.mPhiY.size
                elif isinstance(dicObj.currYDic.mPhiY,tuple):
                    sizeYDic[ifdim,ialg] = np.sum([x.size for x in dicObj.currYDic.mPhiY])
                else:
                    sizeYDic[ifdim,ialg] = 0    
                prepYTime[ifdim,ialg,ichan] = time.time()-t0                     
                
                #run the CS DEC simulations
                # for isnr in tqdm(range(0,Nsnr),desc="SNR", position=1,leave=False):
                for isnr in range(0,Nsnr):
                    sigma2=1.0/SNRs[isnr]
                    yp=yp_noiseless+zp_bb*np.sqrt(sigma2)
                    t0 = time.time()
                    omprunner.setDictionary(dicObj)
                    hest,paths,_,_=omprunner.OMP(yp,OMPstopmodifier*sigma2*K*Nframe*Nrfr,ichan,vp,wp, Xt,Xa,Xd,Xmu,Ncp)
                    MSE[ifdim,ialg,isnr,ichan] = np.mean(np.abs(hk-hest)**2)/np.mean(np.abs(hk)**2)
                    runTimes[ifdim,ialg,isnr,ichan] = time.time()-t0
                    Npaths[ifdim,ialg,isnr,ichan] = len(paths.TDoA)
                    
                #for large Nsims the pilot cache grows too much so we free the memory when not needed
                dicObj.freeCacheOfPilot(ichan,(K,Ncp,Na,Nd),(Lt,La,Ld))
            #
            dicObj.freeCacheOfHDic((K,Ncp,Na,Nd),(Lt,La,Ld))
            #remove the current pointers too, as memory tends to stay reserved for them
            dicObj.currYDic=None
            dicObj.currHDic=None
    if not args.nosave:
        np.savez_compressed(outfoldername+'/chanEstResults.npz',
                    MSE=MSE,
                    Npaths=Npaths,
                    prepYTime=prepYTime,
                    prepHTime=prepHTime,
                    sizeYDic=sizeYDic,
                    sizeHDic=sizeHDic,
                    runTimes=runTimes)
        confAlgsStr = np.array([[x[0]]+ [str(y) for y in x[1]] for x in confAlgs])
        np.savetxt(outfoldername+'/confAlgs.csv',confAlgsStr,delimiter=',',fmt='%s')
        np.savetxt(outfoldername+'/frameDims.csv',frameDims,delimiter=',',fmt='%d')
else:
    data=np.load(outfoldername+'/chanEstResults.npz')    
    MSE=data["MSE"]
    Npaths=data["Npaths"]
    prepYTime=data["prepYTime"]
    prepHTime=data["prepHTime"]
    sizeYDic=data["sizeYDic"]
    sizeHDic=data["sizeHDic"]
    runTimes=data["runTimes"]
    if 'confAlgs' in data:#convert old format to new
        confAlgsStr = data["confAlgs"][:,[4,0,1,2,3]]
        np.savetxt(outfoldername+'/confAlgs.csv',confAlgsStr,delimiter=',',fmt='%s')
        np.savetxt(outfoldername+'/frameDims.csv',frameDims,delimiter=',',fmt='%d')
    confAlgsStr=np.loadtxt(outfoldername+'/confAlgs.csv',delimiter=',',dtype='U32')
    if confAlgsStr.ndim==1:
        confAlgsStr=confAlgsStr.reshape(1,-1)
    confAlgs = [ (x[0], tuple([float(y) for y in x[1:]])) for x in confAlgsStr]
    Nalg=len(confAlgs)
    frameDims=np.loadtxt(outfoldername+'/frameDims.csv',delimiter=',',dtype=int)
    if frameDims.ndim==1:
        frameDims=frameDims.reshape(1,-1)
    NframeDims = len(frameDims)
            
listOfLineTypes = ['-',':','-.','--']
listOfPatterns = [ None, "x" , "-"  , "/", "\\" , "|", "+" , "x", "o", "O", ".", "*" ]
listOfMarkersComplementary= [#markers are visually related vertically for example star vs pengagon, x vs square, + vs diamond etc
    ['o','s','d','v','^','<','>'],
    ['*','x','+','1','2','3','4'],
    ['p','|','D', 4 , 5 , 6 , 7 ]
]
bytesPerFloat = np.array([0],dtype=np.complex128).itemsize

uniqueRunners,order,oinv,count = np.unique([x[0] for x in confAlgs],return_index=True,return_inverse=True,return_counts=True)
uniqueRunners=[ confAlgs[x][0] for x in np.sort(order)]
count=count[np.argsort(order)]
Nrunners = len(uniqueRunners)
runnerIndices = [ int(np.where(np.array(uniqueRunners)==x[0])[0][0]) for x in confAlgs ]
runnerCount = np.zeros_like(runnerIndices)
for r in range(Nrunners):
    mask =  np.array([uniqueRunners[r]==x[0] for x in confAlgs])
    runnerCount[mask] = np.arange(0,count[r])
Nsize=np.max(count)
frameDimsStr = [':'.join(map(str,x)) for x in frameDims]

algLbl={
        "dicBase":"Full Dic.",
        "dicFFT":"NB+FFT",
        "dicMult":"MultiDic.",
        "dicFast":"3D-FFT",
        "dicSphr":"SphereD",
    }

if Nalg>1:  
    labelList=[]
    for a in range(Nalg):
        k,ov=confAlgs[a]
        al=algLbl[k]
        sz=('BR' if ov[-1]>1 else f'X={int(np.mean(ov[0:-1]))}')
        if Nrunners>1:            
            if count[runnerIndices[a]]>1:
                labelList.append(f'{al} {sz}')
            else:
                labelList.append(al)
        elif count[0]>1:
            labelList.append(sz)
    
if NframeDims>1:    
    if Nalg>1:
        labelList = [x+'-'+y for y in frameDimsStr for x in labelList ]
    else:
        labelList = frameDims

if NframeDims==1 and Nalg==1:
    labelList=[confAlgs[0][0]]
ldecor=[]
for ifdim in range(NframeDims):
    for ialg in range(Nalg):        
        lidx = ifdim*Nalg+ialg
        if NframeDims*Nalg>1:
            clr=cm.turbo(lidx/(NframeDims*Nalg-1))
        else:
            clr=cm.turbo(.5)
        lin=listOfLineTypes[ifdim % len(listOfLineTypes)]
        if Nrunners<=3:
            mrk=listOfMarkersComplementary[runnerIndices[ialg]][1+runnerCount[ialg]]
        else:
            mrk=listOfMarkersComplementary[runnerCount[ialg]][1+runnerIndices[ialg]]
        pat=listOfPatterns[runnerIndices[ialg]]
        ldecor.append((clr,lin,mrk,pat))
            
fig_ctr=0

fig_ctr+=1
plt.figure(fig_ctr)
plt.yscale("log")
barwidth= 0.9/(2*NframeDims)
for ifdim in range(NframeDims):
    offset=(-1/2)*barwidth+ifdim*.9/2
    lbmod = ' '+frameDimsStr[ifdim] if NframeDims>1 else ''
    plt.bar(np.arange(Nalg)+offset,bytesPerFloat*sizeHDic[ifdim,:]*(2.0**-20),width=barwidth,label='$\\Psi$ dict'+lbmod)
    offset=(+1/2)*barwidth+ifdim*.9/2
    plt.bar(np.arange(Nalg)+offset,bytesPerFloat*sizeYDic[ifdim,:]*(2.0**-20),width=barwidth,label='$\\Upsilon$ dict'+lbmod)
plt.xticks(ticks=np.arange(0,Nalg),labels=[x[0] for x in confAlgs],rotation=-15)
# plt.xlabel('Algoritm')
plt.ylabel('Dictionary size MByte')
plt.legend()
plt.savefig(outfoldername+'/DicMBvsAlg.svg')

fig_ctr+=1
plt.figure(fig_ctr)
plt.yscale("log")
barwidth= 0.9/(2*NframeDims)
for ifdim in range(NframeDims):
    offset=(-1/2)*barwidth+ifdim*.9
    lbmod = ' '+frameDimsStr[ifdim] if NframeDims>1 else ''
    plt.bar(np.arange(Nalg)+offset,prepHTime[ifdim,:],width=barwidth,label='$\\Psi$ dict'+lbmod)
    offset=(+1/2)*barwidth+ifdim*.9
    plt.bar(np.arange(Nalg)+offset,np.mean(prepYTime[ifdim,:,:],axis=1),width=barwidth,label='$\\Upsilon$ dict'+lbmod)
plt.xticks(ticks=np.arange(0,Nalg,1),labels=[x[0] for x in confAlgs],rotation=-15)
# plt.xlabel('Algoritm')1
plt.ylabel('precomputation time')
plt.legend()
plt.savefig(outfoldername+'/DicCompvsAlg.svg')

fig_ctr+=1
plt.figure(fig_ctr)
for ifdim in range(NframeDims):
    for ialg in range(Nalg):
        lidx = ifdim*Nalg+ialg
        clr,lin,mrk,pat=ldecor[lidx]
        plt.semilogy(10*np.log10(SNRs),np.mean(MSE[ifdim,ialg,:,:],axis=1),color=clr,marker=mrk,linestyle=lin,label=labelList[lidx])
        # Nframe,K,Ncp,Nrfr,Na,Nd,Nrft = [int(x) for x in frameDims[ifdim].split(':')]
        # plt.semilogy(10*np.log10(SNRs),np.mean(Npaths[ifdim,ialg,:,:],axis=1)/(Nframe*K*Nrfr)/SNRs,color=clr,marker='s',linestyle=':',label='LB'+labelList[lidx])
plt.legend()
plt.xlabel('SNR(dB)')
plt.ylabel('MSE')
plt.savefig(outfoldername+'/MSEvsSNR.svg')

fig_ctr+=1
plt.figure(fig_ctr)
plt.yscale("log")
barwidth= 0.9/(Nalg*NframeDims)  * (np.mean(np.diff(10*np.log10(SNRs))) if len(SNRs)>1 else 1)


for ifdim in range(NframeDims):
    for ialg in range(Nalg):
        lidx = ifdim*Nalg+ialg
        offset=(lidx-(Nalg*NframeDims-1)/2)*barwidth
        clr,lin,mrk,pat=ldecor[lidx]
        plt.bar(10*np.log10(SNRs)+offset,np.mean(runTimes[ifdim,ialg,:,:],axis=1),width=barwidth,color=clr,hatch=pat,label=labelList[lidx])
plt.xlabel('SNR(dB)')
plt.ylabel('runtime')
plt.legend(labelList)
plt.savefig(outfoldername+'/CSCompvsSNR.svg')

fig_ctr+=1
plt.figure(fig_ctr)
barwidth=0.9/(Nalg*NframeDims) * np.mean(np.diff(10*np.log10(SNRs)))
for ifdim in range(NframeDims):
    for ialg in range(0,Nalg):
        lidx = ifdim*Nalg+ialg
        offset=(lidx-(Nalg*NframeDims-1)/2)*barwidth
        clr,lin,mrk,pat=ldecor[lidx]
        plt.bar(10*np.log10(SNRs)+offset,np.mean(Npaths[ifdim,ialg,:,:],axis=1),width=barwidth,color=clr,hatch=pat,label=labelList[lidx])
plt.xlabel('SNR(dB)')
plt.ylabel('N paths')
plt.legend()
plt.savefig(outfoldername+'/NpathvsSNR.svg')

fig_ctr+=1
plt.figure(fig_ctr)
plt.yscale("log")
barwidth= 0.9/(Nalg*NframeDims)  * (np.mean(np.diff(10*np.log10(SNRs))) if len(SNRs)>1 else 1)

for ifdim in range(NframeDims):
    for ialg in range(Nalg):
        lidx = ifdim*Nalg+ialg
        offset=(lidx-(Nalg*NframeDims-1)/2)*barwidth
        clr,lin,mrk,pat=ldecor[lidx]
        plt.bar(10*np.log10(SNRs)+offset,np.mean(runTimes[ifdim,ialg,:,:]/Npaths[ifdim,ialg,:,:],axis=1),width=barwidth,color=clr,hatch=pat,label=labelList[lidx])
plt.xlabel('SNR(dB)')
plt.ylabel('per iteration runtime')
plt.legend(labelList)
plt.savefig(outfoldername+'/CSItervsSNR.svg')
