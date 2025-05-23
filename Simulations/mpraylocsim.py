#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time
import os
import argparse
from tqdm import tqdm

import sys
sys.path.append('../')
from CASTRO5G import MultipathLocationEstimator
from CASTRO5G import threeGPPMultipathGenerator as mpg
#TODO: make the script below the __main__() of a class that can be imported by other python programs
parser = argparse.ArgumentParser(description='Multipath Location Estimation Simulator')
#parameters that affect number of simulations
parser.add_argument('-N', type=int,help='No. simulated channels')
parser.add_argument('--z3D', action='store_true', help='Activate 3d simulation mode')
parser.add_argument('-M',type=str,help='Comma separated min,max Map dimensions')

#parameters that affect error
parser.add_argument('-E', type=str,help='comma-separated list of angle error models. N for no error, S:n:min:max for n gaussian error points with stds min-max in dB, D:m:n:k for n-quantization error models')

#parameters that affect multipath generator
parser.add_argument('-G', type=str,help='Type of generator. "3GPP" or "Geo:N" for N scatterers contained in Map')

#parameters that affect location algorithms
parser.add_argument('-A', type=str,help='comma-separated list of algorithms')

#parameters that affect plots
parser.add_argument('--cdf', type=str,help='plot CDF for a given comma-separated list of errors')
parser.add_argument('--pcl', type=str,help='comma-separated list of error axis-percentile plots (example "D:80")')
parser.add_argument('--map', type=str,help='plot map')
parser.add_argument('--mapcap', type=str,help='limit number of points in map')
parser.add_argument('--vso', type=str,help='scatter plot of location error vs orientation')
parser.add_argument('--rtm', type=str,help='bar plot of algorithm run times')

#parameters that affect workflow
parser.add_argument('--label', type=str,help='str label appended to storage files')
parser.add_argument('--nosave', help='Do not save simulation data to new results file', action='store_true')
parser.add_argument('--nompg',help='Do not perform multipath generation, load existing file', action='store_true')
parser.add_argument('--noloc',help='Do not perform location estimation, load existing file', action='store_true')
parser.add_argument('--show', help='Open plot figures during execution', action='store_true')
parser.add_argument('--print', help='Save plot files in svg to results folder', action='store_true')

#refine to make it consistent before reestructuring all this code

##############copy paste from here
#-G Geo:20 3GPP

#-E=
#NO,
#S:7:0:20,
#D:16x16x16:64x64x64:256x256x256:1024x1024x1024:4096x4096x4096
#16xinfxinf:64xinfxinf:256xinfxinf:1024xinfxinf:4096xinfxinf
#infxinfx16:infxinfx64:infxinfx256:infxinfx1024:infxinfx4096

#2D simulation with Geometric ray tracing simple channel model
# args = parser.parse_args("--noloc --nompg -N 100 -G Geo:20 -E=NO,D:32:64:128:256:512:1024 --label GEO202D --show --print --cdf=no,Dx256 --pcl=D:75 --map=no,Dx256 --vso=no,Dx256 --rtm=no,Dx256".split(' '))
#2D simulation with 3GPP channels with first reflection fitted multipath
# args = parser.parse_args("--noloc --nompg -N 100 -G 3GPP -E=NO,D:32:64:128:256:512:1024 --label 3GPP2D --show --print --cdf=no,Dx256 --pcl=D:75 --map=no,Dx256 --vso=no,Dx256 --rtm=no,Dx256".split(' '))
#2D simulation with Geometric ray tracing simple channel model
# args = parser.parse_args("--noloc --nompg --z3D -N 100 -G Geo:20 -E=NO,D:32:64:128:256:512:1024 --label GEO203D --show --print --cdf=no,Dx256 --pcl=D:75 --map=no,Dx256 --vso=no,Dx256 --rtm=no,Dx256".split(' '))
#2D simulation with Geometric ray tracing simple channel model
# args = parser.parse_args("--z3D -N 100 -G 3GPP -E=NO,D:32:64:128:256:512:1024 --label 3GPP3D --show --print --cdf=no,Dx256 --pcl=D:75 --map=no,Dx256 --vso=no,Dx256 --rtm=no,Dx256".split(' '))

args = parser.parse_args("--noloc --nompg --z3D -N 1000 -G 3GPP:50:UMi:28.0,3GPP:50:InF-DH:28.0,3GPP:50:UMa:3.5,3GPP:50:RMa:.7,Geo:20 -E=NO,D:64:128:256:512:1024 --label 3DlocGenComparison --show --print --cdf=no,Dx512 --pcl=D:75 --map=no,Dx512 --mapcap 100 --vso=no,Dx512 --rtm=no,Dx512".split(' '))
# args = parser.parse_args("--z3D -N 10 -G 3GPP:50:UMi:28.0,3GPP:50:InF-DH:28.0,3GPP:50:UMa:3.5,3GPP:50:RMa:.7,Geo:20 -E=NO,D:64:128:256:512:1024 --label test --show --print --cdf=no,Dx512 --pcl=D:75 --map=no,Dx512 --mapcap 100 --vso=no,Dx512 --rtm=no,Dx512".split(' '))

# numero de simulacions
Nsims=args.N if args.N else 100
bMode3D = args.z3D
locColNames=['X0','Y0','Z0'] if bMode3D else ['X0','Y0']
rotColNames=['AoA0','ZoA0','SoA0'] if bMode3D else ['AoA0']
mapColNames=['Xs','Ys','Zs'] if bMode3D else ['Xs','Ys']
Ndim = 3 if bMode3D else 2
if args.M:
    mapDims = [float(x) for x in args.M.split(',')]
else:
    mapDims = [-100,100,-100,100] + ([-20,20] if bMode3D else[]) #note that this is w.r.t. BS height at zero
dmax=np.array(mapDims[1::2])
dmin=np.array(mapDims[0::2])
# multipath generator
if args.G:
    mpgenInfoStr = args.G.split(',')
    mpgenInfo = [tuple(x.split(':')) for x in mpgenInfoStr]
else:        
    mpgenInfo = [('Geo','20')]
Ngen =len(mpgenInfo)
#TODO: Aquí parece que hai bastantes parametros que completar

# error vector modeling
def parseErrStr(s):
    if 'NO' in s:
        return([('no',)])
    elif 'S' in s:
        _,NS,minStd,maxStd = s.split(':')
        lSTD=np.logspace(float(minStd)/10,float(maxStd)/10,int(NS))
        return( [('S',x) for x in lSTD] )
    elif 'D' in s:
        l=s.split(':')        
        lDicSizes=[tuple(['D']+y.split('x')) for y in l[1:] ]
        return( lDicSizes )
    else:
        print(f'unrecognized error type {s}')
    #TODO add ifs etc. to make this selectable with OMP
if args.E:
    lErrMod = sum([parseErrStr(x) for x in args.E.split(',')],[])
else:
    lErrMod = []
NerrMod=len(lErrMod)

# location algorythms - evolución de location estimator
# TODO: rethink the format of this table and cmd argument
def parseAlgStr(s):
    l=s.split(':')
    l[0]=bool(l[0])
    if l[1]=='inf':
        l[1]=np.inf
    else:
        l[1]=int(l[1])        
    return(tuple(l))

if args.A:
    lLocAlgs=[
            parseAlgStr(s)
            for s in args.A.split(',')
            ]
    #TODO define more elegant syntax for cases and better parser that converts bool properly
else:
    lLocAlgs=[#a opriori aoa0, quantized aoa0, grouping method, optimization method
        (True,np.inf,'','','LS',':','o','b'),
        # (True,64,'','','Linear + 64x gyroscope',':','*','b'),
       #  (False,np.inf,'3path','brute','Brute3P',':','s','r'),
       #  (False,np.inf,'3path','lm','Root3P','-.','s','g'),
       #  (False,64,'3path','lm','Root3Ph','-.','x','g'),
       #  (False,np.inf,'drop1','brute','BruteD1','-','s','r'),
          # (False,np.inf,'drop1','lm','RootD1','-','s','g'),
        # (False,64,'drop1','lm','Non-linear LS orientation','-','x','g'),
        # (False,64,'','margin','Iterative orientation','--','d','m'),
       ]
NlocAlg =len(lLocAlgs)

if args.label:
    outfoldername="../Results/MPRayLocresults%s"%(args.label)
else:
    outfoldername="../Results/MPRayLocresults-%d-%d-%d"%(NerrMod,Nsims,Ngen)
if not os.path.isdir(outfoldername):
    os.mkdir(outfoldername)

#TODO: create command line arguments for these parameters
c=3e8
Ts=1.0/400e6 #2.5ns
Ds=320e-9 #Ts*128 FIR filter
sigmaTauE=40e-9

loc=MultipathLocationEstimator.MultipathLocationEstimator(nPoint=100,orientationMethod='lm')    

# if this parameter is present we get the mpg data from a known file (better for teset&debug)
if args.nompg:    
    allUserData=pd.read_csv(outfoldername+'/userGenData.csv',index_col=['ue']) 
    allPathsData=pd.read_csv(outfoldername+'/chanGenData.csv',index_col=['genNo','ue','n']) 
else:
    allPathsDFlist = []
    t_start_gen=time.time()
    # TODO: this has to be modeled in a separate class
    dminu0=dmin
    if dminu0[2]<0:
        dminu0[2]=0#UEs must be above ground for 3GPP pathlos functions to not complain
    d0=np.random.rand(Nsims,Ndim)*(dmax-dminu0)+dminu0    
    toa0=np.linalg.norm(d0,axis=-1)/c
    tauE=np.random.rand(Nsims)*sigmaTauE    
    if bMode3D:
        aod0,zod0=loc.angVector(d0)
        rot0=np.random.rand(Nsims,3)*np.array([2,1,2])*np.pi #receiver angular measurement offset
        aoa0=rot0[:,0]
    else:    
        aod0=loc.angVector(d0)
        rot0=np.random.rand(Nsims)*2*np.pi #receiver angular measurement offset
        aoa0=rot0
    allUserData = pd.DataFrame(index=pd.Index(np.arange(Nsims),name="ue"),
                               data={
                                   "X0":d0[:,0],
                                   "Y0":d0[:,1],
                                   "AoD0":aod0,
                                   "AoA0":aoa0,
                                   "tauE":tauE
                                   })
    if bMode3D:
        allUserData['Z0'] = d0[:,2]
        allUserData['ZoD0'] = zod0
        allUserData['ZoA0'] = rot0[:,1]
        allUserData['SoA0'] = rot0[:,2]
    for ngen in range(Ngen):
        mpgen=mpgenInfo[ngen][0]     
        if mpgen == 'Geo':        
            if len(mpgenInfo[ngen])>1:
                Npath=int(mpgenInfo[ngen][1])
            else:
                Npath=20
            #generate locations and compute multipath 
            d=np.random.rand(Nsims,Npath,Ndim)*(dmax-dmin)+dmin    
            DoD=d/np.linalg.norm(d,axis=2,keepdims=True)
            DoA=(d-d0[:,None,:])/np.linalg.norm( d-d0[:,None,:] ,axis=2,keepdims=True)   
            #delays based on distance
            li=np.linalg.norm(d,axis=-1)+np.linalg.norm(d-d0[:,None,:],axis=-1)
            toa=li/c
            tdoa=toa-toa0[:,None]-tauE[:,None]        
            #angles from locations
            if bMode3D:
                aod,zod=loc.angVector(DoD)
                R0=np.array([ loc.rMatrix(*x) for x in rot0])
                DDoA=DoA@R0 #transpose of R0.T @ DoA.transpose([0,2,1])
                daoa,dzoa=loc.angVector(DDoA)
            else:
                aod=loc.angVector(DoD)
                R0=np.array([ loc.rMatrix(x) for x in aoa0])
                DDoA=DoA@R0 #transpose of R0.T @ DoA.transpose([0,2,1])
                daoa=loc.angVector(DDoA)
            genPathsData = pd.DataFrame(index=pd.MultiIndex.from_product([np.arange(Nsims),np.arange(Npath)],names=["ue","n"]),
                                        data={
                                            "AoD" : aod.reshape(-1),
                                            "DAoA" : daoa.reshape(-1),
                                            "TDoA" : tdoa.reshape(-1),
                                            "Xs": d[:,:,0].reshape(-1),
                                            "Ys": d[:,:,1].reshape(-1)
                                            })        
            if bMode3D:
                genPathsData['Zs'] = d[:,:,2].reshape(-1)
                genPathsData['ZoD'] = zod.reshape(-1)
                genPathsData['DZoA'] = dzoa.reshape(-1)
        elif mpgen == "3GPP":        
            # TODO: introducir param de entrada para regular blargeBW, scenario, etc
            # Tamen mais tarde - Elección de escenario vía param. de entrada
            
            if len(mpgenInfo[ngen])>1:
                Npath=int(mpgenInfo[ngen][1])
            else:
                Npath=50                
            if len(mpgenInfo[ngen])>2:
                sce=mpgenInfo[ngen][2]
            else:
                sce="UMi"          
            if len(mpgenInfo[ngen])>3:
                fcGHz=float(mpgenInfo[ngen][3])
            else:
                fcGHz=28.0
            if len(mpgenInfo[ngen])>4:
                LBex=(mpgenInfo[ngen][4]=="True")
            else:
                LBex=True
                
            model = mpg.ThreeGPPMultipathChannelModel(fc=fcGHz,scenario=sce,bLargeBandwidthOption=LBex)   
            # Npath = model.maxM*np.max( model.scenarioParams.loc['N'] )                
            lpathDFs = []
            losAll = np.zeros((Nsims),dtype=bool)
            PLall = np.zeros((Nsims,2))# deterministic PL, shadowing in dB
            for n in tqdm(range(Nsims),desc=f'Generating {Nsims} 3GPP {sce} {fcGHz:.2f}GHz multipath channels'):        
                txPos = (0,0,10)
                if bMode3D:
                    rxPos = (d0[n,0],d0[n,1],d0[n,2]+10)#location has tx at 0 but 3GPP considers 10m BS height and 1.5m pedestrian height, this adjust the z axis
                else:
                    rxPos = (d0[n,0],d0[n,1],1.5)# for 2D case height is used in pathloss calculations but not angle fitting
                plinfo,macro,clusters,subpaths = model.create_channel(txPos,rxPos)
                losAll[n]=plinfo[0]
                PLall[n,:]=plinfo[1:]
                # (txPos,rxPos,plinfo,clusters,subpaths)  = model.fullFitAoA(txPos,rxPos,plinfo,clusters,subpaths)
                (txPos,rxPos,plinfo,clusters,subpaths)  = model.randomFitEpctClusters(txPos,rxPos,plinfo,clusters,subpaths,Ec=.75,Es=.75,P=[0,.5,.5,0],mode3D= bMode3D)
                # txArrayAngle = 0#deg
                # rxArrayAngle = np.mod(np.pi+np.arctan2(y0[n],x0[n]),2*np.pi)*180/np.pi
                # (txPos,rxPos,plinfo,clusters,subpaths)  = model.fullDeleteBacklobes(txPos,rxPos,plinfo,clusters,subpaths,tAoD=txArrayAngle,rAoA=rxArrayAngle)
                #remove subpaths that have not been converted to first order
                subpathsProcessed = subpaths.loc[~np.isinf(subpaths.Xs)].copy() # must be a hard copy and not a mere "view" of the original DF
                # discardzodmask = (subpathsProcessed.ZoA>180) | (subpathsProcessed.ZoD>180) | (subpathsProcessed.ZoA<0) | (subpathsProcessed.ZoD<0) #bug fix for indoor office
                # subpathsProcessed = subpathsProcessed[~discardzodmask]
                #remove weak subpaths if there are more than 50 for faster computation
                if ( subpathsProcessed.shape[0] > Npath ):            
                    srtdP = subpathsProcessed.P.sort_values(ascending=False)
                    indexStrongest=srtdP.iloc[0:Npath].index
                    subpathsProcessed = subpathsProcessed.loc[indexStrongest]
                #     if nvalid[n]<4:
                #         model.dChansGenerated.pop(txPos+rxPos)
                # if not plinfo[0]:#NLOS
                #     x1stnlos = subpathsProcessed.sort_values("TDoA").iloc[0].Xs
                #     y1stnlos = subpathsProcessed.sort_values("TDoA").iloc[0].Ys
                #     lD = np.sqrt(x1stnlos**2+y1stnlos**2)
                #     lA = np.sqrt((x1stnlos-x0[n])**2+(y1stnlos-y0[n])**2)
                #     tau1stlos = (lD+lA)/c
                #     tauE[n] = tauE[n] - (tau1stlos-toa0[n])
                subpathsProcessed.TDoA -= tauE[n]
                subpathsProcessed.AoD = subpathsProcessed.AoD*np.pi/180
                if bMode3D:
                    subpathsProcessed.ZoD = subpathsProcessed.ZoD*np.pi/180
                    DoA=loc.uVector(subpathsProcessed.AoA*np.pi/180 , subpathsProcessed.ZoA*np.pi/180 ).T                
                    R0=loc.rMatrix(rot0[n,0],rot0[n,1],rot0[n,2])
                    DDoA=DoA@R0
                    subpathsProcessed.rename(inplace=True,columns={"AoA":"DAoA",
                                                                   "ZoA":"DZoA",})  
                    daoa,dzoa=loc.angVector(DDoA)     
                    subpathsProcessed.DAoA = daoa
                    subpathsProcessed.DZoA = dzoa
                    subpathsProcessed.Zs = subpathsProcessed.Zs -10#location has tx at 0 but 3GPP considers 10m BS height and 1.5m pedestrian height, this adjust the z axis
                else:
                    subpathsProcessed.AoA = np.mod( subpathsProcessed.AoA*np.pi/180 -rot0[n], 2*np.pi)                
                    subpathsProcessed.rename(inplace=True,columns={"AoA":"DAoA"})       
                    subpathsProcessed.drop(inplace=True,columns=["ZoA","ZoD"])
                subpathsProcessed.reset_index(inplace=True)#moves cluster-subpath pairs n,m to normal columns
                subpathsProcessed.rename(inplace=True, columns={"n":"Cluster",
                                                                "m":"Subpath",})            
                lpathDFs.append(subpathsProcessed)
            genPathsData=pd.concat(lpathDFs,keys=np.arange(Nsims),names=["ue",'n'])
            model.initCache()
            # allUserData["LOS"]=losAll
            # allUserData[["DeterministicPLdB","ShadowingdB"]]=PLall
        else:
            print("MultiPath generation method %s not recognized"%mpgen)
        allPathsDFlist.append(genPathsData)
    allPathsData=pd.concat(allPathsDFlist,keys=np.arange(Ngen),names=["genNo","ue",'n'])
    
    if not args.nosave: 
        allUserData.to_csv(outfoldername+'/userGenData.csv') 
        allPathsData.to_csv(outfoldername+'/chanGenData.csv') 
    print("Total Multipath Generation Time:%s seconds"%(time.time()-t_start_gen))
    

# nPathAll= allPathsData.groupby(level=['genNo','ue']).size().to_numpy()
# Nmaxpath=np.max(nPathAll)
t_start_err = time.time()
lErrorPathDFs = []
for nv in tqdm(range(NerrMod),desc='applying error models to paths'):
    if bMode3D:
        errPathDF = allPathsData[['AoD', 'DAoA', 'TDoA','ZoD', 'DZoA']].copy()
    else:
        errPathDF = allPathsData[['AoD', 'DAoA', 'TDoA']].copy()
    errType=lErrMod[nv][0]
    #if errType=='no': do nothing
    if errType=='S': 
        Ntot = errPathDF.shape[0]
        errStd=lErrMod[nv][1]        
        errPathDF.AoD  = np.mod(errPathDF.AoD +errStd*2*np.pi*np.random.randn(Ntot),2*np.pi)
        errPathDF.DAoA = np.mod(errPathDF.DAoA+errStd*2*np.pi*np.random.randn(Ntot),2*np.pi)
        errPathDF.TDoA += errStd*Ds*np.random.randn(Ntot)
    elif errType=='D':
        if len(lErrMod[nv])==2:
            c1,c2,c3=3*[lErrMod[nv][1]]
        else:
            c1,c2,c3=lErrMod[nv][1:4]
        if not c1=='inf':
            errPathDF.AoD  = np.mod(np.round( errPathDF.AoD  *int(c1)/2/np.pi)*2*np.pi/int(c1),2*np.pi)
        if not c2=='inf':
            errPathDF.DAoA = np.mod(np.round( errPathDF.DAoA *int(c2)/2/np.pi)*2*np.pi/int(c2),2*np.pi)
        if not c3=='inf':
            errPathDF.TDoA  =np.round(errPathDF.TDoA *int(c3)/Ds )*Ds/int(c3)        
        if bMode3D:
            if len(lErrMod[nv])==2:            
                c4,c5=2*[lErrMod[nv][1]]
            else:                
                c4,c5=lErrMod[nv][4:6]        
            if not c4=='inf':
                errPathDF.ZoD  = np.mod(np.round( errPathDF.ZoD  *int(c4)/2/np.pi)*2*np.pi/int(c4),2*np.pi)
            if not c5=='inf':
                errPathDF.DZoA = np.mod(np.round( errPathDF.DZoA *int(c5)/2/np.pi)*2*np.pi/int(c5),2*np.pi)
    #TODO Compressed Sensing
    else:
        print("Multipath estimation error model %s to be written"%errType)
    lErrorPathDFs.append(errPathDF)
errPathDF=pd.concat(lErrorPathDFs,keys=range(NerrMod),names=["errNo","genNo","ue",'n'])
# errPathDF.to_csv(outfoldername+'/pathErrData.csv')
    

print("Total Multipath Estimation Time:%s seconds"%(time.time()-t_start_err))

if args.noloc: 
    allLocEstData=pd.read_csv(outfoldername+'/locEstData.csv',index_col=['alg','errNo','genNo','ue']) 
    allMapEstData=pd.read_csv(outfoldername+'/mapEstData.csv',index_col=['alg','errNo','genNo','ue','n']) 
else:
    t_start_loc=time.time() 
    rot0_est=np.zeros((NlocAlg,NerrMod,Ngen,Nsims,3)) if bMode3D else np.zeros((NlocAlg,NerrMod,Ngen,Nsims))
    d0_est=np.zeros((NlocAlg,NerrMod,Ngen,Nsims,Ndim))
    tauE_est=np.zeros((NlocAlg,NerrMod,Ngen,Nsims))
    run_time=np.zeros((NlocAlg,NerrMod,Ngen,Nsims))
    
    lMapEstDFs = []
    for nc in range(NlocAlg):
        (aoa0Apriori,aoa0Quant,grouping,orientMthd)=lLocAlgs[nc][0:4]
        for nv in range(NerrMod):
            for ng in range(Ngen):
                for ns in tqdm(range(Nsims),desc=f'Location with {lLocAlgs[nc][4]} err {lErrMod[nv]} gen {mpgenInfo[ng]}'):
                    Np=errPathDF.loc[nv,ng,ns].shape[0]
                    t_start_point = time.time()
                    if aoa0Apriori:
                        if not np.isinf(aoa0Quant):
                            rot0_est[nc,nv,ng,ns]= np.round(allUserData.loc[ns][rotColNames] *aoa0Quant/(np.pi*2))*2*np.pi/aoa0Quant        
                        else:
                            rot0_est[nc,nv,ng,ns]= allUserData.loc[ns][rotColNames]
                        (d0_est[nc,nv,ng,ns,:],tauE_est[nc,nv,ng,ns],d_est)=loc.computeAllPaths(errPathDF.loc[nv,ng,ns],rotation=rot0_est[nc,nv,ng,ns])
                    else:
                    #TODO make changes in location estimator and get rid of these ifs
                        if not np.isinf(aoa0Quant):
                            o_args= { 'groupMethod':grouping,'initRotation': np.round(allUserData.loc[ns][rotColNames].to_numpy() *aoa0Quant/(np.pi*2))*2*np.pi/aoa0Quant  }
                        else:
                            o_args= { 'groupMethod':grouping}
                        (d0_est[nc,nv,ng,ns,:],tauE_est[nc,nv,ng,ns],d_est,rot0_est[nc,nv,ng,ns],_)= loc.computeAllLocationsFromPaths(errPathDF.loc[nv,ng,ns],orientationMethod=orientMthd,orientationMethodArgs=o_args)
                    run_time[nc,nv,ng,ns] = time.time() - t_start_point
                    lMapEstDFs.append(pd.DataFrame(data=d_est,columns=( ['Xs','Ys','Zs'] if bMode3D else  ['Xs','Ys'])))
    if bMode3D:        
        aoa0_est=rot0_est[:,:,:,:,0]
    else:
        aoa0_est=rot0_est
    allLocEstData=pd.DataFrame(index=pd.MultiIndex.from_product([range(NlocAlg), range(NerrMod),range(Ngen),range(Nsims)],names=["alg","errNo","genNo","ue"]),
                                columns=['X0','Y0','tauE','AoA0','runtime'],
                                data=np.column_stack([
                                    d0_est[:,:,:,:,0].reshape(-1),
                                    d0_est[:,:,:,:,1].reshape(-1),
                                    tauE_est.reshape(-1),
                                    aoa0_est.reshape(-1),
                                    run_time.reshape(-1)
                                    ]))
    if bMode3D:
        allLocEstData['Z0']=d0_est[:,:,:,:,2].reshape(-1)
        allLocEstData['ZoA0']=rot0_est[:,:,:,:,1].reshape(-1)
        allLocEstData['SoA0']=rot0_est[:,:,:,:,2].reshape(-1)
    allMapEstData=pd.concat(lMapEstDFs,names=["alg","errNo","genNo","ue",'n'],keys=pd.MultiIndex.from_product([range(NlocAlg), range(NerrMod), range(Ngen),range(Nsims)]))
    if not args.nosave: 
        allLocEstData.to_csv(outfoldername+'/locEstData.csv') 
        allMapEstData.to_csv(outfoldername+'/mapEstData.csv') 
    print("Total Location Time:%s seconds"%(time.time()-t_start_loc))


location_error=np.linalg.norm( allLocEstData[locColNames]-allUserData[locColNames] ,axis=-1).reshape(NlocAlg,NerrMod,Ngen,Nsims)
mapping_dif=allMapEstData[mapColNames]-allPathsData[mapColNames]
mapping_meandist=mapping_dif.apply(np.linalg.norm,axis=1).groupby(["alg","errNo","genNo","ue"]).mean()
mapping_error=mapping_meandist.to_numpy().reshape(NlocAlg,NerrMod,Ngen,Nsims)
d0_dumb=np.random.rand(Nsims,Ndim)*(dmax-dmin)+dmin
error_dumb=np.linalg.norm(d0_dumb-allUserData[locColNames],axis=-1)
d_dumb=np.random.rand(*allMapEstData.shape)*(dmax-dmin)+dmin
map_dumb=(allMapEstData-d_dumb).apply(np.linalg.norm,axis=1).groupby(["alg","errNo","genNo","ue"]).mean().to_numpy().reshape(NlocAlg,NerrMod,Ngen,Nsims)

errorCRLBnormalized=np.zeros((Ngen,Nsims))
mappingCRLBnormalized=np.zeros((Ngen,Nsims))
for ng in range(Ngen):
    for ns in range(Nsims):
        d0=allUserData.loc[ns][locColNames].to_numpy()
        d=allPathsData.loc[ng,ns,:][mapColNames].to_numpy()
        Npath=d.shape[0]
        if bMode3D:
            Tm=loc.getTParamToLoc(d0,d,['dTDoA','dAoA','dZoA'],['dx0','dy0','dz0'])
            scale=np.repeat([Ts,2*np.pi,2*np.pi],Npath)
        else:
            Tm=loc.getTParamToLoc(d0,d,['dTDoA','dAoA'],['dx0','dy0'])
            scale=np.repeat([Ts,2*np.pi],Npath)
        Tm=Tm/scale
        errorCRLBnormalized[ng,ns]=np.sqrt(np.trace(np.linalg.lstsq(Tm@Tm.T,np.eye(Ndim),rcond=None)[0])) 
        
        if bMode3D:
            Tm=loc.getTParamToLoc(d0,d,['dTDoA','dAoD','dZoD','dAoA','dZoA'],['dx','dy','dz'])
            scale=np.repeat([Ts,2*np.pi,np.pi,2*np.pi,np.pi],Npath)
        else:
            Tm=loc.getTParamToLoc(d0,d,['dTDoA','dAoD','dAoA'],['dx','dy'])
            scale=np.repeat([Ts,2*np.pi,2*np.pi],Npath)
        Tm=Tm/scale
        #TODO check 1/Npath, result not coherent
        mappingCRLBnormalized[ng,ns]=np.sqrt(np.trace(np.linalg.lstsq(Tm@Tm.T,np.eye(Ndim*Npath),rcond=None)[0]))/Npath

# mapping_error[:,:,x==0]=np.inf#fix better
tauE_err = np.abs( allLocEstData['tauE']-allUserData['tauE'] ).to_numpy().reshape(NlocAlg,NerrMod,Ngen,Nsims)
rot0_err=np.mean(np.abs(allLocEstData[rotColNames] - allUserData[rotColNames]),axis=1).to_numpy().reshape(NlocAlg,NerrMod,Ngen,Nsims)
runtime = allLocEstData.runtime.to_numpy().reshape(NlocAlg,NerrMod,Ngen,Nsims)

plt.close('all')
listOfLineTypes = ['-',':','-.','--']
listOfPatterns = [ None, "x" , "-"  , "/", "\\" , "|", "+" , "x", "o", "O", ".", "*" ]
listOfMarkersComplementary= [#markers are visually related vertically for example star vs pengagon, x vs square, + vs diamond etc
    ['o','s','d','v','^','<','>'],
    ['*','x','+','1','2','3','4'],
    ['p','|','D', 4 , 5 , 6 , 7 ]
]
listOfPrimaryColors = 'brgcmy'

def prettyErrName(errkey):
    T = errkey[0]
    if T=='no':
        return("Perfect Multipath Information")
    elif T=='S':
        value=float(errkey[1])
        return( f'AWGN Multipath error $\\sigma^2$={value} dB')
    elif T=='D':
        value=int(errkey[1])
        return( f'Dictionary Multipath Quantizer $N_D$={value}$^5$')

fig_ctr=0
if args.cdf:
    lCDF =[
        tuple(cdfcase.split('x'))
        for cdfcase in args.cdf.split(',')
    ]
    for cdf in lCDF:
        indErr = lErrMod.index(cdf)
        
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lLocAlgs[nc][4:]
            for ng in range(Ngen):
                if Ngen>1:
                    color=listOfPrimaryColors[ng]
                    marker=listOfMarkersComplementary[nc][ng]
                    lb=caseStr+" "+str(mpgenInfoStr[ng])
                else:
                    lb=caseStr
                plt.semilogx(np.percentile(location_error[nc,indErr,ng,~np.isnan(location_error[nc,indErr,ng,:])],np.linspace(0,100,21)),np.linspace(0,1,21),line+marker+color,label=lb)   
        # plt.semilogx(np.percentile(error_dumb,np.linspace(0,100,21)),np.linspace(0,1,21),':k',label="random guess")
        if lErrMod[indErr][0]=='D':
            Ndic=float(lErrMod[indErr][1])
            for ng in range(Ngen):
                line=listOfLineTypes[ng] if ng<4 else ':'
                mrk=listOfMarkersComplementary[1][ng-4] if ng>=4 else ''
                plt.semilogx(np.percentile(np.sqrt((1/Ndic)**2/12)*errorCRLBnormalized[ng,:],np.linspace(0,100,21)),np.linspace(0,1,21),line+mrk+'k',label="CRLB "+str(mpgenInfoStr[ng]))    
        plt.xlabel('Location error(m)')
        if cdf[0]!='no':
            plt.axis([.5e-2,2e2,0,1.1])
        plt.ylabel('C.D.F.')
        plt.title(prettyErrName(cdf))
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+(f'/cdflocerr_{cdf}.svg'))
            
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lLocAlgs[nc][4:]
            for ng in range(Ngen):
                if Ngen>1:
                    color=listOfPrimaryColors[ng]
                    marker=listOfMarkersComplementary[nc][ng]
                    lb=caseStr+" "+str(mpgenInfoStr[ng])
                else:
                    lb=caseStr
                mapping_error_data_valid = mapping_error[nc,indErr,ng,(~np.isnan(mapping_error[nc,indErr,ng,:]))&(~np.isinf(mapping_error[nc,indErr,ng,:]))]
                plt.semilogx(np.percentile(mapping_error_data_valid,np.linspace(0,100,21)),np.linspace(0,1,21),line+marker+color,label=lb)
        # plt.semilogx(np.percentile(map_dumb,np.linspace(0,100,21)),np.linspace(0,1,21),':k',label="random guess")
        if lErrMod[indErr][0]=='D':
            Ndic=float(lErrMod[indErr][1])
            for ng in range(Ngen):
                line=listOfLineTypes[ng] if ng<4 else ':'
                mrk=listOfMarkersComplementary[1][ng-4] if ng>=4 else ''
                plt.semilogx(np.percentile(np.sqrt((1/Ndic)**2/12)*mappingCRLBnormalized[ng,:],np.linspace(0,100,21)),np.linspace(0,1,21),line+mrk+'k',label="CRLB "+str(mpgenInfoStr[ng]))    
        plt.xlabel('Mapping error(m)')
        if cdf[0]!='no':
            plt.axis([.5e-2,2e2,0,1.1])
        plt.ylabel('C.D.F.')
        plt.title(prettyErrName(cdf))
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+(f'/cdfmaperr_{cdf}.svg'))


if args.pcl:
    lPCL =[
        tuple(case.split(':'))
        for case in args.pcl.split(',')
    ]
    
    for pcl in lPCL:
        Pctl = float(pcl[1])
        errType = pcl[0]
        errCaseMask=[x[0]==errType for x in lErrMod]
        # if (errType == 'D') and not (np.any([[y=='inf' for y in x[1:]] for x in lErrMod if x[0]==errType ])):
        #     errLabels=["$10^{%.0f}$"%(np.log10(np.prod(np.array(x[1:],dtype=int)))) for x in lErrMod if x[0]==errType]
        #     errTitle = "Dictionary memory size (number of columns)"
        # else:
        errLabels=np.array([ float(x[1]) for x in lErrMod if x[0]==errType])
        errTitle = "Per Dimension Dictionary Size"
        errTics=np.arange(len(errLabels))
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lLocAlgs[nc][4:]
            for ng in range(Ngen):
                if Ngen>1:
                    color=listOfPrimaryColors[ng]
                    marker=listOfMarkersComplementary[nc][ng]
                    lb=caseStr+" "+str(mpgenInfoStr[ng])
                else:
                    lb=caseStr
                plt.semilogy(np.arange(len(errLabels)),np.percentile(location_error[nc,errCaseMask,ng,:],Pctl,axis=1),line+marker+color,label=lb)
            # plt.semilogy(np.arange(len(errLabels)),np.percentile(location_error[:,:,losAll][nc,errCaseMask,:],Pctl,axis=1),line+marker+'r',label=caseStr)
            # plt.semilogy(np.arange(len(errLabels)),np.percentile(location_error[:,:,~losAll][nc,errCaseMask,:],Pctl,axis=1),line+marker+'g',label=caseStr)
        # plt.semilogy(errTics,np.ones_like(errTics)*np.percentile(error_dumb,Pctl),':k',label="random guess")
        lNant=np.array([float(x[1]) for x in lErrMod if x[0]=='D'])
        for ng in range(Ngen):
            line=listOfLineTypes[ng] if ng<4 else ':'
            mrk=listOfMarkersComplementary[1][ng-4] if ng>=4 else ''
            plt.semilogy(errTics,np.percentile(errorCRLBnormalized[ng,:],Pctl)*np.sqrt((1/lNant)**2/12),line+mrk+'k',label="CRLB "+str(mpgenInfoStr[ng]))       
        plt.xticks(ticks=errTics,labels=errLabels)
        plt.xlabel(errTitle)
        plt.ylabel('%.1f percentile location error(m)'%(Pctl))
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+'/err_vs_%s.svg'%(errType))
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lLocAlgs[nc][4:]
            for ng in range(Ngen):
                if Ngen>1:
                    color=listOfPrimaryColors[ng]
                    marker=listOfMarkersComplementary[nc][ng]
                    lb=caseStr+" "+str(mpgenInfoStr[ng])
                else:
                    lb=caseStr
                plt.semilogy(np.arange(len(errLabels)),np.percentile(mapping_error[nc,errCaseMask,ng,:],Pctl,axis=1),line+marker+color,label=lb)
        # plt.semilogy(errTics,np.ones_like(errTics)*np.percentile(map_dumb,80),':k',label="random guess")
        lNant=np.array([float(x[1]) for x in lErrMod if x[0]=='D']) 
        for ng in range(Ngen):
            line=listOfLineTypes[ng] if ng<4 else ':'
            mrk=listOfMarkersComplementary[1][ng-4] if ng>=4 else ''
            plt.semilogy(errTics,np.percentile(mappingCRLBnormalized[ng,:],Pctl)*np.sqrt((1/lNant)**2/12),line+mrk+'k',label="CRLB "+str(mpgenInfoStr[ng]))       
        plt.xticks(ticks=errTics,labels=errLabels)
        plt.xlabel(errTitle)
        plt.ylabel('%.1f percentile mapping error(m)'%(Pctl))
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+'/map_vs_%s.svg'%(errType))
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lLocAlgs[nc][4:]
            for ng in range(Ngen):
                if Ngen>1:
                    color=listOfPrimaryColors[ng]
                    marker=listOfMarkersComplementary[nc][ng]
                    lb=caseStr+" "+str(mpgenInfoStr[ng])
                else:
                    lb=caseStr
                plt.semilogy(np.arange(len(errLabels)),np.percentile(1e9*tauE_err[nc,errCaseMask,ng,:],Pctl,axis=(1)),line+marker+color,label=lb)
        plt.xticks(ticks=errTics,labels=errLabels)
        plt.xlabel(errTitle)
        plt.ylabel('%.1f percentile clock error(ns)'%(Pctl))
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+'/tau_vs_%s.svg'%(errType))
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lLocAlgs[nc][4:]
            for ng in range(Ngen):
                if Ngen>1:
                    color=listOfPrimaryColors[ng]
                    marker=listOfMarkersComplementary[nc][ng]
                    lb=caseStr+" "+str(mpgenInfoStr[ng])
                else:
                    lb=caseStr
                plt.semilogy(np.arange(len(errLabels)),np.percentile(rot0_err[nc,errCaseMask,ng,:],Pctl,axis=1),line+marker+color,label=lb)
        plt.xticks(ticks=errTics,labels=errLabels)
        plt.xlabel('Channel Error')
        plt.ylabel('%.1f percentile AoA0 error(º)'%(Pctl))
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+'/aoa0_vs_%s.svg'%(errType))
            

if args.map:
    lMAP =[
        tuple(case.split('x'))
        for case in args.map.split(',')
    ]
    if args.mapcap:
        Nuemap = int(args.mapcap)
    else:
        Nuemap = Nsims
    for themap in lMAP:
        indErr = lErrMod.index(themap)         
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        plt.plot(allUserData.X0.loc[0:Nuemap],allUserData.Y0.loc[0:Nuemap],'ok',fillstyle='none',label='locations')
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lLocAlgs[nc][4:]
            for ng in range(Ngen):
                if Ngen>1:
                    color=listOfPrimaryColors[ng]
                    marker=listOfMarkersComplementary[1][1+ng]
                    lb=caseStr+" "+str(mpgenInfoStr[ng])
                else:
                    lb=caseStr
                plt.plot(np.vstack((allUserData.loc[0:Nuemap].X0,allLocEstData.loc[nc,indErr,ng,0:Nuemap].X0)),np.vstack((allUserData.loc[0:Nuemap].Y0,allLocEstData.loc[nc,indErr,ng,0:Nuemap].Y0)),line+marker+color,fillstyle='none',label=lb)
        handles, labels = plt.gca().get_legend_handles_labels()
        # labels will be the keys of the dict, handles will be values
        temp = {k:v for k,v in zip(labels, handles)}
        plt.legend(temp.values(), temp.keys(), loc='best')
        plt.xlabel('$d_{ox}$ (m)')
        plt.ylabel('$d_{oy}$ (m)')
        if args.print:
            plt.savefig(outfoldername+f'/errormap_{cdf}.svg')
       
if args.vso:
    lVSO =[
        tuple(case.split('x'))
        for case in args.vso.split(',')
    ]
    
    for vso in lVSO:
        indErr = lErrMod.index(vso)         
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)  
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lLocAlgs[nc][4:]
            for ng in range(Ngen):
                if Ngen>1:
                    color=listOfPrimaryColors[ng]
                    marker=listOfMarkersComplementary[nc][1+ng]
                    lb=caseStr+" "+str(mpgenInfoStr[ng])
                else:
                    lb=caseStr
                plt.loglog(rot0_err[nc,indErr,ng,:],location_error[nc,indErr,ng,:],marker+color,label=lb)
        plt.xlabel('$\hat{aoa}$_o error (rad)')
        plt.ylabel('Location error (m)')
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+f'/err_vs_aoa0_{vso}.svg')

if args.rtm:
    lRTM =[#TODO multi color bar plot per dictionaries instead
        tuple(case.split('x'))
        for case in args.rtm.split(',')
    ]     
    Nrtm=len(lRTM)
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)       
    barwidth=0.9/Nrtm
    for n in range(0,Nrtm):
        rtm = lRTM[n]
        indErr = lErrMod.index(rtm)  
        offset=(n-(Nrtm-1)/2)*barwidth 
        plt.bar(np.arange(NlocAlg)+offset,np.mean(runtime[:,indErr,ng,:],axis=1),width=barwidth,label=f'{rtm}')
    plt.xticks(np.arange(NlocAlg),[x[4] for x in lLocAlgs])
    plt.xlabel('Algoritm')
    plt.ylabel('Run time per iteration (s)')
    plt.gca().set_yscale('log')
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+f'/runtime_{rtm}.svg')

if args.show:
    plt.show()
