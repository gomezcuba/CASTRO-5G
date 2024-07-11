#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

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
#parameters that affect error

# case 1: no error
parser.add_argument('--noerror', help='Zero error case', action='store_true')

# case 2: add a personalized number of error points
parser.add_argument('-S', type=str,help='S:min:max S gaussian error points with stds min-max in dB')
parser.add_argument('-D', type=str,help='64:64:64,64:128:inf Quantization error models, comma separated')

#parameters that affect multipath generator
parser.add_argument('-G', type=str,help='Type of generator')

#parameters that affect location algorithms
parser.add_argument('--algs', type=str,help='comma-separated list of algorithms')

#parameters that affect plots
parser.add_argument('--cdf', type=str,help='plot CDF for a given comma-separated list of errors')
parser.add_argument('--pcl', type=str,help='comma-separated list of error axis-percentile plots (example "D:80")')
parser.add_argument('--map', type=str,help='plot map')
parser.add_argument('--vso', type=str,help='scatter plot of location error vs orientation')

#parameters that affect workflow
parser.add_argument('--label', type=str,help='str label appended to storage files')
parser.add_argument('--nosave', help='Do not save simulation data to new results file', action='store_true')
parser.add_argument('--nompg',help='Do not perform multipath generation, load existing file', action='store_true')
parser.add_argument('--noloc',help='Do not perform location estimation, load existing file', action='store_true')
parser.add_argument('--show', help='Open plot figures during execution', action='store_true')
parser.add_argument('--print', help='Save plot files in eps to results folder', action='store_true')
# parser.add_argument('--svg', help='Save plot files in svg to results folder', action='store_true')

## TO DO: (In progress) -- CLI arguments (line100)
#parser.add_argument('-xmax',type=int,help='Simulation model x-axis max. size coordinate (meters from the origin)')
#parser.add_argument('-xmin',type=int,help='Simulation model x-axis min. size coordinate (meters from the origin)')
#parser.add_argument('-ymax',type=int,help='Simulation model y-axis max. size coordinate (meters from the origin)')
#parser.add_argument('-ymin',type=int,help='Simulation model y-axis min. size coordinate (meters from the origin)')
#refine to make it consistent before reestructuring all this code

# args = parser.parse_args("--nompg --noloc -N 1000 -S 7 -D 16:16:16,64:64:64,256:256:256,1024:1024:1024,4096:4096:4096,16:inf:inf,64:inf:inf,256:inf:inf,1024:inf:inf,4096:inf:inf,inf:inf:16,inf:inf:64,inf:inf:256,inf:inf:1024,inf:inf:4096 --noerror --label test --show --print".split(' '))

# args = parser.parse_args("--noloc --nompg -N 100 --noerror -D=16:16:16,64:64:64,128:128:128,256:256:256,512:512:512,1024:1024:1024 -G 3gpp --noerror --label newadapt --show --print --cdf=no:0:0:0,dic:256:256:256 --pcl=dic:50 --map=no:0:0:0,dic:256:256:256 --vso=no:0:0:0,dic:256:256:256".split(' '))
# args = parser.parse_args("--noloc --nompg -N 100 --noerror -D=16:16:16,64:64:64,128:128:128,256:256:256,512:512:512,1024:1024:1024 -G 3gpp --noerror --label test --show --print --cdf=no:0:0:0,dic:256:256:256 --pcl=dic:75 --map=no:0:0:0,dic:256:256:256 --vso=no:0:0:0,dic:256:256:256".split(' '))
# args = parser.parse_args("-N 1000 --nompg --noloc -G Geo:20 --noerror -D=64:64:64,256:256:256,1024:1024:1024 --label test --show --print --cdf=no:0:0:0,dic:256:256:256 --pcl=dic:80 --map=no:0:0:0,dic:256:256:256 --vso=no:0:0:0,dic:256:256:256".split(' '))
args = parser.parse_args("-N 100 --nompg -G 3gpp --noerror -S=2:0:20 -D=16:16:16,64:64:64,256:256:256 --label test --show --print --cdf=no:0:0:0,dic:16:16:16,dic:64:64:64,dic:256:256:256 --pcl=dic:80 --map=no:0:0:0,dic:256:256:256 --vso=no:0:0:0,dic:256:256:256".split(' '))

# numero de simulacions
Nsims=args.N if args.N else 100

# error vector modeling
if args.noerror:
    lErrMod=[('no','0','0','0')]
else:
    lErrMod=[]
if args.S:
    NS,minStd,maxStd = args.S.split(':')
    lSTD=np.logspace(float(minStd)/10,float(maxStd)/10,int(NS))
    lErrMod = lErrMod+[('std',x,x,x) for x in lSTD]
if args.D:
    lDicSizes=[tuple(['dic']+[x for x in y.split(':')]) for y in args.D.split(',') ]
    lErrMod = lErrMod+lDicSizes    
#TODO add ifs etc. to make this selectable with OMP
NerrMod=len(lErrMod)

# multipath generator
if args.G:
    mpgenInfo = args.G.split(':')
else:        
    mpgenInfo = ['Geo','20']
mpgen=mpgenInfo[0]
#TODO: Aquí parece que hai bastantes parametros que completar

#location algorythms - evolución de location estimator
if args.algs:
    lLocAlgs=[
            tuple(case.split(':'))
            for case in args.algs.split(',')
            ]
    #TODO define more elegant syntax for cases and better parser that converts bool properly
else:
    lLocAlgs=[#a opriori aoa0, quantized aoa0, grouping method, optimization method
       (True,np.inf,'',''),
        # (True,64,'',''),
       # (False,np.inf,'3path','brute'),
       # (False,np.inf,'3path','lm'),
#       (False,np.inf,'3path','zero'),
       # (False,64,'3path','lm'),
#       (False,64,'3path','zero'),
#       (False,np.inf,'drop1','brute'),
        # (False,np.inf,'drop1','lm'),
#       (False,np.inf,'drop1','zero'),
        (False,64,'drop1','lm'),
#       (False,64,'drop1','zero'),
       ]
NlocAlg =len(lLocAlgs)

if args.label:
    outfoldername="../Results/MPRayLocresults%s"%(args.label)
else:
    outfoldername="../Results/MPRayLocresults-%d-%d-%s"%(NerrMod,Nsims,mpgen)
if not os.path.isdir(outfoldername):
    os.mkdir(outfoldername)

#TODO: create command line arguments for these parameters
dmax=np.array([100,100])
dmin=np.array([-100,-100])
c=3e8
Ts=1.0/400e6 #2.5ns
Ds=320e-9 #Ts*128 FIR filter
sigmaTauE=40e-9

# if this parameter is present we get the mpg data from a known file (better for teset&debug)
if args.nompg:    
    allUserData=pd.read_csv(outfoldername+'/userGenData.csv',index_col=['ue']) 
    allPathsData=pd.read_csv(outfoldername+'/chanGenData.csv',index_col=['ue','n','m']) 
else:        
    t_start_gen=time.time()
    # TODO: this has to be modeled in a separate class
    d0=np.random.rand(Nsims,2)*(dmax-dmin)+dmin
    aod0=np.mod( np.arctan2(d0[:,1],d0[:,0]) , 2*np.pi)
    aoa0=np.random.rand(Nsims)*2*np.pi #receiver angular measurement offset
    toa0=np.linalg.norm(d0,axis=-1)/c
    tauE=np.random.rand(Nsims)*sigmaTauE
    allUserData = pd.DataFrame(index=pd.Index(np.arange(Nsims),name="ue"),
                               data={
                                   "X0":d0[:,0],
                                   "Y0":d0[:,1],
                                   "AoD0":aod0,
                                   "AoA0":aoa0,
                                   "tauE":tauE
                                   })
    if mpgen == 'Geo':        
        Npath=int(mpgenInfo[1])
        #generate locations and compute multipath 
        d=np.random.rand(Nsims,Npath,2)*(dmax-dmin)+dmin
        #angles from locations
        #TODO 3D
        aod=np.mod( np.arctan2(d[:,:,1],d[:,:,0]) , 2*np.pi)
        aoa=np.mod( np.arctan2((d[:,:,1]-d0[:,None,1]),(d[:,:,0]-d0[:,None,0])) , 2*np.pi)
        daoa=np.mod(aoa-aoa0[:,None],2*np.pi)
        #delays based on distance
        li=np.linalg.norm(d,axis=-1)+np.linalg.norm(d-d0[:,None,:],axis=-1)
        toa=li/c
        tdoa=toa-toa0[:,None]-tauE[:,None]
        allPathsData = pd.DataFrame(index=pd.MultiIndex.from_product([np.arange(Nsims),np.arange(Npath)],names=["ue","n"]),
                                    data={
                                        "AoD" : aod.reshape(-1),
                                        "AoA" : daoa.reshape(-1),
                                        "TDoA" : tdoa.reshape(-1),
                                        "Xs": d[:,:,0].reshape(-1),
                                        "Ys": d[:,:,1].reshape(-1)
                                        })
    elif mpgen == "3gpp":        
        # TODO: introducir param de entrada para regular blargeBW, scenario, etc
        # Tamen mais tarde - Elección de escenario vía param. de entrada
        model = mpg.ThreeGPPMultipathChannelModel(scenario="UMi",bLargeBandwidthOption=True)        
        txPos = (0,0,10)        
        Npath = 50
        # Npath = model.maxM*np.max( model.scenarioParams.loc['N'] )                
        lpathDFs = []
        losAll = np.zeros((Nsims),dtype=bool)
        for n in tqdm(range(Nsims),desc=f'Generating {Nsims} 3GPP multipath channels'):
            #TODO 3D
            rxPos = (d0[n,0],d0[n,1],1.5)
            plinfo,macro,clusters,subpaths = model.create_channel(txPos,rxPos)
            losAll[n]=plinfo[0]
            # (txPos,rxPos,plinfo,clusters,subpaths)  = model.fullFitAoA(txPos,rxPos,plinfo,clusters,subpaths)
            (txPos,rxPos,plinfo,clusters,subpaths)  = model.randomFitEpctClusters(txPos,rxPos,plinfo,clusters,subpaths,Ec=.75,Es=.75,P=[0,.5,.5,0],mode3D=False)
            # txArrayAngle = 0#deg
            # rxArrayAngle = np.mod(np.pi+np.arctan2(y0[n],x0[n]),2*np.pi)*180/np.pi
            # (txPos,rxPos,plinfo,clusters,subpaths)  = model.fullDeleteBacklobes(txPos,rxPos,plinfo,clusters,subpaths,tAoD=txArrayAngle,rAoA=rxArrayAngle)
            #remove subpaths that have not been converted to first order
            subpathsProcessed = subpaths.loc[~np.isinf(subpaths.Xs)].copy() # must be a hard copy and not a mere "view" of the original DF
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
            subpathsProcessed.AoD = np.mod( subpathsProcessed.AoD*np.pi/180 , 2*np.pi)
            subpathsProcessed.AoA = np.mod( subpathsProcessed.AoA*np.pi/180 -aoa0[n], 2*np.pi)
            lpathDFs.append(subpathsProcessed.rename(columns={"AoA":"DAoA"}))
        allPathsData=pd.concat(lpathDFs,keys=np.arange(Nsims),names=["ue",'n','m'])
    else:
        print("MultiPath generation method %s not recognized"%mpgen)
    
    if not args.nosave: 
        allUserData.to_csv(outfoldername+'/userGenData.csv') 
        allPathsData.to_csv(outfoldername+'/chanGenData.csv') 
    print("Total Multipath Generation Time:%s seconds"%(time.time()-t_start_gen))
    

nPathAll= allPathsData.groupby(level='ue').size().to_numpy()
Nmaxpath=np.max(nPathAll)
t_start_err = time.time()
lErrorPathDFs = []
for nv in tqdm(range(NerrMod),desc='applying error models to paths'):
    (errType,c1,c2,c3)=lErrMod[nv]
    if errType=='no':
        errPathDF = allPathsData.copy()
    elif errType=='std': 
        errPathDF = allPathsData.copy()
        Ntot = errPathDF.shape[0]
        errPathDF.AoD  = np.mod(errPathDF.AoD +float(c1)*2*np.pi*np.random.randn(Ntot),2*np.pi)
        errPathDF.DAoA = np.mod(errPathDF.DAoA+float(c2)*2*np.pi*np.random.randn(Ntot),2*np.pi)
        errPathDF.TDoA += float(c3)*Ds*np.random.randn(Ntot)
    elif errType=='dic':
        errPathDF = allPathsData.copy()
        if not c1=='inf':
            errPathDF.AoD  = np.mod(np.round( errPathDF.AoD  *int(c1)/2/np.pi)*2*np.pi/int(c1),2*np.pi)
        if not c2=='inf':
            errPathDF.DAoA = np.mod(np.round( errPathDF.DAoA *int(c2)/2/np.pi)*2*np.pi/int(c2),2*np.pi)
        if not c3=='inf':
            errPathDF.TDoA  =np.round(errPathDF.TDoA *int(c3)/Ds )*Ds/int(c3)
    #TODO Compressed Sensing
    else:
        print("Multipath estimation error model %s to be written"%errType)
    lErrorPathDFs.append(errPathDF)
errPathDF=pd.concat(lErrorPathDFs,keys=range(NerrMod),names=["errNo","ue",'n','m'])

print("Total Multipath Estimation Time:%s seconds"%(time.time()-t_start_err))

if args.noloc: 
    data=np.load(outfoldername+'/locEstData.npz') 
    aoa0_est=data["aoa0_est"]
    d0_est=data["d0_est"]
    d_est=data["d_est"]
    tauE_est=data["tauE_est"]
    run_time=data["run_time"]    
    # errPathDF=pd.read_csv(outfoldername+'/pathErrData.csv',index_col=['errNo','ue','n','m'])
    loc=MultipathLocationEstimator.MultipathLocationEstimator(nPoint=100,orientationMethod='lm')    
else:
    t_start_loc=time.time() 
    aoa0_est=np.zeros((NlocAlg,NerrMod,Nsims))
    d0_est=np.zeros((NlocAlg,NerrMod,Nsims,2))
    d_est=np.zeros((NlocAlg,NerrMod,Nsims,Nmaxpath,2))
    tauE_est=np.zeros((NlocAlg,NerrMod,Nsims))
    run_time=np.zeros((NlocAlg,NerrMod))
    
    loc=MultipathLocationEstimator.MultipathLocationEstimator(nPoint=100,orientationMethod='lm')
    
    for nc in range(NlocAlg):
        (aoa0Apriori,aoa0Quant,grouping,orientMthd)=lLocAlgs[nc]
        for nv in range(NerrMod):  
            t_start_point = time.time()
            for ns in tqdm(range(Nsims),desc=f'Location with {lLocAlgs[nc]} err {lErrMod[nv]}'):
                Np=errPathDF.loc[nv,ns].shape[0]
                if aoa0Apriori:
                    if not np.isinf(aoa0Quant):
                        aoa0_est[nc,nv,ns]= np.round( allUserData.loc[ns].AoA0 *aoa0Quant/(np.pi*2))*2*np.pi/aoa0Quant        
                    else:
                        aoa0_est[nc,nv,ns]= allUserData.loc[ns].AoA0
                    (d0_est[nc,nv,ns,:],tauE_est[nc,nv,ns],d_est[nc,nv,ns,0:Np,:])=loc.computeAllPaths(errPathDF.loc[nv,ns],rotation=aoa0_est[nc,nv,ns])
                else:
                #TODO make changes in location estimator and get rid of these ifs
                    if not np.isinf(aoa0Quant):
                        o_args= { 'groupMethod':grouping,'hintRotation': np.round(allUserData.AoA0.loc[ns]*aoa0Quant/(np.pi*2))*2*np.pi/aoa0Quant }
                    else:
                        o_args= { 'groupMethod':grouping}
                    (d0_est[nc,nv,ns],tauE_est[nc,nv,ns],d_est[nc,nv,ns,0:Np,:],aoa0_est[nc,nv,ns],_)= loc.computeAllLocationsFromPaths(errPathDF.loc[nv,ns],orientationMethod=orientMthd,orientationMethodArgs=o_args)
            run_time[nc,nv] = time.time() - t_start_point
    if not args.nosave: 
        np.savez(outfoldername+'/locEstData.npz',
                aoa0_est=aoa0_est,
                d0_est=d0_est,
                d_est=d_est,
                tauE_est=tauE_est,
                run_time=run_time)
        # errPathDF.to_csv(outfoldername+'/pathErrData.csv')
    print("Total Location Time:%s seconds"%(time.time()-t_start_loc))

d0=allUserData[['X0','Y0']].to_numpy()
location_error=np.linalg.norm(d0_est-d0,axis=-1)
# mapping_dist=allPathsData[['Xs','Ys']]-errPathDF[['Xs','Ys']]
# mapping_dist.apply(np.linalg.norm,axis=1)
# mapping_error=mapping_dist.apply(np.linalg.norm,axis=1).groupby(['errNo','ue']).sum().to_numpy().reshape(NlocAlg,NerrMod,Nsims)
d=np.zeros((Nsims,Nmaxpath,2))
for ns in range(Nsims):
    d[ns,0:nPathAll[ns],:]=allPathsData[['Xs','Ys']].loc[ns]
mapping_error=np.sum(np.linalg.norm(d-d_est,axis=-1),axis=-1)/nPathAll

# mapping_error[:,:,x==0]=np.inf#fix better
tauE_err = np.abs(tauE_est+allUserData.tauE.to_numpy())
aoa0_err=np.abs(allUserData.AoA0.to_numpy()-aoa0_est)*180/np.pi
d0_dumb=np.random.rand(Nsims,2)*(dmax-dmin)+dmin
error_dumb=np.linalg.norm(d0-d0_dumb,axis=-1)
d_dumb=np.random.rand(Nsims,Nmaxpath,2)*(dmax-dmin)+dmin
map_dumb=np.linalg.norm(d-d_dumb,axis=-1)
plt.close('all')

def lineCfgFromAlg(algCfg):
    aoa0Apriori,aoa0Quant,grouping,orientMthd=algCfg
    if aoa0Apriori:
        caseStr="LS Location" if np.isinf(aoa0Quant) else "aoa0 quantized sensor"
        color='b'
        marker='o' if np.isinf(aoa0Quant) else '*'
        line=':'
    else:
        caseStr="%s - %s %s"%(grouping,orientMthd,('Q-ini' if aoa0Quant else 'BF-ini') if orientMthd == 'lm' else '')
        line='-' if grouping=='drop1' else ('-.' if orientMthd == 'lm' else ':')
        marker='x' if aoa0Quant else 's'
        color='r' if orientMthd=='brute' else 'g'        
    return(caseStr,line,marker,color)

fig_ctr=0
if args.cdf:
    lCDF =[
        tuple(cdfcase.split(':'))
        for cdfcase in args.cdf.split(',')
    ]
    for cdf in lCDF:
        indErr = lErrMod.index(cdf)
        
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lineCfgFromAlg(lLocAlgs[nc])
            plt.semilogx(np.percentile(location_error[nc,indErr,~np.isnan(location_error[nc,indErr,:])],np.linspace(0,100,21)),np.linspace(0,1,21),line+marker+color,label=caseStr)        
        plt.semilogx(np.percentile(error_dumb,np.linspace(0,100,21)),np.linspace(0,1,21),':k',label="random guess")
        # Ts = loc.getTParamToLoc(d0[:,0],d0[:,1],toa0+tauE,aoa0,allPathsData.Xs.to_numpy(),allPathsData.Ys.to_numpy(),['dDAoA',],['dx0','dy0'])
        # varaoaDist=np.var(np.minimum(np.mod(aoa-aoa0-daoa_est[indErr,:,:],np.pi*2),2*np.pi-np.mod(aoa-aoa0-daoa_est[indErr,:,:],np.pi*2))) * (Npath*Nsims)/np.sum(nvalid)
        # M=np.matmul(Ts.transpose([2,1,0]),Ts.transpose([2,0,1]))
        # errorCRLBnormalized = np.array([np.sqrt(np.trace(np.linalg.lstsq(M[n,:,:],np.eye(2),rcond=None)[0])) for n in range(M.shape[0])])
        # plt.semilogx(np.percentile(np.sqrt(varaoaDist)*errorCRLBnormalized,np.linspace(0,100,21)),np.linspace(0,1,21),'--k',label="approx. CRLB")    
        # if lErrMod[indErr][0]=='dic':
        #     Nrant=float(lErrMod[indErr][2])
            # plt.semilogx(np.percentile(np.sqrt((np.pi/Nrant)**2/12)*errorCRLBnormalized,np.linspace(0,100,21)),np.linspace(0,1,21),'--k',label="approx. CRLB")    
        plt.xlabel('Location error(m)')
        plt.ylabel('C.D.F.')
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+('/cdflocerr_%s-%s-%s-%s.svg'%tuple(cdf)))
            
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lineCfgFromAlg(lLocAlgs[nc])
            mapping_error_data_valid = mapping_error[nc,indErr,(~np.isnan(mapping_error[nc,indErr,:]))&(~np.isinf(mapping_error[nc,indErr,:]))]
            plt.semilogx(np.percentile(mapping_error_data_valid,np.linspace(0,100,21)),np.linspace(0,1,21),line+marker+color,label=caseStr)
        plt.semilogx(np.percentile(map_dumb,np.linspace(0,100,21)),np.linspace(0,1,21),':k',label="random guess")
        # if Npath<=50:
        #     Ts = loc.getTParamToLoc(x0,y0,tauE+toa0,aoa0,x,y,['dDAoA'],['dx','dy'])            
        #     M=np.matmul(Ts.transpose([2,1,0]),Ts.transpose([2,0,1]))
        #     #(1/Npath)*
        #     errorCRLBnormalized = np.array([np.sqrt(np.trace(np.linalg.lstsq(M[n,:,:],np.eye(2*Npath),rcond=None)[0])) for n in range(M.shape[0])])
        #     varaoaDist=np.var(np.minimum(np.mod(aoa-aoa0-daoa_est[indErr,:,:],np.pi*2),2*np.pi-np.mod(aoa-aoa0-daoa_est[indErr,:,:],np.pi*2))) * (Npath*Nsims)/np.sum(nvalid)
        #     plt.semilogx(np.percentile(np.sqrt(varaoaDist)*errorCRLBnormalized,np.linspace(0,100,21)),np.linspace(0,1,21),'--k',label="$\\sim$ CRLB")    
            # lNant=np.array([float(x[2]) for x in lErrMod if x[0]=='dic' and x[1]=='inf' and x[3]=='inf']) 
            # plt.semilogy(errTics,np.percentile(errorCRLBnormalized,80)*np.sqrt((np.pi/lNant)**2/12),'--k',label="approx. CRLB")       
        plt.xlabel('Mapping error(m)')
        plt.ylabel('C.D.F.')
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+('/cdfmaperr_%s-%s-%s-%s.svg'%tuple(cdf)))


if args.pcl:
    lPCL =[
        tuple(case.split(':'))
        for case in args.pcl.split(',')
    ]
    
    for pcl in lPCL:
        Pctl = float(pcl[1])
        errType = pcl[0]
        errCaseMask=[x[0]==errType for x in lErrMod]
        # if (errType == 'dic') and not (np.any([[y=='inf' for y in x[1:]] for x in lErrMod if x[0]==errType ])):
        #     errLabels=["$10^{%.0f}$"%(np.log10(np.prod(np.array(x[1:],dtype=int)))) for x in lErrMod if x[0]==errType]
        #     errTitle = "Dictionary memory size (number of columns)"
        # else:
        errLabels=[ "%s:%s:%s"%(x[1:]) for x in lErrMod if x[0]==errType]
        errTitle = "Dictionary Size"
        errTics=np.arange(len(errLabels))
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lineCfgFromAlg(lLocAlgs[nc])
            plt.semilogy(np.arange(len(errLabels)),np.percentile(location_error[nc,errCaseMask,:],Pctl,axis=1),line+marker+color,label=caseStr)
            # plt.semilogy(np.arange(len(errLabels)),np.percentile(location_error[:,:,losAll][nc,errCaseMask,:],Pctl,axis=1),line+marker+'r',label=caseStr)
            # plt.semilogy(np.arange(len(errLabels)),np.percentile(location_error[:,:,~losAll][nc,errCaseMask,:],Pctl,axis=1),line+marker+'g',label=caseStr)
        # plt.semilogy(errTics,np.ones_like(errTics)*np.percentile(error_dumb,80),':k',label="random guess")
        # Ts = loc.getTParamToLoc(x0,y0,tauE+toa0,aoa0,x,y,['dDAoA'],['dx0','dy0'])
        # M=np.matmul(Ts.transpose([2,1,0]),Ts.transpose([2,0,1]))
        # errorCRLBnormalized = np.array([np.sqrt(np.trace(np.linalg.lstsq(M[n,:,:],np.eye(2),rcond=None)[0])) for n in range(M.shape[0])])
        # varaoaDist=np.var(np.minimum(np.mod(aoa-aoa0-daoa_est[errCaseMask,:,:],np.pi*2),2*np.pi-np.mod(aoa-aoa0-daoa_est[errCaseMask,:,:],np.pi*2)),axis=(1,2)) * (Npath*Nsims)/np.sum(nvalid)
        # plt.semilogy(errTics,np.percentile(errorCRLBnormalized,80)*np.sqrt(varaoaDist),'--k',label="$\\sim$ CRLB")      
        # lNant=np.array([float(x[2]) for x in lErrMod if x[0]=='dic' and x[1]=='inf' and x[3]=='inf']) 
        # plt.semilogy(errTics,np.percentile(errorCRLBnormalized,80)*np.sqrt((np.pi/lNant)**2/12),'--k',label="approx. CRLB")       
        plt.xticks(ticks=errTics,labels=errLabels)
        plt.xlabel(errTitle)
        plt.ylabel('%.1f percentile location error(m)'%(Pctl))
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+'/err_vs_%s.svg'%(errType))
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lineCfgFromAlg(lLocAlgs[nc])
            plt.semilogy(np.arange(len(errLabels)),np.percentile(mapping_error[nc,errCaseMask,:],Pctl,axis=1),line+marker+color,label=caseStr)
        plt.semilogy(errTics,np.ones_like(errTics)*np.percentile(map_dumb,80),':k',label="random guess")
        # if Npath<=50:
        #     Ts = loc.getTParamToLoc(x0,y0,tauE+toa0,aoa0,x,y,['dDAoA'],['dx','dy'])            
        #     M=np.matmul(Ts.transpose([2,1,0]),Ts.transpose([2,0,1]))
        #     #(1/Npath)*
        #     errorCRLBnormalized = np.array([np.sqrt(np.trace(np.linalg.lstsq(M[n,:,:],np.eye(2*Npath),rcond=None)[0])) for n in range(M.shape[0])])
        #     varaoaDist=np.var(np.minimum(np.mod(aoa-aoa0-daoa_est[errCaseMask,:,:],np.pi*2),2*np.pi-np.mod(aoa-aoa0-daoa_est[errCaseMask,:,:],np.pi*2)),axis=(1,2)) * (Npath*Nsims)/np.sum(nvalid)
        #     plt.semilogy(errTics,np.percentile(errorCRLBnormalized,80)*np.sqrt(varaoaDist),'--k',label="$\\sim$ CRLB")      
        #     # lNant=np.array([float(x[2]) for x in lErrMod if x[0]=='dic' and x[1]=='inf' and x[3]=='inf']) 
        #     # plt.semilogy(errTics,np.percentile(errorCRLBnormalized,80)*np.sqrt((np.pi/lNant)**2/12),'--k',label="approx. CRLB")       
        plt.xticks(ticks=errTics,labels=errLabels)
        plt.xlabel(errTitle)
        plt.ylabel('%.1f percentile mapping error(m)'%(Pctl))
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+'/map_vs_%s.svg'%(errType))
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lineCfgFromAlg(lLocAlgs[nc])
            plt.semilogy(np.arange(len(errLabels)),np.percentile(1e9*tauE_err[nc,errCaseMask,:],Pctl,axis=(1)),line+marker+color,label=caseStr)
        plt.xticks(ticks=errTics,labels=errLabels)
        plt.xlabel(errTitle)
        plt.ylabel('%.1f percentile clock error(ns)'%(Pctl))
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+'/tau_vs_%s.svg'%(errType))
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lineCfgFromAlg(lLocAlgs[nc])
            plt.semilogy(np.arange(len(errLabels)),np.percentile(aoa0_err[nc,errCaseMask,:],Pctl,axis=(1)),line+marker+color,label=caseStr)
        plt.xticks(ticks=errTics,labels=errLabels)
        plt.xlabel('Channel Error')
        plt.ylabel('%.1f percentile AoA0 error(º)'%(Pctl))
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+'/aoa0_vs_%s.svg'%(errType))
            

if args.map:
    lMAP =[
        tuple(case.split(':'))
        for case in args.map.split(',')
    ]
    
    for themap in lMAP:
        indErr = lErrMod.index(themap)         
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lineCfgFromAlg(lLocAlgs[nc])
            plt.plot(np.vstack((d0[:,0],d0_est[nc,indErr,:,0])),np.vstack((d0[:,1],d0_est[nc,indErr,:,1])),line+marker+color,label=caseStr)
        plt.plot(d0[:,0].T,d0[:,1].T,'ok',label='locations')
        handles, labels = plt.gca().get_legend_handles_labels()
        # labels will be the keys of the dict, handles will be values
        temp = {k:v for k,v in zip(labels, handles)}
        plt.legend(temp.values(), temp.keys(), loc='best')
        plt.xlabel('$d_{ox}$ (m)')
        plt.ylabel('$d_{oy}$ (m)')
        if args.print:
            plt.savefig(outfoldername+'/errormap_%s-%s-%s-%s.svg'%tuple(cdf))
       
if args.vso:
    lVSO =[
        tuple(case.split(':'))
        for case in args.vso.split(',')
    ]
    
    for vso in lVSO:
        indErr = lErrMod.index(vso)         
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)  
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lineCfgFromAlg(lLocAlgs[nc])
            plt.loglog(np.abs(aoa0_est[nc,indErr,:]-allUserData.AoA0.to_numpy()),location_error[nc,indErr,:],marker+color,label=caseStr)
        plt.xlabel('$\hat{aoa}$_o error (rad)')
        plt.ylabel('Location error (m)')
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+'/err_vs_aoa0_%s-%s-%s-%s.svg'%tuple(cdf))
        

if args.show:
    plt.show()
