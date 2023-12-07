#!/usr/bin/python
from progress.bar import Bar
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import scipy.optimize as opt

import numpy as np
import time
import os
import sys
import argparse
import ast

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
parser.add_argument('-S', type=str,help='S:min:max S gaussian error points with stds min-max')
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

args = parser.parse_args("--noloc --nompg -N 100 --noerror -D=16:16:16,64:64:64,128:128:128,256:256:256,512:512:512,1024:1024:1024 -G 3gpp --noerror --label newadapt --show --print --cdf=no:0:0:0,dic:256:256:256 --pcl=dic:50 --map=no:0:0:0,dic:256:256:256 --vso=no:0:0:0,dic:256:256:256".split(' '))
# args = parser.parse_args("--noloc --nompg -N 100 --noerror -D=16:16:16,64:64:64,128:128:128,256:256:256,512:512:512,1024:1024:1024 -G 3gpp --noerror --label test --show --print --cdf=no:0:0:0,dic:256:256:256 --pcl=dic:75 --map=no:0:0:0,dic:256:256:256 --vso=no:0:0:0,dic:256:256:256".split(' '))
# args = parser.parse_args("-N 1000 --nompg --noloc -G Geo:20 --noerror -D=64:64:64,256:256:256,1024:1024:1024 --label test --show --print --cdf=no:0:0:0,dic:256:256:256 --pcl=dic:80 --map=no:0:0:0,dic:256:256:256 --vso=no:0:0:0,dic:256:256:256".split(' '))
#args = parser.parse_args("-N 100 --noerror --label test --show --print".split(' '))

# numero de simulacions
Nsims=args.N if args.N else 100

# error vector modeling
if args.noerror:
    lErrMod=[('no','0','0','0')]
else:
    lErrMod=[]
if args.S:
    NS,minStd,maxStd = args.S.split(':')
    lSTD=np.logspace(minStd,maxStd,args.S)
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
       # (False,np.inf,'3P','brute'),
       # (False,np.inf,'3P','mmse'),
#       (False,np.inf,'3P','zero'),
       # (False,64,'3P','mmse'),
#       (False,64,'3P','zero'),
#       (False,np.inf,'D1','brute'),
       # (False,np.inf,'D1','mmse'),
#       (False,np.inf,'D1','zero'),
        # (False,64,'D1','mmse'),
#       (False,64,'D1','zero'),
       ]
NlocAlg =len(lLocAlgs)

if args.label:
    outfoldername="../Results/MPRayLocresults%s"%(args.label)
else:
    outfoldername="../Results/MPRayLocresults-%d-%d-%s"%(NerrMod,Nsims,mpgen)
if not os.path.isdir(outfoldername):
    os.mkdir(outfoldername)

#TODO: create command line arguments for these parameters
Xmax=100
Xmin=-100
Ymax=100
Ymin=-100

c=3e8
# if this parameter is present we get the mpg data from a known file (better for teset&debug)
if args.nompg:
    data=np.load(outfoldername+'/chanGenData.npz') 
    x=data["x"]         
    y=data["y"]
    x0=data["x0"]
    y0=data["y0"]
    aod0=np.mod( np.arctan2(y0,x0) , 2*np.pi)
    aod=data["aod"]
    aoa0=data["aoa0"]
    aoa=data["aoa"]
    c=3e8
    tau0=y0/np.sin(aod0)/c
    tdoa=data["tdoa"]
    tauE=data["tauE"]
    aod_est=data["aod_est"]
    aoa_est=data["aoa_est"]
    tau_est=data["tau_est"]
    nvalid=data["nvalid"]
    (Npath,Nsims)=x.shape
else:        
    t_start_gen=time.time()
    # TODO: this has to be modeled in a separate class
    if mpgen == 'Geo':        
        Npath=int(mpgenInfo[1])
        #generate locations and compute multipath 
        x=np.random.rand(Npath,Nsims)*(Xmax-Xmin)+Xmin
        y=np.random.rand(Npath,Nsims)*(Ymax-Ymin)+Ymin
        x0=np.random.rand(Nsims)*(Xmax-Xmin)+Xmin
        y0=np.random.rand(Nsims)*(Ymax-Ymin)+Ymin
        #angles from locations
        aod0=np.mod( np.arctan2(y0,x0) , 2*np.pi)
        aoa0=np.random.rand(Nsims)*2*np.pi #receiver angular measurement offset
        aod=np.mod( np.arctan2(y,x) , 2*np.pi)
        aoa=np.mod( np.arctan2((y-y0),(x-x0)) , 2*np.pi)
        #delays based on distance
        tau=(np.sqrt(x**2+y**2)+np.sqrt((x-x0)**2+(y-y0)**2))/c
        tau0=np.sqrt(x0**2+y0**2)/c
        tauE=np.random.rand(Nsims)*40e-9
        tdoa=tau-tau0-tauE
        nvalid=Npath*np.ones(Nsims,dtype=int)
    elif mpgen == "3gpp":        
        # TODO: introducir param de entrada para regular blargeBW, scenario, etc
        # Tamen mais tarde - Elección de escenario vía param. de entrada
        model = mpg.ThreeGPPMultipathChannelModel(scenario="UMi",bLargeBandwidthOption=True)
        x0=np.random.rand(Nsims)*(Xmax-Xmin)+Xmin
        y0=np.random.rand(Nsims)*(Ymax-Ymin)+Ymin
        tau0=np.sqrt(x0**2+y0**2)/c
        aod0=np.mod( np.arctan2(y0,x0) , 2*np.pi)
        aoa0=np.random.rand(Nsims)*2*np.pi #receiver angular measurement offset
        tauE=np.random.randn(Nsims)*40e-9
        
        txPos = (0,0,10)        
        Nclippaths = 50
        # Npath = model.maxM*np.max( model.scenarioParams.loc['N'] )                
        Npath = Nclippaths
        nvalid = np.zeros(Nsims,dtype=int)
        tdoa = np.zeros((Npath,Nsims))
        aod = np.zeros((Npath,Nsims))
        aoa = np.zeros((Npath,Nsims))
        x = np.zeros((Npath,Nsims))
        y = np.zeros((Npath,Nsims))
        losAll = np.zeros((Nsims),dtype=bool)
        bar = Bar("Generating %d 3GPP multipath channels"%(Nsims), max=Nsims)
        bar.check_tty = False
        for n in range(Nsims):
            # while nvalid[n]<4:#discard channels with too few paths
            rxPos = (x0[n],y0[n],1.5)
            plinfo,macro,clusters,subpaths = model.create_channel(txPos,rxPos)     
            losAll[n]=plinfo[0]
            # (txPos,rxPos,plinfo,clusters,subpaths)  = model.fullFitAOA(txPos,rxPos,plinfo,clusters,subpaths)
            (txPos,rxPos,plinfo,clusters,subpaths)  = model.randomFitEpctClusters(txPos,rxPos,plinfo,clusters,subpaths,Ec=.75,Es=.75,P=[0,.5,.5,0],mode3D=False)
            # txArrayAngle = 0#deg
            # rxArrayAngle = np.mod(np.pi+np.arctan2(y0[n],x0[n]),2*np.pi)*180/np.pi
            # (txPos,rxPos,plinfo,clusters,subpaths)  = model.fullDeleteBacklobes(txPos,rxPos,plinfo,clusters,subpaths,tAOD=txArrayAngle,rAOA=rxArrayAngle)
            #remove subpaths that have not been converted to first order
            subpathsDiscarded = subpaths.loc[~np.isinf(subpaths.Xs)]
            #remove weak subpaths if there are more than 50 for faster computation
            if ( subpathsDiscarded.shape[0] > Nclippaths ):            
                srtdP = subpathsDiscarded.P.sort_values(ascending=False)
                indexStrongest=srtdP.iloc[0:50].index
                subpathsDiscarded = subpathsDiscarded.loc[indexStrongest]
            nvalid[n]=subpathsDiscarded.shape[0]
            #     if nvalid[n]<4:
            #         model.dChansGenerated.pop(txPos+rxPos)
            # if not plinfo[0]:#NLOS
            #     x1stnlos = subpathsDiscarded.sort_values("TDOA").iloc[0].Xs
            #     y1stnlos = subpathsDiscarded.sort_values("TDOA").iloc[0].Ys
            #     lD = np.sqrt(x1stnlos**2+y1stnlos**2)
            #     lA = np.sqrt((x1stnlos-x0[n])**2+(y1stnlos-y0[n])**2)
            #     tau1stlos = (lD+lA)/c
            #     tauE[n] = tauE[n] - (tau1stlos-tau0[n])
            tdoa[0:nvalid[n],n] = subpathsDiscarded.TDOA-tauE[n]
            aod[0:nvalid[n],n] = np.mod( subpathsDiscarded.AOD*np.pi/180 ,2*np.pi)
            aoa[0:nvalid[n],n] = np.mod( subpathsDiscarded.AOA*np.pi/180 ,2*np.pi)
            x[0:nvalid[n],n] = subpathsDiscarded.Xs
            y[0:nvalid[n],n] = subpathsDiscarded.Ys
            bar.next()
        bar.finish()
    else:
        print("MultiPath generation method %s not recognized"%mpgen)
    
    aod_est=np.zeros((NerrMod,Npath,Nsims))
    aoa_est=np.zeros((NerrMod,Npath,Nsims))
    tau_est=np.zeros((NerrMod,Npath,Nsims))
    for nv in range(NerrMod):
        (errType,c1,c2,c3)=lErrMod[nv]
        if errType=='no':
            aod_est[nv,:,:]=np.mod(aod,2*np.pi)
            aoa_est[nv,:,:]=np.mod(aoa-aoa0,2*np.pi)
            tau_est[nv,:,:]=tdoa
        elif errType=='std': 
            aod_est[nv,:,:]=np.mod(aod+float(c1)*2*np.pi*np.random.randn(Npath,Nsims),2*np.pi)
            aoa_est[nv,:,:]=np.mod(aoa-aoa0+float(c2)*2*np.pi*np.random.randn(Npath,Nsims),2*np.pi)
            tau_est[nv,:,:]=tdoa+float(c3)*320e-9*np.random.randn(Npath,Nsims)
        elif errType=='dic':
            if c1=='inf':
                aod_est[nv,:,:]=np.mod(aod,2*np.pi)
            else:
                aod_est[nv,:,:]=np.mod(np.round(aod*int(c1)/2/np.pi)*2*np.pi/int(c1),2*np.pi)
            if c2=='inf':
                aoa_est[nv,:,:]=np.mod(aoa-aoa0,2*np.pi)
            else:
                aoa_est[nv,:,:]=np.mod(np.round((aoa-aoa0)*int(c2)/2/np.pi)*2*np.pi/int(c2),2*np.pi)
            if c3=='inf':
                tau_est[nv,:,:]=tdoa
            else:
                Ts=1.0/400e6 #2.5ns
                Ds=320e-9 #Ts*128 FIR filter
                tau_est[nv,:,:]=np.round(tdoa*int(c3)/Ds)*Ds/int(c3)
        else:
            print("Multipath estimation error model %s to be written"%errType)
    if not args.nosave: 
        np.savez(outfoldername+'/chanGenData.npz',
                 x=x,
                 y=y,
                 x0=x0,
                 y0=y0,
                 aod=aod,
                 aoa0=aoa0,
                 aoa=aoa,
                 tdoa=tdoa,
                 tauE=tauE,
                 aod_est=aod_est,
                 aoa_est=aoa_est,
                 tau_est=tau_est,
                 nvalid=nvalid)
    print("Total Multipath Estimation Time:%s seconds"%(time.time()-t_start_gen))

if args.noloc: 
    data=np.load(outfoldername+'/locEstData.npz') 
    aoa0_est=data["aoa0_est"]
    x0_est=data["x0_est"]
    y0_est=data["y0_est"]
    tauE_est=data["tauE_est"]
    x_est=data["x_est"]
    y_est=data["y_est"]
    run_time=data["run_time"]    
    loc=MultipathLocationEstimator.MultipathLocationEstimator(Npoint=100,RootMethod='lm')    
else:        
    t_start_loc=time.time() 
    aoa0_est=np.zeros((NlocAlg,NerrMod,Nsims))
    x0_est=np.zeros((NlocAlg,NerrMod,Nsims))
    y0_est=np.zeros((NlocAlg,NerrMod,Nsims))
    tauE_est=np.zeros((NlocAlg,NerrMod,Nsims))
    x_est=np.zeros((NlocAlg,NerrMod,Npath,Nsims))
    y_est=np.zeros((NlocAlg,NerrMod,Npath,Nsims))
    run_time=np.zeros((NlocAlg,NerrMod))
    
    loc=MultipathLocationEstimator.MultipathLocationEstimator(Npoint=100,RootMethod='lm')
    
    for nc in range(NlocAlg):
        (aoa0Apriori,aoa0Quant,grouping,optimMthd)=lLocAlgs[nc]
        for nv in range(NerrMod):
            bar = Bar("Case aoa known:%s aoa0quant: %s grouping:%s optimMthd:%s err=%s"%(aoa0Apriori,aoa0Quant,grouping,optimMthd,lErrMod[nv]), max=Nsims)
            bar.check_tty = False        
            t_start_point = time.time()
            for ns in range(Nsims):
                if aoa0Apriori:
                    if not np.isinf(aoa0Quant):
                        aoa0_est[nc,nv,ns]=np.round(aoa0[ns]*aoa0Quant/(np.pi*2))*2*np.pi/aoa0Quant        
                    else:
                        aoa0_est[nc,nv,ns]=aoa0[ns]
                    (x0_est[nc,nv,ns],y0_est[nc,nv,ns],tauE_est[nc,nv,ns],x_est[nc,nv,0:nvalid[ns],ns],y_est[nc,nv,0:nvalid[ns],ns])=loc.computeAllPaths(aod_est[nv,0:nvalid[ns],ns],aoa_est[nv,0:nvalid[ns],ns],tau_est[nv,0:nvalid[ns],ns],aoa0_est[nc,nv,ns])
                else:
                #TODO make changes in location estimator and get rid of these ifs
                    if not np.isinf(aoa0Quant):
                        aoa0_hint=np.round(aoa0[ns]*aoa0Quant/(np.pi*2))*2*np.pi/aoa0Quant
                    else:
                        aoa0_hint=None
                    group_m= '3path' if grouping=='3P' else 'drop1'
                    aoa0_m= 'fsolve' if (optimMthd=='mmse')or(grouping=='D1') else optimMthd
                    (aoa0_est[nc,nv,ns],x0_est[nc,nv,ns],y0_est[nc,nv,ns],tauE_est[nc,nv,ns],x_est[nc,nv,0:nvalid[ns],ns],y_est[nc,nv,0:nvalid[ns],ns],_)= loc.computeAllLocationsFromPaths(aod_est[nv,0:nvalid[ns],ns],aoa_est[nv,0:nvalid[ns],ns],tau_est[nv,0:nvalid[ns],ns],AoA0_method=aoa0_m,group_method=group_m,hint_AoA0=aoa0_hint)
                bar.next()
            bar.finish()
            run_time[nc,nv] = time.time() - t_start_point
    if not args.nosave: 
        np.savez(outfoldername+'/locEstData.npz',
                aoa0_est=aoa0_est,
                x0_est=x0_est,
                y0_est=y0_est,
                tauE_est=tauE_est,
                x_est=x_est,
                y_est=y_est,
                run_time=run_time)
    print("Total Location Time:%s seconds"%(time.time()-t_start_loc))

location_error=np.sqrt(np.abs(x0-x0_est)**2+np.abs(y0-y0_est)**2)
mapping_error=np.sqrt(np.abs(x-x_est)**2+np.abs(y-y_est)**2) 
# mapping_error[:,:,x==0]=np.inf#fix better
tauE_err = np.abs(tauE_est+tauE)
aoa0_err=np.abs(aoa0-aoa0_est)*180/np.pi
x0_dumb=np.random.rand(Nsims)*(Xmax-Xmin)+Xmin
y0_dumb=np.random.rand(Nsims)*(Ymax-Ymin)+Ymin
error_dumb=np.sqrt(np.abs(x0-x0_dumb)**2+np.abs(y0-y0_dumb)**2)
x_dumb=np.random.rand(Npath,Nsims)*(Xmax-Xmin)+Xmin
y_dumb=np.random.rand(Npath,Nsims)*(Ymax-Ymin)+Ymin
map_dumb=np.sqrt(np.abs(x_dumb-x)**2+np.abs(y_dumb-y)**2).reshape((-1))
plt.close('all')

def lineCfgFromAlg(algCfg):
    aoa0Apriori,aoa0Quant,grouping,optimMthd=algCfg
    if aoa0Apriori:
        caseStr="LS Location" if np.isinf(aoa0Quant) else "aoa0 quantized sensor"
        color='b'
        marker='o' if np.isinf(aoa0Quant) else '*'
        line=':'
    else:
        caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if aoa0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
        line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
        marker='x' if aoa0Quant else 's'
        color='r' if optimMthd=='brute' else 'g'        
    return(caseStr,line,marker,color)

fig_ctr=0
if args.cdf:
    lCDF =[
        tuple(case.split(':'))
        for case in args.cdf.split(',')
    ]
    for cdf in lCDF:
        indErr = lErrMod.index(cdf)
        
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lineCfgFromAlg(lLocAlgs[nc])
            plt.semilogx(np.percentile(location_error[nc,indErr,~np.isnan(location_error[nc,indErr,:])],np.linspace(0,100,21)),np.linspace(0,1,21),line+marker+color,label=caseStr)        
        plt.semilogx(np.percentile(error_dumb,np.linspace(0,100,21)),np.linspace(0,1,21),':k',label="random guess")
        Ts = loc.getTParamToLoc(x0,y0,tau0+tauE,aoa0,x,y,['dDAoA',],['dx0','dy0'])
        varaoaDist=np.var(np.minimum(np.mod(aoa-aoa0-aoa_est[indErr,:,:],np.pi*2),2*np.pi-np.mod(aoa-aoa0-aoa_est[indErr,:,:],np.pi*2))) * (Npath*Nsims)/np.sum(nvalid)
        M=np.matmul(Ts.transpose([2,1,0]),Ts.transpose([2,0,1]))
        errorCRLBnormalized = np.array([np.sqrt(np.trace(np.linalg.lstsq(M[n,:,:],np.eye(2),rcond=None)[0])) for n in range(M.shape[0])])
        plt.semilogx(np.percentile(np.sqrt(varaoaDist)*errorCRLBnormalized,np.linspace(0,100,21)),np.linspace(0,1,21),'--k',label="approx. CRLB")    
        # if lErrMod[indErr][0]=='dic':
        #     Nrant=float(lErrMod[indErr][2])
            # plt.semilogx(np.percentile(np.sqrt((np.pi/Nrant)**2/12)*errorCRLBnormalized,np.linspace(0,100,21)),np.linspace(0,1,21),'--k',label="approx. CRLB")    
        plt.xlabel('Location error(m)')
        plt.ylabel('C.D.F.')
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+('/cdflocerr_%s-%s-%s-%s.eps'%tuple(cdf)))
            
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lineCfgFromAlg(lLocAlgs[nc])
            mapping_error_data_valid = mapping_error[nc,indErr,(~np.isnan(mapping_error[nc,indErr,:]))&(~np.isinf(mapping_error[nc,indErr,:]))]
            plt.semilogx(np.percentile(mapping_error_data_valid,np.linspace(0,100,21)),np.linspace(0,1,21),line+marker+color,label=caseStr)
        plt.semilogx(np.percentile(map_dumb,np.linspace(0,100,21)),np.linspace(0,1,21),':k',label="random guess")
        if Npath<=50:
            Ts = loc.getTParamToLoc(x0,y0,tauE+tau0,aoa0,x,y,['dDAoA'],['dx','dy'])            
            M=np.matmul(Ts.transpose([2,1,0]),Ts.transpose([2,0,1]))
            #(1/Npath)*
            errorCRLBnormalized = np.array([np.sqrt(np.trace(np.linalg.lstsq(M[n,:,:],np.eye(2*Npath),rcond=None)[0])) for n in range(M.shape[0])])
            varaoaDist=np.var(np.minimum(np.mod(aoa-aoa0-aoa_est[indErr,:,:],np.pi*2),2*np.pi-np.mod(aoa-aoa0-aoa_est[indErr,:,:],np.pi*2))) * (Npath*Nsims)/np.sum(nvalid)
            plt.semilogx(np.percentile(np.sqrt(varaoaDist)*errorCRLBnormalized,np.linspace(0,100,21)),np.linspace(0,1,21),'--k',label="$\\sim$ CRLB")    
            # lNant=np.array([float(x[2]) for x in lErrMod if x[0]=='dic' and x[1]=='inf' and x[3]=='inf']) 
            # plt.semilogy(errTics,np.percentile(errorCRLBnormalized,80)*np.sqrt((np.pi/lNant)**2/12),'--k',label="approx. CRLB")       
        plt.xlabel('Mapping error(m)')
        plt.ylabel('C.D.F.')
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+('/cdfmaperr_%s-%s-%s-%s.eps'%tuple(cdf)))


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
        plt.semilogy(errTics,np.ones_like(errTics)*np.percentile(error_dumb,80),':k',label="random guess")
        Ts = loc.getTParamToLoc(x0,y0,tauE+tau0,aoa0,x,y,['dDAoA'],['dx0','dy0'])
        M=np.matmul(Ts.transpose([2,1,0]),Ts.transpose([2,0,1]))
        errorCRLBnormalized = np.array([np.sqrt(np.trace(np.linalg.lstsq(M[n,:,:],np.eye(2),rcond=None)[0])) for n in range(M.shape[0])])
        varaoaDist=np.var(np.minimum(np.mod(aoa-aoa0-aoa_est[errCaseMask,:,:],np.pi*2),2*np.pi-np.mod(aoa-aoa0-aoa_est[errCaseMask,:,:],np.pi*2)),axis=(1,2)) * (Npath*Nsims)/np.sum(nvalid)
        plt.semilogy(errTics,np.percentile(errorCRLBnormalized,80)*np.sqrt(varaoaDist),'--k',label="$\\sim$ CRLB")      
        # lNant=np.array([float(x[2]) for x in lErrMod if x[0]=='dic' and x[1]=='inf' and x[3]=='inf']) 
        # plt.semilogy(errTics,np.percentile(errorCRLBnormalized,80)*np.sqrt((np.pi/lNant)**2/12),'--k',label="approx. CRLB")       
        plt.xticks(ticks=errTics,labels=errLabels)
        plt.xlabel(errTitle)
        plt.ylabel('%.1f percentile location error(m)'%(Pctl))
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+'/err_vs_%s.eps'%(errType))
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lineCfgFromAlg(lLocAlgs[nc])
            plt.semilogy(np.arange(len(errLabels)),np.percentile(mapping_error[nc,errCaseMask,:,:],Pctl,axis=(1,2)),line+marker+color,label=caseStr)
        plt.semilogy(errTics,np.ones_like(errTics)*np.percentile(map_dumb,80),':k',label="random guess")
        if Npath<=50:
            Ts = loc.getTParamToLoc(x0,y0,tauE+tau0,aoa0,x,y,['dDAoA'],['dx','dy'])            
            M=np.matmul(Ts.transpose([2,1,0]),Ts.transpose([2,0,1]))
            #(1/Npath)*
            errorCRLBnormalized = np.array([np.sqrt(np.trace(np.linalg.lstsq(M[n,:,:],np.eye(2*Npath),rcond=None)[0])) for n in range(M.shape[0])])
            varaoaDist=np.var(np.minimum(np.mod(aoa-aoa0-aoa_est[errCaseMask,:,:],np.pi*2),2*np.pi-np.mod(aoa-aoa0-aoa_est[errCaseMask,:,:],np.pi*2)),axis=(1,2)) * (Npath*Nsims)/np.sum(nvalid)
            plt.semilogy(errTics,np.percentile(errorCRLBnormalized,80)*np.sqrt(varaoaDist),'--k',label="$\\sim$ CRLB")      
            # lNant=np.array([float(x[2]) for x in lErrMod if x[0]=='dic' and x[1]=='inf' and x[3]=='inf']) 
            # plt.semilogy(errTics,np.percentile(errorCRLBnormalized,80)*np.sqrt((np.pi/lNant)**2/12),'--k',label="approx. CRLB")       
        plt.xticks(ticks=errTics,labels=errLabels)
        plt.xlabel(errTitle)
        plt.ylabel('%.1f percentile mapping error(m)'%(Pctl))
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+'/map_vs_%s.eps'%(errType))
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
            plt.savefig(outfoldername+'/tau_vs_%s.eps'%(errType))
        fig_ctr=fig_ctr+1
        plt.figure(fig_ctr)
        for nc in range(NlocAlg):
            caseStr,line,marker,color=lineCfgFromAlg(lLocAlgs[nc])
            plt.semilogy(np.arange(len(errLabels)),np.percentile(aoa0_err[nc,errCaseMask,:],Pctl,axis=(1)),line+marker+color,label=caseStr)
        plt.xticks(ticks=errTics,labels=errLabels)
        plt.xlabel('Channel Error')
        plt.ylabel('%.1f percentile AOA0 error(º)'%(Pctl))
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+'/aoa0_vs_%s.eps'%(errType))
            

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
            plt.plot(np.vstack((x0,x0_est[nc,indErr,:])),np.vstack((y0,y0_est[nc,indErr,:])),line+marker+color,label=caseStr)
        plt.plot(x0.T,y0.T,'ok',label='locations')
        handles, labels = plt.gca().get_legend_handles_labels()
        # labels will be the keys of the dict, handles will be values
        temp = {k:v for k,v in zip(labels, handles)}
        plt.legend(temp.values(), temp.keys(), loc='best')
        plt.xlabel('$d_{ox}$ (m)')
        plt.ylabel('$d_{oy}$ (m)')
        if args.print:
            plt.savefig(outfoldername+'/errormap_%s-%s-%s-%s.eps'%tuple(cdf))
       
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
            plt.loglog(np.abs(aoa0_est[nc,indErr,:]-aoa0[:]),location_error[nc,indErr,:],marker+color,label=caseStr)
        plt.xlabel('$\hat{aoa}$_o error (rad)')
        plt.ylabel('Location error (m)')
        plt.legend()
        if args.print:
            plt.savefig(outfoldername+'/err_vs_aoa0_%s-%s-%s-%s.eps'%tuple(cdf))
        

if args.show:
    plt.show()
