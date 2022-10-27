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

import MultipathLocationEstimator
#TODO: make the script below the __main__() of a class that can be imported by other python programs
parser = argparse.ArgumentParser(description='Multipath Location Estimation Simulator')
#parameters that affect number of simulations
parser.add_argument('-N', type=int,help='No. simulated channels')
#parameters that affect error
parser.add_argument('--noerror', help='Add zero error case', action='store_true')
parser.add_argument('-S', type=int,help='Add S normalized error Std. points')
parser.add_argument('--minstd', help='Minimum error std in dB')
parser.add_argument('--maxstd', help='Maximum error std in dB')
parser.add_argument('-D', type=str,help='Add dictionary-based error models, comma separated')
#parameters that affect multipath generator
parser.add_argument('-G', type=int,help='Type of generator')
parser.add_argument('--npathgeo', type=int,help='No. paths per channel (Geo only)')
#parameters that affect location algorithms
parser.add_argument('--algs', type=str,help='comma-separated list of algorithms')
#parameters that affect workflow
parser.add_argument('--label', type=str,help='text label for stored results')
parser.add_argument('--nosave', help='Do not save simulation data to new results file', action='store_true')
parser.add_argument('--nompg',help='Do not perform multipath generation, load prior results file', action='store_true')
parser.add_argument('--noloc',help='Do not perform location estimation, load prior results file', action='store_true')
parser.add_argument('--show', help='Open plot figures in window', action='store_true')
parser.add_argument('--print', help='Save plot files in eps to results folder', action='store_true')

args = parser.parse_args("-N 100 -S 7 -D inf:16:inf,inf:64:inf,inf:256:inf,inf:1024:inf,inf:4096:inf,16:inf:inf,64:inf:inf,256:inf:inf,1024:inf:inf,4096:inf:inf,inf:inf:16,inf:inf:64,inf:inf:256,inf:inf:1024,inf:inf:4096 --noerror --label test --show --print".split(' '))

#args = parser.parse_args("-N 100 --noerror --label test --show --print".split(' '))

Nsims=args.N if args.N else 100
if args.noerror:
    lErrMod=[('no',0,0,0)]
else:
    lSTD=[]
if args.S:
    minStd = args.minstd if args.minstd else -7
    maxStd = args.maxstd if args.maxstd else -1
    stepStd=(maxStd-minStd)/(args.S-1)
    lSTD=np.logspace(minStd,maxStd,args.S) 
    lErrMod = lErrMod+[('std',x,x,x) for x in lSTD]
if args.D:
    lDicSizes=[tuple(['dic']+[x for x in y.split(':')]) for y in args.D.split(',') ]
    lErrMod = lErrMod+lDicSizes    
#TODO add ifs etc. to make this selectable with OMP
NerrMod=len(lErrMod)
mpgen = args.G if args.G else 'Geo'

if args.algs:
    lCases=[
            tuple(case.split(':'))
            for case in args.algs.split(',')
            ]
    #TODO define more elegant syntax for cases and better parser that converts bool properly
else:
    lCases=[#a opriori phi0, quantized phi0, grouping method, optimization method
       (True,False,'',''),
       (True,True,'',''),
       (False,False,'3P','brute'),
       (False,False,'3P','mmse'),
#       (False,False,'3P','zero'),
       (False,True,'3P','mmse'),
#       (False,True,'3P','zero'),
#       (False,False,'D1','brute'),
       (False,False,'D1','mmse'),
#       (False,False,'D1','zero'),
       (False,True,'D1','mmse'),
#       (False,True,'D1','zero'),
       ]
Ncases =len(lCases)

if args.label:
    outfoldername="./MPRayLocresults%s-%d-%d-%s"%(args.label,NerrMod,Nsims,mpgen)
else:
    outfoldername="./MPRayLocresults-%d-%d-%s"%(NerrMod,Nsims,mpgen)
if not os.path.isdir(outfoldername):
    os.mkdir(outfoldername)

#TODO: create command line arguments for these parameters
Xmax=50
Xmin=-50
Ymax=50
Ymin=-50

phi0GyrQuant=2*np.pi/64

if args.nompg:
    data=np.load(outfoldername+'/chanGenData.npz') 
    x=data["x"]         
    y=data["y"]
    x0=data["x0"]
    y0=data["y0"]
    theta0=np.mod( np.arctan(y0/x0)+np.pi*(x0<0) , 2*np.pi)
    theta=data["theta"]
    phi0=data["phi0"]
    phi=data["phi"]
    c=3e8
    tau0=y0/np.sin(theta0)/c
    tau=data["tau"]
    tauE=data["tauE"]
    theta_est=data["theta_est"]
    phi_est=data["phi_est"]
    tau_est=data["tau_est"]
    (Npath,Nsims)=x.shape
else:        
    t_start_gen=time.time()
    if mpgen == 'Geo':
        if args.npathgeo:
            Npath=args.npathgeo
        else:
            Npath=20
        #generate locations and compute multipath 
        x=np.random.rand(Npath,Nsims)*(Xmax-Xmin)+Xmin
        y=np.random.rand(Npath,Nsims)*(Ymax-Ymin)+Ymin
        x0=np.random.rand(1,Nsims)*(Xmax-Xmin)+Xmin
        y0=np.random.rand(1,Nsims)*(Ymax-Ymin)+Ymin
        #angles from locations
        theta0=np.mod( np.arctan(y0/x0)+np.pi*(x0<0) , 2*np.pi)
        theta=np.mod( np.arctan(y/x)+np.pi*(x<0) , 2*np.pi)
        phi0=np.random.rand(1,Nsims)*2*np.pi #receiver angular measurement offset
        phi=np.mod(np.pi - (np.arctan((y-y0)/(x0-x))+np.pi*((x0-x)<0)) , 2*np.pi)
        #delays based on distance
        c=3e8
        tau=(np.abs(y/np.sin(theta))+np.abs((y-y0)/np.sin(phi)))/c
        tau0=y0/np.sin(theta0)/c
        tauE=tau0+np.random.randn(1,Nsims)*40e-9
    elif mpgen == "3gpp":
        #TBW
        print("MultiPath generation method %s to be written"%mpgen)
    else:
        print("MultiPath generation method %s not recognized"%mpgen)
    
    theta_est=np.zeros((NerrMod,Npath,Nsims))
    phi_est=np.zeros((NerrMod,Npath,Nsims))
    tau_est=np.zeros((NerrMod,Npath,Nsims))
    for nv in range(NerrMod):
        (errType,c1,c2,c3)=lErrMod[nv]
        if errType=='no':
            theta_est[nv,:,:]=np.mod(theta,2*np.pi)
            phi_est[nv,:,:]=np.mod(phi-phi0,2*np.pi)
            tau_est[nv,:,:]=tau-tauE
        elif errType=='std': 
            theta_est[nv,:,:]=np.mod(theta+c1*2*np.pi*np.random.rand(Npath,Nsims),2*np.pi)
            phi_est[nv,:,:]=np.mod(phi-phi0+c2*2*np.pi*np.random.rand(Npath,Nsims),2*np.pi)
            tau_est[nv,:,:]=tau-tauE+c3*320e-9*np.random.rand(Npath,Nsims)
        elif errType=='dic':
            if c1=='inf':
                theta_est[nv,:,:]=np.mod(theta,2*np.pi)
            else:
                theta_est[nv,:,:]=np.mod(np.round(theta*int(c1)/2/np.pi)*2*np.pi/int(c1),2*np.pi)
            if c2=='inf':
                phi_est[nv,:,:]=np.mod(phi-phi0,2*np.pi)
            else:
                phi_est[nv,:,:]=np.mod(np.round((phi-phi0)*int(c2)/2/np.pi)*2*np.pi/int(c2),2*np.pi)
            if c3=='inf':
                tau_est[nv,:,:]=tau-tauE
            else:
                Ts=1.0/400e6#2.5ns
                Ds=320e-9#Ts*128 FIR filter
                tau_est[nv,:,:]=np.round((tau-tauE)*int(c3)/Ds)*Ds/int(c3)
        else:
            print("Multipath estimation error model %s to be written"%errType)
    if not args.nosave: 
        np.savez(outfoldername+'/chanGenData.npz',
                 x=x,
                 y=y,
                 x0=x0,
                 y0=y0,
                 theta=theta,
                 phi0=phi0,
                 phi=phi,
                 tau=tau,
                 tauE=tauE,
                 theta_est=theta_est,
                 phi_est=phi_est,
                 tau_est=tau_est)
    print("Total Generation Time:%s seconds"%(time.time()-t_start_gen))

if args.noloc: 
    data=np.load(outfoldername+'/locEstData.npz') 
    phi0_est=data["phi0_est"]
    x0_est=data["x0_est"]
    y0_est=data["y0_est"]
    tauE_est=data["tauE_est"]
    x_est=data["x_est"]
    y_est=data["y_est"]
    run_time=data["run_time"]
else:        
    t_start_loc=time.time() 
    phi0_est=np.zeros((Ncases,NerrMod,Nsims))
    x0_est=np.zeros((Ncases,NerrMod,Nsims))
    y0_est=np.zeros((Ncases,NerrMod,Nsims))
    tauE_est=np.zeros((Ncases,NerrMod,Nsims))
    x_est=np.zeros((Ncases,NerrMod,Npath,Nsims))
    y_est=np.zeros((Ncases,NerrMod,Npath,Nsims))
    run_time=np.zeros((Ncases,NerrMod))
    
    loc=MultipathLocationEstimator.MultipathLocationEstimator(Npoint=100,RootMethod='lm')
    
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        for nv in range(NerrMod):
            bar = Bar("Case phi known:%s phi0quant: %s grouping:%s optimMthd:%s err=%s"%(phi0Apriori,phi0Quant,grouping,optimMthd,lErrMod[nv]), max=Nsims)
            bar.check_tty = False        
            t_start_point = time.time()
            for ns in range(Nsims):
                if phi0Apriori:
                    if phi0Quant:
                        phi0_est[nc,nv,ns]=np.round(phi0[:,ns]/phi0GyrQuant)*phi0GyrQuant        
                    else:
                        phi0_est[nc,nv,ns]=phi0[:,ns]
                    (x0_est[nc,nv,ns],y0_est[nc,nv,ns],tauE_est[nc,nv,ns],x_est[nc,nv,:,ns],y_est[nc,nv,:,ns])=loc.computeAllPaths(theta_est[nv,:,ns],phi_est[nv,:,ns],tau_est[nv,:,ns],phi0_est[nc,nv,ns])
                else:
                #TODO make changes in location estimator and get rid of these ifs
                    if phi0Quant:
                        phi0_hint=np.round(phi0[:,ns]/phi0GyrQuant)*phi0GyrQuant        
                    else:
                        phi0_hint=None
                    group_m= '3path' if grouping=='3P' else 'drop1'
                    phi0_m= 'fsolve' if (optimMthd=='mmse')or(grouping=='D1') else optimMthd
                    (phi0_est[nc,nv,ns],x0_est[nc,nv,ns],y0_est[nc,nv,ns],tauE_est[nc,nv,ns],x_est[nc,nv,:,ns],y_est[nc,nv,:,ns],_)= loc.computeAllLocationsFromPaths(theta_est[nv,:,ns],phi_est[nv,:,ns],tau_est[nv,:,ns],phi0_method=phi0_m,group_method=group_m,hint_phi0=phi0_hint)
                bar.next()
            bar.finish()
            run_time[nc,nv] = time.time() - t_start_point
    if not args.nosave: 
        np.savez(outfoldername+'/locEstData.npz',
                phi0_est=phi0_est,
                x0_est=x0_est,
                y0_est=y0_est,
                tauE_est=tauE_est,
                x_est=x_est,
                y_est=y_est,
                run_time=run_time)
    print("Total Generation Time:%s seconds"%(time.time()-t_start_gen))

location_error=np.sqrt(np.abs(x0-x0_est)**2+np.abs(y0-y0_est)**2)
mapping_error=np.sqrt(np.abs(x-x_est)**2+np.abs(y-y_est)**2) 
tauE_err = np.abs(tauE_est+(tauE-tau0))
phi0_err=np.abs(phi0-phi0_est)*180/np.pi
x0_dumb=np.random.rand(1,Nsims)*(Xmax-Xmin)+Xmin
y0_dumb=np.random.rand(1,Nsims)*(Ymax-Ymin)+Ymin
error_dumb=np.sqrt(np.abs(x0-x0_dumb)**2+np.abs(y0-y0_dumb)**2).reshape((-1))
x_dumb=np.random.rand(Npath,Nsims)*(Xmax-Xmin)+Xmin
y_dumb=np.random.rand(Npath,Nsims)*(Ymax-Ymin)+Ymin
map_dumb=np.sqrt(np.abs(x_dumb-x)**2+np.abs(y_dumb-y)**2).reshape((-1))
plt.close('all')

fig_ctr=0
if args.noerror:
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        plt.semilogx(np.percentile(location_error[nc,0,~np.isnan(location_error[nc,0,:])],np.linspace(0,100,21)),np.linspace(0,1,21),line+marker+color,label=caseStr)        
    plt.semilogx(np.percentile(error_dumb,np.linspace(0,100,21)),np.linspace(0,1,21),':k',label="random guess")
    plt.xlabel('Location error(m)')
    plt.ylabel('C.D.F.')
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/cdflocerr_noerr.eps')
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        plt.semilogx(np.percentile(mapping_error[nc,0,~np.isnan(mapping_error[nc,0,:])],np.linspace(0,100,21)),np.linspace(0,1,21),line+marker+color,label=caseStr)
    plt.semilogx(np.percentile(map_dumb,np.linspace(0,100,21)),np.linspace(0,1,21),':k',label="random guess")
    plt.xlabel('Mapping error(m)')
    plt.ylabel('C.D.F.')
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/cdfmaperr_noerr.eps')

if args.S:    
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    P1pos=np.argmax([x[1]==1e-3 and x[0]=='std' for x in lErrMod])
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        plt.semilogx(np.percentile(location_error[nc,P1pos,~np.isnan(location_error[nc,P1pos,:])],np.linspace(0,100,21)),np.linspace(0,1,21),line+marker+color,label=caseStr)
    plt.semilogx(np.percentile(error_dumb,np.linspace(0,100,21)),np.linspace(0,1,21),':k',label="random guess")
    plt.xlabel('Location error(m)')
    plt.ylabel('C.D.F.')
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/cdflocerr_1Perr.eps')
        
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    P1pos=np.argmax([x[1]==1e-3 and x[0]=='std' for x in lErrMod])
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        plt.semilogx(np.percentile(mapping_error[nc,P1pos,~np.isnan(mapping_error[nc,P1pos,:,:])],np.linspace(0,100,21)),np.linspace(0,1,21),line+marker+color,label=caseStr)
    plt.semilogx(np.percentile(map_dumb,np.linspace(0,100,21)),np.linspace(0,1,21),':k',label="random guess")
    plt.xlabel('Mapping error(m)')
    plt.ylabel('C.D.F.')
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/cdfmaperr_1Perr.eps')
        
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    stdCaseMask=[x[0]=='std' for x in lErrMod]
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        plt.loglog(lSTD,np.percentile(location_error[nc,stdCaseMask,:],80,axis=1),line+marker+color,label=caseStr)
    plt.loglog(lSTD,np.ones_like(lSTD)*np.percentile(error_dumb,80),':k',label="random guess")
    plt.xlabel('Error std.')
    plt.ylabel('80\%tile location error(m)')
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/err_vs_std.eps')
        
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    stdCaseMask=[x[0]=='std' for x in lErrMod]
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        plt.loglog(lSTD,np.percentile(mapping_error[nc,stdCaseMask,:,:],80,axis=(1,2)),line+marker+color,label=caseStr)
    plt.loglog(lSTD,np.ones_like(lSTD)*np.percentile(map_dumb,80),':k',label="random guess")
    plt.xlabel('Error std.')
    plt.ylabel('80\%tile mapping error(m)')
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/merr_vs_std.eps')
        
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    for nc in range(Ncases):
        if nc in [1,2,6]:
            (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
            if phi0Apriori:
                caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
                color='r'
                marker='*' if phi0Quant else 'o'
                line=':'
            else:
                caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
                line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
                marker='x' if phi0Quant else 's'
                color='b' if optimMthd=='brute' else 'g'
            plt.plot(np.vstack((x0[0,:],x0_est[nc,P1pos,:])),np.vstack((y0[0,:],y0_est[nc,P1pos,:])),line+marker+color,label=caseStr)
    plt.plot(x0.T,y0.T,'ok',label='locations')
    handles, labels = plt.gca().get_legend_handles_labels()
    # labels will be the keys of the dict, handles will be values
    temp = {k:v for k,v in zip(labels, handles)}
    plt.legend(temp.values(), temp.keys(), loc='best')
    plt.xlabel('$d_{ox}$ (m)')
    plt.ylabel('$d_{oy}$ (m)')
    if args.print:
        plt.savefig(outfoldername+'/errormap.eps')
        
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    for nc in range(Ncases):
        if nc in [1,2,6]:
            (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
            if phi0Apriori:
                caseStr="$\\sigma=0$ phi0 quantized sensor" if phi0Quant else "phi0 known"
                color='r'
                marker='*' if phi0Quant else 'o'
                line=':'
            else:
                caseStr="$\\sigma=0$ %s - %s %s"%(grouping,optimMthd,'Q. hint' if ( (optimMthd != 'brute') and phi0Quant ) else '')
                line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
                marker='x' if phi0Quant else 's'
                color='b' if optimMthd=='brute' else 'g'
            plt.loglog(np.abs(phi0_est[nc,0,:]-phi0[0,:]),location_error[nc,0,:],marker+color,label=caseStr)
    for nc in range(Ncases):
        if nc in [2,6]:
            (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
            if phi0Apriori:
                caseStr="$phi0 quantized sensor" if phi0Quant else "phi0 known"
                color='r'
                marker='*' if phi0Quant else 'o'
                line=':'
            else:
                caseStr="$\\sigma=.01$ %s - %s %s"%(grouping,optimMthd,'Q. hint' if ( (optimMthd != 'brute') and phi0Quant ) else '')
                line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
                marker='+' if phi0Quant else 'd'
                color='b' if optimMthd=='brute' else 'g'
            plt.loglog(np.abs(phi0_est[nc,P1pos,:]-phi0[0,:]),location_error[nc,P1pos,:],marker+color,label=caseStr)
    plt.xlabel('$\hat{\phi_o}$ error (rad)')
    plt.ylabel('Location error (m)')
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/err_vs_phi0.eps')
        
if args.D:
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    K2pos=np.argmax([x[1]=='256' and x[0]=='dic' for x in lErrMod])
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        plt.semilogx(np.percentile(location_error[nc,K2pos,~np.isnan(location_error[nc,K2pos,:])],np.linspace(0,100,21)),np.linspace(0,1,21),line+marker+color,label=caseStr)
    plt.semilogx(np.percentile(error_dumb,np.linspace(0,100,21)),np.linspace(0,1,21),':k',label="random guess")
    plt.xlabel('Location error(m)')
    plt.ylabel('C.D.F.')
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/cdflocerr_Kt256.eps')
    
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    dicCaseMask=[x[0]=='dic'  and x[2]=='inf' and x[3]=='inf' for x in lErrMod]
    lNant=np.array([float(x[1]) for x in lErrMod if x[0]=='dic' and x[2]=='inf' and x[3]=='inf'])
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        aux_err=np.zeros_like(lNant)
        subarray=location_error[nc,dicCaseMask,:]
        for nsub in range(lNant.size):
            aux_err[nsub]=np.percentile(subarray[nsub,~np.isnan(subarray[nsub,:])],80)
        plt.semilogy(np.log2(lNant),aux_err,line+marker+color,label=caseStr)
    plt.semilogy(np.log2(lNant),np.ones_like(lNant)*np.percentile(error_dumb,80),':k',label="random guess")
    plt.xlabel('$K_{\\theta}$')
    plt.ylabel('80\%tile location error(m)')
    plt.xticks(ticks=np.log2(lNant),labels=['$%d$'%x for x in lNant])
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/err_vs_ntant.eps')
        
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    dicCaseMask=[x[0]=='dic'  and x[2]=='inf' and x[3]=='inf' for x in lErrMod]
    lNant=np.array([float(x[1]) for x in lErrMod if x[0]=='dic' and x[2]=='inf' and x[3]=='inf'])
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        aux_err=np.zeros_like(lNant)
        subarray=mapping_error[nc,dicCaseMask,:,:]
        for nsub in range(lNant.size):
            aux_err[nsub]=np.percentile(subarray[nsub,~np.isnan(subarray[nsub,:])],80)
        plt.semilogy(np.log2(lNant),aux_err,line+marker+color,label=caseStr)
    plt.semilogy(np.log2(lNant),np.ones_like(lNant)*np.percentile(map_dumb,80),':k',label="random guess")
    plt.xlabel('$K_{\\theta}$')
    plt.ylabel('80\%tile mapping error(m)')
    plt.xticks(ticks=np.log2(lNant),labels=['$%d$'%x for x in lNant])
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/loc_vs_ntant.eps')
        
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    dicCaseMask=[x[0]=='dic'  and x[2]=='inf' and x[3]=='inf' for x in lErrMod]
    lNant=np.array([float(x[1]) for x in lErrMod if x[0]=='dic' and x[2]=='inf' and x[3]=='inf'])
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        aux_err=np.zeros_like(lNant)
        subarray=1e9*tauE_err[nc,dicCaseMask,:]
        for nsub in range(lNant.size):
            aux_err[nsub]=np.percentile(subarray[nsub,~np.isnan(subarray[nsub,:])],80)
        plt.semilogy(np.log2(lNant),aux_err,line+marker+color,label=caseStr)
    plt.xlabel('$K_{\\theta}$')
    plt.ylabel('80\%tile $\\tau_e$ error(ns)')
    plt.xticks(ticks=np.log2(lNant),labels=['$%d$'%x for x in lNant])
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/taue_vs_ntant.eps')
        
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    dicCaseMask=[x[0]=='dic'  and x[2]=='inf' and x[3]=='inf' for x in lErrMod]
    lNant=np.array([float(x[1]) for x in lErrMod if x[0]=='dic' and x[2]=='inf' and x[3]=='inf'])
    for nc in range(1,Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        aux_err=np.zeros_like(lNant)
        subarray=phi0_err[nc,dicCaseMask,:]
        for nsub in range(lNant.size):
            aux_err[nsub]=np.percentile(subarray[nsub,~np.isnan(subarray[nsub,:])],80)
        plt.semilogy(np.log2(lNant),aux_err,line+marker+color,label=caseStr)
    plt.xlabel('$K_{\\theta}$')
    plt.ylabel('80\%tile $\\phi_o$ error($^o$)')
    plt.xticks(ticks=np.log2(lNant),labels=['$%d$'%x for x in lNant])
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/phi0e_vs_ntant.eps')
        
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    dicCaseMask=[x[0]=='dic'  and x[1]=='inf' and x[3]=='inf' for x in lErrMod]
    lNant=np.array([float(x[2]) for x in lErrMod if x[0]=='dic' and x[1]=='inf' and x[3]=='inf'])
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        plt.semilogy(np.log2(lNant),np.percentile(location_error[nc,dicCaseMask,:],80,axis=1),line+marker+color,label=caseStr)
    plt.semilogy(np.log2(lNant),np.ones_like(lNant)*np.percentile(error_dumb,80),':k',label="random guess")
    plt.xlabel('$K_{\\phi}$')
    plt.ylabel('80\%tile location error(m)')
    plt.xticks(ticks=np.log2(lNant),labels=['$%d$'%x for x in lNant])
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/err_vs_nrant.eps')    
    
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    dicCaseMask=[x[0]=='dic'  and x[1]=='inf' and x[3]=='inf' for x in lErrMod]
    lNant=np.array([float(x[2]) for x in lErrMod if x[0]=='dic' and x[1]=='inf' and x[3]=='inf'])
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        plt.semilogy(np.log2(lNant),np.percentile(mapping_error[nc,dicCaseMask,:,:],80,axis=(1,2)),line+marker+color,label=caseStr)
    plt.semilogy(np.log2(lNant),np.ones_like(lNant)*np.percentile(map_dumb,80),':k',label="random guess")
    plt.xlabel('$K_{\\phi}$')
    plt.ylabel('80\%tile mapping error(m)')
    plt.xticks(ticks=np.log2(lNant),labels=['$%d$'%x for x in lNant])
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/map_vs_nrant.eps')  
        
        
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    dicCaseMask=[x[0]=='dic'  and x[1]=='inf' and x[2]=='inf' for x in lErrMod]
    lNant=np.array([float(x[3]) for x in lErrMod if x[0]=='dic' and x[1]=='inf' and x[2]=='inf'])
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        plt.semilogy(np.log2(lNant),np.percentile(location_error[nc,dicCaseMask,:],80,axis=1),line+marker+color,label=caseStr)
    plt.semilogy(np.log2(lNant),np.ones_like(lNant)*np.percentile(error_dumb,80),':k',label="random guess")
    plt.xlabel('$K_{\\tau}$')
    plt.ylabel('80\%tile location error(m)')
    plt.xticks(ticks=np.log2(lNant),labels=['$%d$'%x for x in lNant])
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/err_vs_ntau.eps')
    
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    dicCaseMask=[x[0]=='dic'  and x[1]=='inf' and x[2]=='inf' for x in lErrMod]
    lNant=np.array([float(x[3]) for x in lErrMod if x[0]=='dic' and x[1]=='inf' and x[2]=='inf'])
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        plt.semilogy(np.log2(lNant),np.percentile(mapping_error[nc,dicCaseMask,:,:],80,axis=(1,2)),line+marker+color,label=caseStr)
    plt.semilogy(np.log2(lNant),np.ones_like(lNant)*np.percentile(map_dumb,80),':k',label="random guess")
    plt.xlabel('$K_{\\tau}$')
    plt.ylabel('80\%tile mapping error(m)')
    plt.xticks(ticks=np.log2(lNant),labels=['$%d$'%x for x in lNant])
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/map_vs_ntau.eps')
    
    fig_ctr=fig_ctr+1
    plt.figure(fig_ctr)
    dicCaseMask=[x[0]=='dic'  and x[1]=='inf' and x[2]=='inf' for x in lErrMod]
    lNant=np.array([float(x[3]) for x in lErrMod if x[0]=='dic' and x[1]=='inf' and x[2]=='inf'])
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='*' if phi0Quant else 'o'
            line=':'
        else:
            caseStr="%s - %s %s"%(grouping,optimMthd,('Q-ini' if phi0Quant else 'BF-ini') if optimMthd == 'mmse' else '')
            line='-' if grouping=='D1' else ('-.' if optimMthd == 'mmse' else ':')
            marker='x' if phi0Quant else 's'
            color='b' if optimMthd=='brute' else 'g'
        plt.semilogy(np.log2(lNant),np.percentile(1e9*tauE_err[nc,dicCaseMask,:],80,axis=1),line+marker+color,label=caseStr)
    plt.xlabel('$K_{\\tau}$')
    plt.ylabel('80\%tile $\\tau_e$ error (ns)')
    plt.xticks(ticks=np.log2(lNant),labels=['$%d$'%x for x in lNant])
    plt.legend()
    if args.print:
        plt.savefig(outfoldername+'/taue_vs_ntau.eps')

if args.show:
    plt.show()