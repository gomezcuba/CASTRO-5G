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
parser.add_argument('-S', type=int,help='No. error Std. points')
parser.add_argument('--minstd', help='Minimum error std in dB')
parser.add_argument('--maxstd', help='Maximum error std in dB')
parser.add_argument('--noerror', help='Add zero error case', action='store_true')
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

args = parser.parse_args("--nompg --noloc -N 100 -S 7 --noerror --label test --show --print".split(' '))

Nsims=args.N if args.N else 100
if args.S:
    minStd = args.minstd if args.minstd else -7
    maxStd = args.maxstd if args.maxstd else -1
    lSTD = np.logspace(minStd,maxStd,args.S)
    stepStd=(maxStd-minStd)/(args.S-1)
    if args.noerror:
        lSTD = np.concatenate([[0],lSTD])
elif args.noerror:
    lSTD=np.array([0])
else:
    print("ERROR: neither multipath error std nor no-error case specified")
Nstds=len(lSTD)
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
    outfoldername="./MPRayLocresults%s-%d-%d-%s"%(args.label,Nstds,Nsims,mpgen)
else:
    outfoldername="./MPRayLocresults-%d-%d-%s"%(Nstds,Nsims,mpgen)
if not os.path.isdir(outfoldername):
    os.mkdir(outfoldername)

#TODO: create command line arguments for these parameters
Xmax=50
Xmin=-50
Ymax=50
Ymin=-50
mpEstErrModel='Gaussian' #TODO add ifs etc. to make this selectable with OMP
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
    theta_est=data["theta_est"]
    phi_est=data["phi_est"]
    tau_est=data["tau_est"]
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
        tau0=y0/np.sin(theta0)/c
        tau=(np.abs(y/np.sin(theta))+np.abs((y-y0)/np.sin(phi)))/c
    elif mpgen == "3gpp":
        #TBW
        print("MultiPath generation method %s to be written"%mpgen)
    else:
        print("MultiPath generation method %s not recognized"%mpgen)
    
    theta_est=np.zeros((Nstds,Npath,Nsims))
    phi_est=np.zeros((Nstds,Npath,Nsims))
    tau_est=np.zeros((Nstds,Npath,Nsims))
    if mpEstErrModel=='Gaussian':
        for nv in range(Nstds):
            theta_est[nv,:,:]=np.mod(theta+lSTD[nv]*2*np.pi*np.random.rand(Npath,Nsims),2*np.pi)
            phi_est[nv,:,:]=np.mod(phi-phi0+lSTD[nv]*2*np.pi*np.random.rand(Npath,Nsims),2*np.pi)
            tau_est[nv,:,:]=tau+lSTD[nv]*40e-9*np.random.rand(Npath,Nsims)
    else:
        print("Multipath estimation error model %s to be written"%mpEstErrModel)
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
                 theta_est=theta_est,
                 phi_est=phi_est,
                 tau_est=tau_est)
    print("Total Generation Time:%s seconds"%(time.time()-t_start_gen))

if args.noloc: 
    data=np.load(outfoldername+'/locEstData.npz') 
    phi0_est=data["phi0_est"]
    x0_est=data["x0_est"]
    y0_est=data["y0_est"]
    x_est=data["x_est"]
    y_est=data["y_est"]
    run_time=data["run_time"]
else:        
    t_start_loc=time.time() 
    phi0_est=np.zeros((Ncases,Nstds,Nsims))
    x0_est=np.zeros((Ncases,Nstds,Nsims))
    y0_est=np.zeros((Ncases,Nstds,Nsims))
    x_est=np.zeros((Ncases,Nstds,Npath,Nsims))
    y_est=np.zeros((Ncases,Nstds,Npath,Nsims))
    run_time=np.zeros((Ncases,Nstds))
    
    loc=MultipathLocationEstimator.MultipathLocationEstimator(Npoint=1000,Nref=20,Ndiv=2,RootMethod='lm')
    
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        for nv in range(Nstds):
            bar = Bar("Case phi known:%s phi0quant: %s grouping:%s optimMthd:%s Var=%.3f"%(phi0Apriori,phi0Quant,grouping,optimMthd,10*np.log10(lSTD[nv])), max=Nsims)
            bar.check_tty = False        
            t_start_point = time.time()
            for ns in range(Nsims):
                if phi0Apriori:
                    if phi0Quant:
                        phi0_est[nc,nv,ns]=np.round(phi0[:,ns]/phi0GyrQuant)*phi0GyrQuant        
                    else:
                        phi0_est[nc,nv,ns]=phi0[:,ns]
                    (x0_est[nc,nv,ns],y0_est[nc,nv,ns],_,x_est[nc,nv,:,ns],y_est[nc,nv,:,ns])=loc.computeAllPathsLinear(theta_est[nv,:,ns],phi_est[nv,:,ns],tau_est[nv,:,ns],phi0_est[nc,nv,ns])
                else:
                #TODO make changes in location estimator and get rid of these ifs
                    if phi0Quant:
                        phi0_hint=np.round(phi0[:,ns]/phi0GyrQuant)*phi0GyrQuant        
                    else:
                        phi0_hint=None
                    if grouping=='3P':
                        if optimMthd=='brute':
                            methodAlgo='bisec'
                        else:
                            methodAlgo='fsolve'
                    else:
                        methodAlgo='fsolve_linear'
                    (phi0_est[nc,nv,ns],x0_est[nc,nv,ns],y0_est[nc,nv,ns],x_est[nc,nv,:,ns],y_est[nc,nv,:,ns],_)= loc.computeAllLocationsFromPaths(theta_est[nv,:,ns],phi_est[nv,:,ns],tau_est[nv,:,ns],method=methodAlgo,hint_phi0=phi0_hint)
                bar.next()
            bar.finish()
            run_time[nc,nv] = time.time() - t_start_point
    if not args.nosave: 
        np.savez(outfoldername+'/locEstData.npz',
                phi0_est=phi0_est,
                x0_est=x0_est,
                y0_est=y0_est,
                x_est=x_est,
                y_est=y_est,
                run_time=run_time)
    print("Total Generation Time:%s seconds"%(time.time()-t_start_gen))

location_error=np.sqrt(np.abs(x0-x0_est)**2+np.abs(y0-y0_est)**2)
plt.close('all')
plt.figure(1)
for nc in range(Ncases):
    (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
    if phi0Apriori:
        caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
        color='r'
        marker='x' if phi0Quant else 's'
        line=':'
    else:
        caseStr="%s - %s - %shint"%(grouping,optimMthd,'' if phi0Quant else 'no-')
        line='-.' if grouping=='3P' else '-'
        marker='*' if phi0Quant else 'o'
        color='b' if optimMthd=='brute' else 'g'
    plt.semilogx(np.percentile(location_error[nc,0,:],np.linspace(0,100,21)),np.linspace(0,1,21),line+marker+color,label=caseStr)
error_dumb=np.sqrt(np.abs(x0-x)**2+np.abs(y0-y)**2).reshape((-1))
plt.semilogx(np.percentile(error_dumb,np.linspace(0,100,21)),np.linspace(0,1,21),':k',label="random guess")
plt.xlabel('Location error(m)')
plt.ylabel('C.D.F.')
plt.legend()
if args.print:
    plt.savefig(outfoldername+'/cdflocerr_noerr.eps')

plt.figure(2)
P1pos=np.argmax(lSTD==1e-3)
for nc in range(Ncases):
    (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
    if phi0Apriori:
        caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
        color='r'
        marker='x' if phi0Quant else 's'
        line=':'
    else:
        caseStr="%s - %s - %shint"%(grouping,optimMthd,'' if phi0Quant else 'no-')
        line='-.' if grouping=='3P' else '-'
        marker='*' if phi0Quant else 'o'
        color='b' if optimMthd=='brute' else 'g'
        
    
    plt.semilogx(np.percentile(location_error[nc,P1pos,:],np.linspace(0,100,21)),np.linspace(0,1,21),line+marker+color,label=caseStr)
plt.semilogx(np.percentile(error_dumb,np.linspace(0,100,21)),np.linspace(0,1,21),':k',label="random guess")
plt.xlabel('Location error(m)')
plt.ylabel('C.D.F.')
plt.legend()
if args.print:
    plt.savefig(outfoldername+'/cdflocerr_1Perr.eps')

plt.figure(3)
lSTDaxis=lSTD
lSTDaxis[lSTD==0]=10**(minStd-stepStd)
for nc in range(Ncases):
    (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
    if phi0Apriori:
        caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
        color='r'
        marker='x' if phi0Quant else 's'
        line=':'
    else:
        caseStr="%s - %s - %shint"%(grouping,optimMthd,'' if phi0Quant else 'no-')
        line='-.' if grouping=='3P' else '-'
        marker='*' if phi0Quant else 'o'
        color='b' if optimMthd=='brute' else 'g'
    plt.loglog(lSTDaxis,np.percentile(location_error[nc,:,:],80,axis=1),line+marker+color,label=caseStr)
plt.loglog(lSTDaxis,np.ones_like(lSTD)*np.percentile(error_dumb,80),':k',label="random guess")
plt.xlabel('Error std.')
plt.ylabel('95\%tile location error(m)')
plt.legend()
plt.xticks(ticks=lSTDaxis,labels=['0']+['$10^{%d}$'%x for x in np.log10(lSTDaxis[1:])])
if args.print:
    plt.savefig(outfoldername+'/err_vs_std.eps')

plt.figure(4)
for nc in range(Ncases):
    if nc in [1,2,6]:
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='x' if phi0Quant else 's'
            line=':'
        else:
            caseStr="%s - %s - %shint"%(grouping,optimMthd,'' if phi0Quant else 'no-')
            line='-.' if grouping=='3P' else '-'
            marker='*' if phi0Quant else 'o'
            color='b' if optimMthd=='brute' else 'g'
        plt.plot(np.vstack((x0[0,:],x0_est[nc,P1pos,:])),np.vstack((y0[0,:],y0_est[nc,P1pos,:])),line+marker+color,label=caseStr)
plt.plot(x0.T,y0.T,'ok',label='locations')
handles, labels = plt.gca().get_legend_handles_labels()
# labels will be the keys of the dict, handles will be values
temp = {k:v for k,v in zip(labels, handles)}
plt.legend(temp.values(), temp.keys(), loc='best')
if args.print:
    plt.savefig(outfoldername+'/errormap.eps')

plt.figure(5)
for nc in range(Ncases):
    if nc in [1,2,6]:
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='x' if phi0Quant else 's'
            line=':'
        else:
            caseStr="%s - %s - %shint"%(grouping,optimMthd,'' if phi0Quant else 'no-')
            line='-.' if grouping=='3P' else '-'
            marker='*' if phi0Quant else 'o'
            color='b' if optimMthd=='brute' else 'g'
        plt.loglog(np.abs(phi0_est[nc,0,:]-phi0[0,:]),location_error[nc,0,:],marker+color,label=caseStr)
plt.xlabel('$\hat{\phi_o}$ error (rad)')
plt.ylabel('Location error (m)')
plt.legend()
if args.print:
    plt.savefig(outfoldername+'/err_vs_phi0.eps')

plt.figure(6)
for nc in range(Ncases):
    if nc in [1,2,6]:
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        if phi0Apriori:
            caseStr="phi0 quantized sensor" if phi0Quant else "phi0 known"
            color='r'
            marker='x' if phi0Quant else 's'
            line=':'
        else:
            caseStr="%s - %s - %shint"%(grouping,optimMthd,'' if phi0Quant else 'no-')
            line='-.' if grouping=='3P' else '-'
            marker='*' if phi0Quant else 'o'
            color='b' if optimMthd=='brute' else 'g'
        plt.loglog(np.abs(phi0_est[nc,P1pos,:]-phi0[0,:]),location_error[nc,P1pos,:],marker+color,label=caseStr)
plt.xlabel('$\hat{\phi_o}$ error (rad)')
plt.ylabel('Location error (m)')
plt.legend()
if args.print:
    plt.savefig(outfoldername+'/err_vs_phi0_inperf.eps')

if args.show:
    plt.show()