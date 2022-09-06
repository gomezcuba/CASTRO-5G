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

import MultipathLocationEstimator
#TODO: make the script below the __main__() of a class that can be imported by other python programs
parser = argparse.ArgumentParser(description='Multipath Location Estimation Simulator')
parser.add_argument('-N', type=int,help='No. simulated channels')
parser.add_argument('-V', type=int,help='No. variance points')
parser.add_argument('-G', type=int,help='Type of generator')
parser.add_argument('--npathgeo', type=int,help='No. paths per channel (Geo only)')
parser.add_argument('--algs', type=str,help='comma-separated list of algorithms')
parser.add_argument('--label', type=str,help='text label for stored files')
parser.add_argument('--norun',help='Do not perform simulation, load prior results file', action='store_true')
parser.add_argument('--nosave', help='Do not save simulation data a new results file', action='store_true')
parser.add_argument('--print', help='Generate plot files in eps', action='store_true')
parser.add_argument('--show', help='Open plot figures in window', action='store_true')
#args = parser.parse_args("-N 1000 -V 8 --label test --show --print".split(' '))
args = parser.parse_args("-N 1000 -V 8 --label test --norun --show --print".split(' '))

if args.N:
    Nsims=args.N
else:
    Nsims=100
if args.V:
    Nvars=args.V
else:
    Nvars=5    
lVAR=np.concatenate([[0],np.logspace(-4,-1,Nvars-1)])
if args.G: 
    mpgen = args.G
else:
    mpgen='Geo'
    
RUN_SIM=not args.norun

if args.algs:
    lcases=[
            tuple(case.split('-'))
            for case in args.algs.split(',')
            ]
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

#if args.label:
#    outfoldername="./OMPZresults%s-%d-%d-%d-%d"%(args.label,V,Nsim,Nframes)
#else:
#    outfoldername="./OMPZresults-%d-%d-%d-%d"%(D,V,N,Nframes)
#if not os.path.isdir(outfoldername):
#    os.mkdir(outfoldername)

#TODO: create command line arguments for these parameters
Xmax=50
Xmin=-50
Ymax=50
Ymin=-50
mpEstErrModel='Gaussian' #TODO add ifs etc. to make this selectable with OMP
phi0GyrQuant=2*np.pi/64

if RUN_SIM:
    if mpgen == 'Geo':
        if args.npathgeo:
            Npath=args.npathgeo
        else:
            Npath=20
        #generate locations and compute multipath 
        x=np.random.rand(Npath,Nsims)*(Xmax-Xmin)+Xmin
        y=np.random.rand(Npath,Nsims)*(Ymax-Ymin)+Ymin
        x0=np.random.rand(1,Nsims)*(Xmax-Xmin)-Xmin
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
    
    theta_est=np.zeros((Nvars,Npath,Nsims))
    phi_est=np.zeros((Nvars,Npath,Nsims))
    tau_est=np.zeros((Nvars,Npath,Nsims))
    if mpEstErrModel=='Gaussian':
        for nv in range(Nvars):
            theta_est[nv,:,:]=np.mod(theta+lVAR[nv]*2*np.pi*np.random.rand(Npath,Nsims),2*np.pi)
            phi_est[nv,:,:]=np.mod(phi-phi0+lVAR[nv]*2*np.pi*np.random.rand(Npath,Nsims),2*np.pi)
            tau_est[nv,:,:]=tau+lVAR[nv]*40e-9*np.random.rand(Npath,Nsims)
    else:
        print("Multipath estimation error model %s to be written"%mpEstErrModel)
        
    phi0_est=np.zeros((Ncases,Nvars,Nsims))
    x0_est=np.zeros((Ncases,Nvars,Nsims))
    y0_est=np.zeros((Ncases,Nvars,Nsims))
    x_est=np.zeros((Ncases,Nvars,Npath,Nsims))
    y_est=np.zeros((Ncases,Nvars,Npath,Nsims))
    run_time=np.zeros((Ncases,Nvars))
    
    loc=MultipathLocationEstimator.MultipathLocationEstimator(Npoint=1000,Nref=20,Ndiv=2,RootMethod='lm')
    
    for nc in range(Ncases):
        (phi0Apriori,phi0Quant,grouping,optimMthd)=lCases[nc]
        for nv in range(Nvars):
            bar = Bar("Case phi known:%s phi0quant: %s grouping:%s optimMthd:%s Var=%.3f"%(phi0Apriori,phi0Quant,grouping,optimMthd,10*np.log10(lVAR[nv])), max=Nsims)
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
plt.savefig('cdfpsi0_noerr.eps')

plt.figure(2)
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
        
    
    plt.semilogx(np.percentile(location_error[nc,np.argmax(lVAR==1e-4),:],np.linspace(0,100,21)),np.linspace(0,1,21),line+marker+color,label=caseStr)
plt.semilogx(np.percentile(error_dumb,np.linspace(0,100,21)),np.linspace(0,1,21),':k',label="random guess")
plt.xlabel('Location error(m)')
plt.ylabel('C.D.F.')
plt.legend()
plt.savefig('cdfpsi0_1Perr.eps')

plt.figure(3)
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
    plt.loglog(np.concatenate([[1e-5],lVAR[1:]]),np.percentile(location_error[nc,:,:],80,axis=1),line+marker+color,label=caseStr)
plt.loglog(np.concatenate([[1e-5],lVAR[1:]]),np.ones_like(lVAR)*np.percentile(error_dumb,80),':k',label="random guess")
plt.xlabel('Error std.')
plt.ylabel('95\%tile location error(m)')
plt.legend()
plt.savefig('err_vs_std.eps')
if args.show:
    plt.show()