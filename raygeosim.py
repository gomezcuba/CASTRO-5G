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

plt.close('all')

import MultipathLocationEstimator

Npath=20
Nsims=100

#random locations in a 40m square
x=np.random.rand(Npath,Nsims)*100-50
y=np.random.rand(Npath,Nsims)*100-50
x0=np.random.rand(1,Nsims)*100-50
y0=np.random.rand(1,Nsims)*100-50

#angles from locations
theta0=np.mod( np.arctan(y0/x0)+np.pi*(x0<0) , 2*np.pi)
theta=np.mod( np.arctan(y/x)+np.pi*(x<0) , 2*np.pi)
phi=np.mod(np.pi - (np.arctan((y-y0)/(x0-x))+np.pi*((x0-x)<0)) , 2*np.pi)

#delays based on distance
c=3e8
tau0=y0/np.sin(theta0)/c
tau=(np.abs(y/np.sin(theta))+np.abs((y-y0)/np.sin(phi)))/c

#typical channel multipath estimation outputs
#AoD = np.mod(theta,2*np.pi)
Tserr=2.5e-9
Nanterr=256
AoD = np.mod(theta+np.random.rand(Npath,Nsims)*2*np.pi/Nanterr,2*np.pi)
phi0=np.random.rand(1,Nsims)*2*np.pi #receiver angular measurement offset
AoA = np.mod(phi-phi0+np.random.rand(Npath,Nsims)*2*np.pi/Nanterr,2*np.pi)
clock_error=(40/c)*np.random.rand(1,Nsims) #delay estimation error
del_error=(Tserr)*np.random.randn(Npath,Nsims) #delay estimation error
dels = tau-tau0+clock_error+del_error

loc=MultipathLocationEstimator.MultipathLocationEstimator(Npoint=1000,Nref=20,Ndiv=2,RootMethod='lm')

t_start_b = time.time()
phi0_b=np.zeros((1,Nsims))
x0_b=np.zeros((1,Nsims))
y0_b=np.zeros((1,Nsims))
x_b=np.zeros((Npath,Nsims))
y_b=np.zeros((Npath,Nsims))
bar = Bar("bisec", max=Nsims)
bar.check_tty = False
for nsim in range(Nsims):
    (phi0_b[:,nsim],x0_b[:,nsim],y0_b[:,nsim],x_b[:,nsim],y_b[:,nsim])= loc.computeAllLocationsFromPaths(AoD[:,nsim],AoA[:,nsim],dels[:,nsim],method='bisec')
    bar.next()
bar.finish()
error_bisec=np.sqrt(np.abs(x0-x0_b)**2+np.abs(y0-y0_b))
t_run_b = time.time() - t_start_b
plt.figure(1)
plt.semilogx(np.sort(error_bisec).T,np.linspace(0,1,error_bisec.size),'b')
#
t_start_r= time.time()
phi0_r=np.zeros((1,Nsims))
x0_r=np.zeros((1,Nsims))
y0_r=np.zeros((1,Nsims))
x_r=np.zeros((Npath,Nsims))
y_r=np.zeros((Npath,Nsims))
bar = Bar("froot", max=Nsims)
bar.check_tty = False
for nsim in range(Nsims):
    (phi0_r[:,nsim],x0_r[:,nsim],y0_r[:,nsim],x_r[:,nsim],y_r[:,nsim])= loc.computeAllLocationsFromPaths(AoD[:,nsim],AoA[:,nsim],dels[:,nsim],method='fsolve')
    bar.next()
bar.finish()
error_root=np.sqrt(np.abs(x0-x0_r)**2+np.abs(y0-y0_r))
t_run_r = time.time() - t_start_r
plt.semilogx(np.sort(error_root).T,np.linspace(0,1,error_root.size),'-.r')


#
t_start_r2= time.time()
phi0_r2=np.zeros((1,Nsims))
x0_r2=np.zeros((1,Nsims))
y0_r2=np.zeros((1,Nsims))
x_r2=np.zeros((Npath,Nsims))
y_r2=np.zeros((Npath,Nsims))
bar = Bar("froot_linear", max=Nsims)
bar.check_tty = False
for nsim in range(Nsims):
    (phi0_r2[:,nsim],x0_r2[:,nsim],y0_r2[:,nsim],x_r2[:,nsim],y_r2[:,nsim])= loc.computeAllLocationsFromPaths(AoD[:,nsim],AoA[:,nsim],dels[:,nsim],method='fsolve_linear')
    bar.next()
bar.finish()
error_root2=np.sqrt(np.abs(x0-x0_r2)**2+np.abs(y0-y0_r2))
t_run_r2 = time.time() - t_start_r2
plt.semilogx(np.sort(error_root2).T,np.linspace(0,1,error_root2.size),'-.g')

t_start_k= time.time()
x0_k=np.zeros((1,Nsims))
y0_k=np.zeros((1,Nsims))
x_k=np.zeros((Npath,Nsims))
y_k=np.zeros((Npath,Nsims))
bar = Bar("know phi 3-path method", max=Nsims)
bar.check_tty = False
for nsim in range(Nsims):
    (x0all,y0all,tauEall)=loc.computePosFrom3PathsKnownPhi0(AoD[:,nsim],AoA[:,nsim],dels[:,nsim],phi0[:,nsim])
    x0_k[:,nsim]=np.mean(x0all)
    y0_k[:,nsim]=np.mean(y0all)
    bar.next()
bar.finish()
error_k=np.sqrt(np.abs(x0-x0_k)**2+np.abs(y0-y0_k))
t_run_k = time.time() - t_start_k
plt.semilogx(np.sort(error_k).T,np.linspace(0,1,error_k.size),':or')


t_start_k2= time.time()
x0_k2=np.zeros((1,Nsims))
y0_k2=np.zeros((1,Nsims))
x_k2=np.zeros((Npath,Nsims))
y_k2=np.zeros((Npath,Nsims))
bar = Bar("know phi linear method", max=Nsims)
bar.check_tty = False
for nsim in range(Nsims):
    (x0_k2[:,nsim],y0_k2[:,nsim],_,x_k2[:,nsim],y_k2[:,nsim])=loc.computeAllPathsLinear(AoD[:,nsim],AoA[:,nsim],dels[:,nsim],phi0[:,nsim])
    bar.next()
bar.finish()
error_k2=np.sqrt(np.abs(x0-x0_k2)**2+np.abs(y0-y0_k2))
t_run_k2 = time.time() - t_start_k2
plt.semilogx(np.sort(error_k2).T,np.linspace(0,1,error_k2.size),':xg')


error_dumb=np.sqrt(np.abs(x0-x)**2+np.abs(y0-y)**2).reshape((-1))
plt.semilogx(np.sort(error_dumb).T,np.linspace(0,1,error_dumb.size),':k')

plt.xlabel('Location error(m)')
plt.ylabel('C.D.F.')
plt.legend(['brute force $\hat\psi_o$ 3path','fzero $\psi_o$ 3path','fzero $\psi_o$ linear','$\psi_o$ known, 3path','$\psi_o$ known, linear','randomguess'])
plt.savefig('cdflocgeosim.eps')


plt.figure(2)

#plt.plot(x0.T,y0.T,'ob',label='locations')
#plt.gca().add_collection(matplotlib.collections.LineCollection(np.transpose(np.array([np.vstack((x0,x0_b)),np.vstack((y0,y0_b))]),(2,1,0))))
#plt.legend()

#plt.plot(np.vstack((x0,x0_b)),np.vstack((y0,y0_b)),':xr',label='brute force')
#plt.plot(np.vstack((x0,x0_r)),np.vstack((y0,y0_r)),':+k',label='fzero')
plt.plot(np.vstack((x0,x0_r2)),np.vstack((y0,y0_r2)),':+g',label='fzero $\psi_o$ linear')
plt.plot(np.vstack((x0,x0_k2)),np.vstack((y0,y0_k2)),':xr',label='known $\psi_o$ linear')
plt.plot(x0.T,y0.T,'ob',label='locations')
handles, labels = plt.gca().get_legend_handles_labels()
# labels will be the keys of the dict, handles will be values
temp = {k:v for k,v in zip(labels, handles)}
plt.legend(temp.values(), temp.keys(), loc='best')
plt.savefig('errormap.eps')

plt.figure(3)
plt.semilogx(np.sort(phi0-phi0_b).T,np.linspace(0,1,phi0.size),'r')
plt.semilogx(np.sort(phi0-phi0_r).T,np.linspace(0,1,phi0.size),'-.b')
plt.semilogx(np.sort(phi0-phi0_r2).T,np.linspace(0,1,phi0.size),':xg')
plt.xlabel('$\psi_o-\hat{\psi}_0$ (rad)')
plt.ylabel('C.D.F.')
plt.legend(['brute force $\hat\psi_o$ 3path','fzero $\psi_o$ 3path','fzero $\psi_o$ linear'])
plt.savefig('cdfpsi0err.eps')