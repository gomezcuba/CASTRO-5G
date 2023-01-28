#!/usr/bin/python

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import scipy.optimize as opt
#from progress bar import bar
import numpy as np
import time
import os
import sys
import argparse

plt.close('all')

x_true=np.random.rand(3)*40-20
y_true=np.random.rand(3)*40-20
x0_true=np.random.rand(1)*40-10
y0_true=np.random.rand(1)*40-10

phi0_true=np.random.rand(1)*2*np.pi
theta0_true=np.mod(np.arctan(y0_true/x0_true)+np.pi*(x0_true<0) , 2*np.pi)
theta_true=np.mod(np.arctan(y_true/x_true)+np.pi*(x_true<0) , 2*np.pi)
phi_true=np.mod(np.pi - ( np.arctan((y_true-y0_true)/(x0_true-x_true))+np.pi*((x0_true-x_true)<0) ) , 2*np.pi)

c=3e8
tau0_true=y0_true/np.sin(theta0_true)/c
tau_true=(np.abs(y_true/np.sin(theta_true))+np.abs((y_true-y0_true)/np.sin(phi_true)))/c

AoD = np.mod(theta_true,2*np.pi)
AoA = np.mod(phi_true-phi0_true,2*np.pi)
dels = tau_true-tau0_true+1e-8

plt.figure(1)
plt.plot(0,0,'sb')
plt.plot(x0_true,y0_true,'^g')
plt.plot([0,x0_true],[0,y0_true],':g')
plt.plot(x_true,y_true,'or')
scaleguide=np.max(np.abs(np.concatenate([y_true,y0_true,x_true,x0_true],0)))
plt.plot([x0_true,x0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.cos(phi0_true)],[y0_true,y0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.sin(phi0_true)],'k')
for p in range(np.shape(theta_true)[0]):
    plt.plot([0,x_true[p],x0_true],[0,y_true[p],y0_true],':k')
    t=np.linspace(0,1,21)
    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k')
    plt.plot(x0_true+scaleguide*.05*(p+1)*np.cos(AoA[p]*t+phi0_true),y0_true+scaleguide*.05*(p+1)*np.sin(AoA[p]*t+phi0_true),'k')
    
plt.axis(np.array([-1.1,1.1,-1.1,1.1])*scaleguide)
plt.title('Angles of several path, with 2D receiver, correct tau0')

phi0_est=phi0_true+[-np.pi/10,0,np.pi/8]

tgD = np.tan(AoD)
tgA = np.tan(np.pi-AoA-phi0_est.reshape(-1,1))
siD = np.sin(AoD)
siA = np.sin(np.pi-AoA-phi0_est.reshape(-1,1))

T=(1/tgD+1/tgA)
S=(1/siD+1/siA)
P=S/T
Q=P/tgA-1/siA


Dl = dels*c

Idp=(Dl[0:2]-Dl[1:3])/(P[:,0:2]-P[:,1:3])
Slp=(Q[:,0:2]-Q[:,1:3])/(P[:,0:2]-P[:,1:3])

y0=(Idp[:,0]-Idp[:,1])/(Slp[:,0]-Slp[:,1])
x0=Idp[:,0]-y0*Slp[:,0]

y0=y0.reshape(-1,1)
x0=x0.reshape(-1,1)

vy=(x0+y0/tgA)/T
vx=vy/tgD
    


plt.figure(2)
plt.plot(0,0,'sb')
plt.plot(x0,y0,'^g')
plt.plot(np.concatenate([np.zeros_like(x0),x0],1),np.concatenate([np.zeros_like(y0),y0],1),':g')
plt.plot(vx,vy,'or')
scaleguide=np.max(np.abs(np.concatenate([vy,x0,vx,y0],1)))
plt.plot(np.concatenate([x0,x0+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.cos(phi0_est.reshape(-1,1))],1).T,np.concatenate([y0,y0+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.sin(phi0_est.reshape(-1,1))],1).T,'k')
for p in range(np.shape(AoD)[0]):
    plt.plot(np.concatenate([np.zeros_like(x0),vx[:,p:p+1],x0],1).T,np.concatenate([np.zeros_like(x0),vy[:,p:p+1],y0],1).T,':k')
    t=np.linspace(0,1,21)
    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k')
    plt.plot((x0+scaleguide*.05*(p+1)*np.cos((AoA[p])*t+phi0_est.reshape(-1,1))).T,(y0+scaleguide*.05*(p+1)*np.sin((AoA[p])*t+phi0_est.reshape(-1,1))).T,'k')

plt.axis(np.array([-1.1,1.1,-1.1,1.1])*scaleguide)
plt.title('Angles of several path, with 2D receiver, multiple incorrect tau0s')