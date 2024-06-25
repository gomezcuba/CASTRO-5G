#!/usr/bin/python
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

import numpy as np

plt.close('all')
Npoint=2
x_true = np.random.rand(Npoint,1)*50-25
y_true = np.random.rand(Npoint,1)*50-25
l0_true = np.random.rand(1,1)*40-20
theta_true=np.arctan(y_true/x_true)+np.pi*(x_true<0)
phi_true=np.arctan(y_true/(l0_true-x_true))+np.pi*((l0_true-x_true)<0)
c=3e8
tau0_true=l0_true/c
tau_true=(np.abs(y_true/np.sin(theta_true))+np.abs(y_true/np.sin(phi_true)))/c

AoD = np.mod(theta_true,2*np.pi)
AoA = np.mod(phi_true,2*np.pi)
tau = tau_true-tau0_true
#AoD = np.random.rand(Npoint,1)*np.pi*2
#AoA = np.random.rand(Npoint,1)*np.pi + np.pi*(AoD>np.pi)
#tau=np.random.rand(Npoint,1)*80e-9


Dl = tau*c

tgD = np.tan(AoD)
tgA = np.tan(AoA)
siD = np.sin(AoD)
siA = np.sin(AoA)

argDen=(1/siD+1/siA-1/tgD-1/tgA)
y=Dl/argDen
x=y/tgD
l=y*(1/siD+1/siA)
l0=l-Dl

plt.figure(1)
plt.plot(0,0,'sb')
plt.plot(l0,np.zeros_like(l0),'^g')
plt.plot(x,y,'or')
scaleguide=np.max(np.abs([y,l0,x]))
for p in range(np.shape(AoD)[0]):
    plt.plot([0,x[p,0],l0[p,0]],[0,y[p,0],0],':k')
    t=np.linspace(0,1,21)
    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k')
    plt.plot(l0[p]-scaleguide*.05*(p+1)*np.cos(AoA[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoA[p]*t),'k')

print("D: %s A: %s"%(AoD*180/np.pi,AoA*180/np.pi))
plt.axis(np.array([-1.1,1.1,-1.1,1.1])*scaleguide)
plt.title('Angles of one path  (1D receiver pos)')

tau_ref_err=np.min(tau)/2*np.linspace(-1,1,5).reshape((1,-1))

Dl = (tau_ref_err+tau)*c

argDen=(1/siD+1/siA-1/tgD-1/tgA)
y=Dl/argDen
x=y/tgD
l=y*(1/siD+1/siA)
l0=l-Dl

plt.figure(2)
plt.plot(0,0,'sb')
plt.plot(l0,np.zeros_like(l0),'^g')
plt.plot(x,y,'or')
scaleguide=np.max(np.abs([y,l0,x]))
for p in range(np.shape(AoD)[0]):
    plt.plot(np.concatenate((np.zeros_like(x[p:p+1,:]),x[p:p+1,:],l0[p:p+1,:]),0),np.concatenate((np.zeros_like(x[p:p+1,:]),y[p:p+1,:],np.zeros_like(x[p:p+1,:])),0),':k')
    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k')
    plt.plot(l0[p:p+1,:].T-scaleguide*.05*(p+1)*np.cos(AoA[p]*t),np.zeros_like(x[p:p+1,:].T)+scaleguide*.05*(p+1)*np.sin(AoA[p]*t),'k')
plt.title('Angles of multiple paths with different tau errors (1D receiver pos)')

fac=(1/tgD+1/tgA)/argDen
est_tau_err=(l0[0,:]-l0[1,:])/(fac[0,:]-fac[1,:])/c

Dl2 = (tau_ref_err-est_tau_err+tau)*c

y=Dl2/argDen
x=y/tgD
l=y*(1/siD+1/siA)
l0=l-Dl2
plt.figure(3)
plt.plot(0,0,'sb')
plt.plot(l0,np.zeros_like(l0),'^g')
plt.plot(x,y,'or')
scaleguide=np.max(np.abs([y,l0,x]))
for p in range(np.shape(AoD)[0]):
    plt.plot(np.concatenate((np.zeros_like(x[p:p+1,:]),x[p:p+1,:],l0[p:p+1,:]),0),np.concatenate((np.zeros_like(x[p:p+1,:]),y[p:p+1,:],np.zeros_like(x[p:p+1,:])),0),':k')
    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k')
    plt.plot(l0[0,0]-scaleguide*.05*(p+1)*np.cos(AoA[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoA[p]*t),'k')
plt.title('Angles of 2 paths with same tau error (unique solution, 1D receiver pos)')