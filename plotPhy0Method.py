#!/usr/bin/python

import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

import MultipathLocationEstimator

Npath=4
x_true=np.random.rand(Npath)*40-20
y_true=np.random.rand(Npath)*40-20
x0_true=np.random.rand(1)*40-10
y0_true=np.random.rand(1)*40-10


phi0_true=np.random.rand(1)*2*np.pi
theta0_true=np.mod(np.arctan(y0_true/x0_true)+np.pi*(x0_true<0) , 2*np.pi)

theta_true=np.mod( np.arctan(y_true/x_true)+np.pi*(x_true<0), 2*np.pi)
phi_true=np.mod( np.pi-( np.arctan((y_true-y0_true)/(x0_true-x_true))+np.pi*((x0_true-x_true)<0) ), 2*np.pi)

c=3e8
tau0_true=y0_true/np.sin(theta0_true)/c

tau_true=(np.abs(y_true/np.sin(theta_true))+np.abs((y_true-y0_true)/np.sin(np.pi-phi_true)))/c

#AoD = np.mod(theta_true,2*np.pi)
AoD = np.mod(theta_true,2*np.pi)
AoA = np.mod(phi_true-phi0_true,2*np.pi)
dels = tau_true-tau0_true


plt.figure(1)
scaleguide=np.max(np.abs(np.concatenate([y_true,y0_true,x_true,x0_true],0)))
t=np.linspace(0,1,100)
for p in range(Npath):
    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k', label='_nolegend_')
plt.plot([x0_true,x0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.cos(phi0_true)],[y0_true,y0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.sin(phi0_true)],'c--', label='_nolegend_')
for of in range(3):
    p=0+of
    plt.plot([0,x_true[p],x0_true],[0,y_true[p],y0_true],'c--', label='_nolegend_' if of > 0 else '')
    plt.plot(x_true[p],y_true[p],'oc', label='_nolegend_')
    
    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k')
    t=np.linspace(0,1,21)
    plt.plot(x0_true+scaleguide*.05*(p+1)*np.cos(AoA[p]*t+phi0_true),y0_true+scaleguide*.05*(p+1)*np.sin(AoA[p]*t+phi0_true),'c--', label='_nolegend_')
plt.plot([x0_true,x0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.cos(phi0_true)],[y0_true,y0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.sin(phi0_true)],'b:', label='_nolegend_')
for of in range(3):
    p=1+of
    plt.plot([0,x_true[p],x0_true],[0,y_true[p],y0_true],'b:', label='_nolegend_' if of > 0 else '')
    plt.plot(x_true[p],y_true[p],'ob', label='_nolegend_')
    t=np.linspace(0,1,21)
    plt.plot(x0_true+scaleguide*.05*(p+1)*np.cos(AoA[p]*t+phi0_true),y0_true+scaleguide*.05*(p+1)*np.sin(AoA[p]*t+phi0_true),'b:', label='_nolegend_')

loc=MultipathLocationEstimator.MultipathLocationEstimator()

(x0_bad,y0_bad,tauEall_bad,x_bad,y_bad)= loc.computeAllPaths(AoD[0:3],AoA[0:3],dels[0:3],np.pi/4)
plt.plot(x0_bad,y0_bad,'^r', label='_nolegend_')
plt.plot([x0_bad,x0_bad+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.cos(np.pi/4)],[y0_bad,y0_bad+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.sin(np.pi/4)],'r:', label='_nolegend_')
for of in range(3):
    p=0+of
    plt.plot([0,x_bad[of],x0_bad],[0,y_bad[of],y0_bad],'r:', label='_nolegend_' if of > 0 else '')
    plt.plot(x_bad[of],y_bad[of],'or', label='_nolegend_')
    t=np.linspace(0,1,21)
    plt.plot(x0_bad+scaleguide*.05*(p+1)*np.cos(AoA[p]*t+np.pi/4),y0_bad+scaleguide*.05*(p+1)*np.sin(AoA[p]*t+np.pi/4),'r:', label='_nolegend_')

(x0_bad,y0_bad,tauEall_bad,x_bad,y_bad)= loc.computeAllPaths(AoD[1:4],AoA[1:4],dels[1:4],np.pi/4)
plt.plot(x0_bad,y0_bad,'^m', label='_nolegend_')
plt.plot([x0_bad,x0_bad+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.cos(np.pi/4)],[y0_bad,y0_bad+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.sin(np.pi/4)],'m:', label='_nolegend_')
for of in range(3):
    p=1+of
    plt.plot([0,x_bad[of],x0_bad],[0,y_bad[of],y0_bad],'m:', label='_nolegend_' if of > 0 else '')
    plt.plot(x_bad[of],y_bad[of],'om', label='_nolegend_')
    t=np.linspace(0,1,21)
    plt.plot(x0_bad+scaleguide*.05*(p+1)*np.cos(AoA[p]*t+np.pi/4),y0_bad+scaleguide*.05*(p+1)*np.sin(AoA[p]*t+np.pi/4),'m:', label='_nolegend_')

plt.plot(0,0,'sk')
plt.plot(x0_true,y0_true,'^g')
plt.plot([0,x0_true],[0,y0_true],':g')
plt.legend(['paths 1,2,3 $\\phi_o$ ok','paths 2,3,4 $\\phi_o$ ok','paths 1,2,3 $\\phi_o$ wrong','paths 2,3,4 $\\phi_o$ wrong','BS','UE'])
plt.title("All angles of a multipath channel")

plt.savefig('graphPhi02groupsExpl.eps')