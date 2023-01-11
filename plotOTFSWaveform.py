#!/usr/bin/python

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np

def dirichlet(t,P):
    if isinstance(t,np.ndarray):
        return np.array([dirichlet(titem,P) for titem in t])
    elif isinstance(t, (list,tuple)):            
        return [dirichlet(titem,P) for titem in t]
    else:
        return 1 if t==0 else np.exp(1j*np.pi*(P-1)*t/P)*np.sin(np.pi*t)/np.sin(np.pi*t/P)/P


def foldFFT(x,N):
    return np.sum(np.fft.fft(np.concatenate((x,np.zeros(N-len(x)%N))).reshape(N,-1).T),0)
def ComplexToSpiralPlot(z):
    x=np.arange(len(z))
    z=np.real(z)
    y=np.imag(z)
    return(x,y,z)


L=16
V=4
D=4
Xoversample=10
t=np.arange(-1,L+1,1/Xoversample);

x=dirichlet(t,L)
v=1;
y=np.zeros(np.shape(x),dtype=np.complex64)
for u in range(V):
    y+=dirichlet(t-u*D,L)*np.exp(2j*np.pi*u*v/V)
z=dirichlet(t,D)*np.exp(2j*np.pi*t/D*v/V)


plt.close("all")
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t,np.abs(x))
plt.plot(t,np.abs(y))
plt.plot(t,np.abs(z),'x')
plt.subplot(2,1,2)
plt.plot(t,np.angle(x))
plt.plot(t,np.angle(y))
plt.plot(t,np.angle(z),'x')

Nfft=128

X=foldFFT(x,Nfft)
Y=foldFFT(y,Nfft)
Z=foldFFT(z,Nfft)
f=np.arange(Nfft)
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(f,np.abs(X))
plt.plot(f,np.abs(Y))
plt.plot(f,np.abs(Z))
plt.subplot(2,1,2)
plt.plot(f,np.angle(X))
plt.plot(f,np.angle(Y))
plt.plot(f,np.angle(Z))

fig = plt.figure(3,figsize=(6.4,10))
#ax = fig.add_subplot(5,1,1, projection='3d')
#ax.view_init(azim=-75,elev=22.5)
#ax.plot(t,x.imag,x.real)
#ax.plot(t[Xoversample::Xoversample],x.imag[Xoversample::Xoversample],x.real[Xoversample::Xoversample],':or')
#plt.xlabel('t')
#plt.ylabel('Re')
#plt.zlabel('Im')
plt.figure(4,figsize=(6.4,10))
plt.subplot(5,1,1)
plt.plot(t,x.real)
plt.stem(t[Xoversample::Xoversample],x.real[Xoversample::Xoversample],markerfmt='or',linefmt='-r',basefmt='k:')
plt.xlabel('t')
plt.ylabel('Re')
plt.axis([min(t),max(t),-1,+1])
plt.figure(5,figsize=(6.4,10))
plt.subplot(5,1,1)
plt.plot(t,x.imag)
plt.stem(t[Xoversample::Xoversample],x.imag[Xoversample::Xoversample],markerfmt='or',linefmt='-r',basefmt='k:')
plt.xlabel('t')
plt.ylabel('Im')
plt.axis([min(t),max(t),-1,+1])


for v in range(V):
    y=np.zeros(np.shape(x),dtype=np.complex64)
    for u in range(V):
        y+=dirichlet(t-u*D,L)*np.exp(2j*np.pi*u*v/V)
    z=dirichlet(t,D)*np.exp(2j*np.pi*t/D*v/V)
    fig = plt.figure(3)
    ax = fig.add_subplot(4,1,1+v, projection='3d')
    ax.view_init(azim=-75,elev=22.5)
    ax = fig.gca(projection='3d')
    ax.plot(t,z.imag,z.real)
    ax.plot(t[Xoversample::Xoversample],z.imag[Xoversample::Xoversample],z.real[Xoversample::Xoversample],':or')
#    ax.plot(t,y.imag,y.real,'gx')
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\mathcal{I}$")
    ax.set_zlabel("$\mathcal{R}$")
    ax.text2D(0.05, 0.95, "$Z_x[0,%d]=1$"%(v), transform=ax.transAxes)
#    plt.zlabel('Im')
#    plt.axis([min(t),max(t),-1,+1])
    plt.figure(4)
    plt.subplot(5,1,2+v)
    plt.plot(t,z.real)
    plt.stem(t[Xoversample::Xoversample],z.real[Xoversample::Xoversample],markerfmt='or',linefmt='-r',basefmt='k:')
#    plt.plot(t,y.real,'gx')
    plt.xlabel('t')
    plt.ylabel('Re')
    plt.axis([min(t),max(t),-1,+1])
    plt.figure(5)
    plt.subplot(5,1,2+v)
    plt.plot(t,z.imag)
    plt.stem(t[Xoversample::Xoversample],z.imag[Xoversample::Xoversample],markerfmt='or',linefmt='-r',basefmt='k:')
    plt.xlabel('t')
    plt.ylabel('Im')
    plt.axis([min(t),max(t),-1,+1])

plt.figure(3)
plt.savefig('pulses3D.eps')
plt.figure(4).savefig('pulsesRe.eps')
plt.figure(5).savefig('pulsesIm.eps')