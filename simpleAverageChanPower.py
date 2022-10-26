#!/usr/bin/python
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

Nt=64
Nr=64
Np=3

Hr=(np.random.randn(1000,Nr,Nt)+1j*np.random.randn(1000,Nr,Nt))/np.sqrt(2)
Ar=np.fft.fft(Hr,Nr,axis=1)/np.sqrt(Nr)
Ar=np.fft.fft(Ar,Nt,axis=2)/np.sqrt(Nt)

Gp=(np.random.randn(Np,1000,1,1)+1j*np.random.randn(Np,1000,1,1))/np.sqrt(2)/np.sqrt(Np)
AoA=np.reshape(2*np.pi*np.random.rand(Np),(-1,1,1,1))
AoD=np.reshape(2*np.pi*np.random.rand(Np),(-1,1,1,1))

Hp=np.sum(Gp*np.exp(-1j*np.pi*(np.reshape(np.arange(0,Nr),(1,1,-1,1))*np.sin(AoA)+np.arange(0,Nt)*np.sin(AoD))),axis=0)
Ap=np.fft.fft(Hp,Nr,axis=1)/np.sqrt(Nr)
Ap=np.fft.fft(Ap,Nt,axis=2)/np.sqrt(Nt)
    
plt.figure(1)    
p1=plt.pcolor(10*np.log10(np.mean(np.abs(Ar)**2,axis=(0))))
cbar = plt.colorbar(p1)
cbar.set_label('$E[|A_{i,j}|^2]$ dB')
plt.savefig('AvgRayleighAmatrix.eps')
plt.figure(2)    
p2=plt.pcolor(10*np.log10(np.mean(np.abs(Ap)**2,axis=(0))))
cbar = plt.colorbar(p2)
cbar.set_label('$E[|A_{i,j}|^2]$ dB')
plt.savefig('AvgSparseAmatrix.eps')