#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

def fULA(incidAngle , Nant = 4, dInterElement = .5):
    # returns an anttenna array response vector corresponding to a Uniform Linear Array for each item in incidAngle (one extra dimensions is added at the end)
    # inputs  incidAngle : numpy array containing one or more incidence angles
    # input         Nant : number of MIMO antennnas of the array
    # input InterElement : separation between antenna elements, default lambda/2
    # output arrayVector : numpy array containing one or more response vectors, with simensions (incidAngle.shape ,Nant ,1 )
    
    if isinstance(incidAngle,np.ndarray):
        incidAngle=incidAngle[...,None]
                        
    return np.exp( -2j * np.pi *  dInterElement * np.arange(Nant) * np.sin(incidAngle) ) /np.sqrt(Nant)

N=8
O=4
mindBpolar = -30
Nplotpoint =1000
angles = np.arange(0,2*np.pi,2*np.pi/Nplotpoint)

fig = plt.figure(1)
h_array = fULA(angles,N,.5)
V_fft = np.fft.fftshift(fULA(np.arange(-1,1,2/(N*O)),N),axes=0)
G_fft= h_array.conj() @ V_fft.T
plt.polar(angles,np.maximum(10*np.log10(np.abs(G_fft)**2),mindBpolar),':',color=(.5,.5,.5),label='')
for k in range(4):
    plt.polar(angles,np.maximum(10*np.log10(np.abs(G_fft[:,k])**2),mindBpolar),'--',label=f'N={N} O={O} $\\lambda/2$ $i_1={k}$')
plt.legend()
plt.savefig('beamDiagDFTlambdahalf.png')

fig = plt.figure(2)
h_array = fULA(angles,N,.5/O)
V_fft = np.fft.fftshift(fULA(np.arange(-1,1,2/(N*O)),N),axes=0)
G_fft= h_array.conj() @ V_fft.T
plt.polar(angles,np.maximum(10*np.log10(np.abs(G_fft)**2),mindBpolar),':',color=(.5,.5,.5),label='')
for k in range(4):
    plt.polar(angles,np.maximum(10*np.log10(np.abs(G_fft[:,k])**2),mindBpolar),'--',label=f'N={N} O={O} $\\lambda/2O$ $i_1={k}$')
plt.legend()
plt.savefig('beamDiagDFTlambdaO.png')