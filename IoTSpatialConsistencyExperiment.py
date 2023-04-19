#!/usr/bin/python

from progress.bar import Bar
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import os
import sys
import argparse
plt.close('all')
#import MultipathLocationEstimator as mploc
#import threeGPPMultipathGenerator as mp3g
import multipathChannel as ch
import OMPCachedRunner as oc
import MIMOPilotChannel as pil

plt.close('all')

class UIMultipathChannelModel:
    def __init__(self,Nt=1,Na=1,Nd=1):
        self.Nt=Nt
        self.Na=Na
        self.Nd=Nd
        self.Natoms=Nt*Nd*Na        
    def zAWGN(self,shape):
        return ( np.random.normal(size=shape) + 1j*np.random.normal(size=shape) ) * np.sqrt( 1 / 2.0 )
    def computeDEC(self,delay,AoD,AoA,coefs):
        tpulse=np.sinc(np.arange(self.Nt).reshape(self.Nt,1,1,1)-delay)
        arrayA=np.exp(-1j*np.pi*np.sin(AoA)*np.arange(self.Na).reshape(1,self.Na,1,1))
        arrayD=np.exp(-1j*np.pi*np.sin(AoD)*np.arange(self.Nd).reshape(1,1,self.Nd,1))        
        return(np.sum(coefs*tpulse*arrayA*arrayD,3))
        
    def generateDEC(self,Npoints=1):
        delay=np.random.rand(Npoints)*self.Nt
        AoD=np.random.rand(Npoints)*np.pi*2
        AoA=np.random.rand(Npoints)*np.pi*2
        coefs=self.zAWGN(Npoints)
        coefs=coefs/np.sqrt(np.sum(np.abs(coefs)**2))
        return (self.computeDEC(delay,AoD,AoA,coefs),coefs,delay,AoD,AoA)
    
def displaceMultipathChannel(de,ad,aa,deltax,deltay):
    c=3e8
    newdelay=de+(np.cos(aa)*deltax+np.sin(aa)*deltay)/c
    path_length=c*de
    #                .....x1y1
    #           .....      ^
    #   tg  ....           |   delta_dist
    #  <.....              |
    #____path_length___>xoyo
    delta_distD=-np.sin(ad)*deltax+np.cos(ad)*deltay
    newaod=ad-np.arctan(delta_distD/path_length)
    delta_distA=+np.sin(aa)*deltax-np.cos(aa)*deltay
    newaoa=aa-np.arctan(delta_distA/path_length)
    #for 3GPP coefs should be updated according to their delay deppendency
    return(newdelay,newaod,newaoa)
    

Nt=32
Nk=32
Nd=16
Na=16

Nxp=4
Nrft=1
Nrfr=4

Nsim=1

c=3e8
Ts=320e-9/Nt#2.5e-9
Nu=10
Npath=3
xmax=100
ymax=100
dmax=10


NMSE=np.zeros((Nsim,Nu))
NMSEdisp=np.zeros((Nsim,Nu))
aodFlipErrors=np.zeros(Nsim)
aoaFlipErrors=np.zeros(Nsim)
aodFlipWeight=np.zeros(Nsim)
aoaFlipWeight=np.zeros(Nsim)

bGenRand=True

chgen = UIMultipathChannelModel(Nt,Nd,Na)
pilgen = pil.MIMOPilotChannel("IDUV")
omprunner = oc.OMPCachedRunner()

if bGenRand:
    (w,v)=pilgen.generatePilots((Nk,Nxp,Nrfr,Na,Nd,Nrft),"IDUV")

for isim in range(Nsim):
    if bGenRand:
        x1=(np.random.rand(1)-.5)*xmax*2
        y1=(np.random.rand(1)-.5)*xmax*2
        
        dist=np.random.rand(Nu-1)*dmax
        dire=np.random.rand(Nu-1)*np.pi*2
        xstep=dist*np.cos(dire)
        ystep=dist*np.sin(dire)
    
    x=np.cumsum(np.concatenate((x1,xstep)))
    y=np.cumsum(np.concatenate((y1,ystep)))
    
    if bGenRand:
        ht,coefs,delay1,aod1,aoa1=chgen.generateDEC(Npath)
    delay1-np.min(delay1)
    
    ord_true=np.argsort(-np.abs(coefs)**2)
    coefs=coefs[ord_true]
    delay1=delay1[ord_true]
    aod1=aod1[ord_true]
    aoa1=aoa1[ord_true]
    
    tdelay=np.zeros((Nu,Npath))
    d1=np.sqrt(x1**2+y1**2)
    tdelay[0,:]=delay1*Ts+d1/c
    aod=np.zeros((Nu,Npath))
    aod[0,:]=aod1
    aoa=np.zeros((Nu,Npath))
    aoa[0,:]=aoa1
    hall=np.zeros((Nu,Nt,Na,Nd),dtype=np.complex64)
#    hall[0,:,:,:]=ht        
    for nu in range(Nu-1):    
        tdelay[nu+1,:],aod[nu+1,:],aoa[nu+1,:]=displaceMultipathChannel(tdelay[nu,:],aod[nu,:],aoa[nu,:],xstep[nu],ystep[nu])
        #for 3GPP coefs must be updated too
    clock_offset=np.minimum(np.min((tdelay-d1/c)/Ts),0)
    for nu in range(Nu):    #note ht was generated without clock offset and must be modified
        hall[nu,:,:,:]=chgen.computeDEC((tdelay[nu,:]-d1/c)/Ts-clock_offset,aod[nu,:],aoa[nu,:],coefs)
        
    hkall=np.fft.fft(hall,Nk,axis=1)        
    
    ###########################################################################################

    p_tx = 500e-3        #(first version )transmitter power W [can be an input parameter].
    delta_f = 15e3       #carrier spacing [Hz]
    p_loss = 1e-12       #pathloss

    Temp = 290                       # Define temperature T (in Kelvin), 17ºC
    k_boltz = 1.380649e-23           # Boltzmann's constant k [J/K]
    N0_noise = k_boltz * Temp        # Potencia de ruido W/Hz

    #Beamforming calculation
    H_beamf_max = None
    best_v_beamf = None
    max_gain = -np.inf
    beams_table = np.eye(Nd)         #table of beam vectors

    for row in beams_table:
        v_beamf = row.reshape(-1, 1) / np.linalg.norm(row)          # initializes beamforming direction vector v  
        H_beamf = np.zeros((Nu, Nk), np.complex64)                   # define gain matrix for the k subcarrier at time nu
        
        for nu in range(Nu):
            for k in range(Nk):
                hkall_2 = hkall[nu, k, :, :]
                hv = hkall_2 @ v_beamf
                w = hv / np.linalg.norm(hv)                     #normalized beamforming vector
                H_beamf[nu, k] = w.conj().T @ hv       

        current_gain = np.sum(np.abs(H_beamf))                  #gain for all subcarriers combined
        if current_gain > max_gain:
            max_gain = current_gain
            best_v_beamf = v_beamf
            H_beamf_max = H_beamf

    print("Max gain: " + str(max_gain))

    # RX 
    #calculation of the SNR for each subcarrier k
    SNR_k = np.zeros((Nu, Nk), dtype=np.float32)    #subcarrier SNR array
    for nu in range(Nu):
        for k in range(Nk):  
            SNR_k[nu, k] = ( p_tx * p_loss * (np.abs(H_beamf_max[nu, k]) **2) ) / ( N0_noise * Nk * delta_f )   

    SNR_k_dB = 10*np.log10(SNR_k)

    #Achievable Rate Calculation
    ach_rate = np.sum(np.log2(1 + SNR_k), axis = 1) * delta_f 
    spect_eff_k = ach_rate / (Nk * delta_f) 
    
    #TX 
    hkall_est = np.zeros((Nu,Nk,Na,Nd),dtype=np.complex64)     #estimated channel by the tx
    SNR_k_est = np.zeros((Nu,Nk))                                  #SNR estimated by tx
    rate_tx = np.zeros((Nu),dtype = np.float32)                    #tx rate 

    #estimated channel matrix and estimated SNR
    hkall_est[:,:,:,:] = hkall[0,:,:,:]    #(first version) the estimated channel matrix in the TX will be that of the RX at instant 0
    SNR_k_est[:,:] = SNR_k[0,:]                    #(first version) the estimated SNR in the TX will be that of the RX at instant 0

    #FEEDBACK (RX) + LINK ADAPTATION (TX) 
    mu = 0.2                         #(first version) coefficient of gradient descent. Is it possible to estimate it?
    marg_ini = 1                      #initial adaptation margin
    epsy = 0.05                       #BLER
    E_dB = np.zeros((Nu))       
    marg_lin = np.zeros((Nu))
    marg_lin[0] = marg_ini
    
    #first loop iteration with the initial values for calculating the tx rate
    rate_tx[0] = np.sum(np.log2(1 + SNR_k_est[0,:] * marg_lin[0]), axis = 0) * delta_f  #se usa axis = 0 porque SNR_k_est[0,:] elimina la primera dimensión
    
    for nu in range(Nu-1): 

        #compare previous TX rate with previous achievable rate RX to generate ACK 
        #If ach_rate > rate_tx rate is considered a success (0), if it fails (1)
        E_dB[nu] = int(ach_rate[nu] <= rate_tx[nu])

        #update of the current margin using the previous one (calculated in dB and passed to u.n.) 
        marg_lin[nu+1] = 10 ** (( 10*np.log10(marg_lin[nu]) - mu * (E_dB[nu] - epsy)) /10 )
    
        #calculate TX rate for the current instant 
        rate_tx[nu+1] = np.sum(np.log2(1 + SNR_k_est[nu+1,:] * marg_lin[nu+1]), axis = 0) * delta_f
    

    #print("------------------------------------------H_beamf-------------------------------------------------------")
    #print(H_beamf)
    #print(H_beamf[0,0])
    #print("---------------------------------------np.abs(H_beamf[0, :]) **2)--------------------------------------")
    #print((np.abs(H_beamf[:, :]) **2))
    #print("------------------------------------------p_tx/Nk----------------------------------------------------")
    #print(p_tx/Nk)
    #print("-------------------------------------------SNR_k-----------------------------------------------------")
    #print(SNR_k)
    print("-------------------------------------------SNR_k_dB-----------------------------------------------------")
    print(SNR_k_dB)
    print("----------------------------------------marg_lin[:]------------------------------------------------")   
    print(marg_lin[:])
    print("----------------------------------------spect_eff[:]-------------------------------------------------")
    print(spect_eff_k[:])
    print("----------------------------------------ach_rate[:]-------------------------------------------------")
    print(ach_rate[:])
    print("----------------------------------------rate_tx[:]--------------------------------------------------")
    print(rate_tx[:])
    print("----------------------     1 means ach_rate > rate_tx in (Nu-1)    ------------------------------")
    print( np.where(ach_rate[:] > rate_tx[:], 1, 0) )
    

plt.plot(np.sort(NMSEdisp[:,1],axis=0),np.linspace(0,1,Nsim))