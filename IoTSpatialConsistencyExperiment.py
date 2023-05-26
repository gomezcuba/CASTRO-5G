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
    

Nt=128
Nk=1024
Nd=16
Na=16

Nxp=4
Nrft=1
Nrfr=4

Nsim=10         

c=3e8
Ts=320e-9/Nt #2.5e-9
Nu=10
Npath=20
xmax=100
ymax=100
dmax=10
NpathFeedback=10

method = 'NpathDisplaced'

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


################## Main parameters #########################
E_dB = np.zeros((Nsim, Nu))
SNR_k = np.zeros((Nsim, Nu, Nk), dtype=np.float32)    #subcarrier SNR array
SNR_k_est = np.zeros((Nsim, Nu,Nk), dtype=np.float32) #SNR estimated by tx
rate_tx = np.zeros((Nsim, Nu),dtype = np.float32)     #tx rate 
ach_rate = np.zeros((Nsim, Nu),dtype = np.float32)    #achievable rate

marg_ini = 1                      #initial adaptation margin    
marg_lin = np.zeros((Nsim, Nu))
##############################################################################

#otro for por caso donde en cada iteración se cambia el método o el NpathFeedback
#al final for se guarda en variables para la representación de cada caso
#se hace una sola figura con los plots de cda variable

#un primer bucle for para elegir el método ------> 3 iteraciones
#un segundo for Nmeth in NpathFeedback: -------> 1 iteracion en los primeros dos casos, 3 iteraiones en el 3er caso


#--- (NpathFeedback tiene un solo elemento) 
#caso 1: CSIT perfecta
#caso 2: perfect1User

#---(NpathFeedback tiene tres elementos (20, 10, 5))
#--- hay que añadir la dimensión Nmethod para estas gráficas
#caso 3: NpathDisplaced ,  NpathFeedback = 20  
#caso 4: NpathDisplaced ,  NpathFeedback = 10
#caso 5: NpathDisplaced ,  NpathFeedback = 5

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
    hall_est=np.zeros((Nu,Nt,Na,Nd),dtype=np.complex64)
#    hall[0,:,:,:]=ht        
    for nu in range(Nu-1):    
        tdelay[nu+1,:],aod[nu+1,:],aoa[nu+1,:]=displaceMultipathChannel(tdelay[nu,:],aod[nu,:],aoa[nu,:],xstep[nu],ystep[nu])
        #for 3GPP coefs must be updated too
    clock_offset=np.minimum(np.min((tdelay-d1/c)/Ts),0)
    
    for nu in range(Nu):    #note ht was generated without clock offset and must be modified
        hall[nu,:,:,:]=chgen.computeDEC((tdelay[nu,:]-d1/c)/Ts-clock_offset,aod[nu,:],aoa[nu,:],coefs)
        hall_est[nu,:,:,:]=chgen.computeDEC((tdelay[nu,0:NpathFeedback]-d1/c)/Ts-clock_offset,aod[nu,0:NpathFeedback],aoa[nu,0:NpathFeedback],coefs[0:NpathFeedback])
        
    #print("NMSE: %s"%(  np.sum(np.abs(hall-hall_est)**2,axis=(1,2,3))/np.sum(np.abs(hall)**2,axis=(1,2,3)) ))
    hkall=np.fft.fft(hall,Nk,axis=1) 
    hkall_est=np.fft.fft(hall_est,Nk,axis=1)    

    if method =='perfectCSIT':
        hkall_est=hkall
    elif method=='perfect1User':
        hkall_est = np.tile(hkall[0,:,:,:] ,[Nu,1,1,1])
    elif method=='NpathDisplaced':
        hkall_est=np.fft.fft(hall_est,Nk,axis=1)
    else:
        print("Method not supported")   
    
    ###########################################################################################
    marg_lin[isim, 0] = marg_ini   #the adaptation margin is initialized with that of the previous simulation

    p_tx = 500e-3        #(first version )transmitter power W [can be an input parameter].
    numerology_mu = 0
    delta_f = 15e3 * (2**numerology_mu)      #carrier spacing [Hz]
    p_loss = 1e-12                           #pathloss

    Temp = 290                       # Define temperature T (in Kelvin), 17ºC
    k_boltz = 1.380649e-23           # Boltzmann's constant k [J/K]
    N0_noise = k_boltz * Temp        # Potencia de ruido W/Hz

    #Beamforming calculation
    H_beamf_max_est = np.zeros((Nu, Nk), complex)
    H_beamf_max_real = np.zeros((Nu, Nk), complex)
    beams_table = np.fft.fft(np.eye(Nd))/np.sqrt(Nd)         #table of beam vectors
    beams_table_rx = np.fft.fft(np.eye(Na))/np.sqrt(Na)         #table of rx beam vectors
     
    for nu in range(Nu):        
        #the next code shows how to include also a dictionary receiver beamforming. Notice that with for loops this would be very big code
        V = beams_table.T
        W = beams_table_rx.T
        H_beamf_all =  W.T.conj() @ hkall_est[nu,:,:,:] @ V # matrix product in numpy makes the for k automatically
        gain_all = np.mean(np.abs(H_beamf_all)**2,axis=0)
        best_ind = np.argmax(gain_all) #this index is scalar
        best_ind_rx, best_ind_tx = np.unravel_index(best_ind ,gain_all.shape)
        max_gain = gain_all[best_ind_rx,best_ind_tx]
        best_v_beamf = V[:,best_ind_tx]
        best_w_beamf = W[:,best_ind_rx]
        H_beamf_max_est[nu,:] = H_beamf_all[:,best_ind_rx,best_ind_tx]
        H_beamf_max_real[nu,:] = W[:,best_ind_rx].T.conj() @ hkall[nu,:,:,:] @ V[:,best_ind_tx]
           

    #LINK ADAPTATION parameters
    mu = 0.2                          #(first version) coefficient of gradient descent. Is it possible to estimate it?
    epsy = 0.05                       #BLER       

    #calculation of parameters at instant 0
    #TX
    SNR_k_est[isim, 0, :] = ( p_tx * p_loss * (np.abs(H_beamf_max_est[0, :]) **2) ) / ( N0_noise * Nk * delta_f ) #calculation of the estimated SNR
    rate_tx[isim, 0] = np.sum(np.log2(1 + SNR_k_est[isim,0,:] * marg_lin[isim, 0]), axis = 0) * delta_f                      ##calculation of the TX rate
    
    #RX
    SNR_k[isim, 0, :] = ( p_tx * p_loss * (np.abs(H_beamf_max_real[0, :]) **2) ) / ( N0_noise * Nk * delta_f )  #calculation of the SNR
    ach_rate[isim, 0] = np.sum(np.log2(1 + SNR_k[isim,0,:]), axis = 0) * delta_f                                       #calculation of the achievable rate
    
    #loop for parameter computation in the remaining instants and link adaptation algorithm
    for nu in range(Nu-1):                                   
         
        #compare previous TX rate with previous achievable rate RX to generate ACK 
        #If ach_rate > rate_tx rate is considered a success (0), if it fails (1)
        E_dB[isim, nu] = int(ach_rate[isim, nu] <= rate_tx[isim, nu])

        #update of the current margin using the previous one (calculated in dB and passed to u.n.)
        marg_lin[isim, nu + 1] = 10 ** (( 10*np.log10(marg_lin[isim, nu]) - mu * (E_dB[isim, nu] - epsy)) /10 ) 

        #TX
        SNR_k_est[isim, nu+1, :] = (p_tx * p_loss * (np.abs(H_beamf_max_est[nu + 1, :]) ** 2)) / (N0_noise * Nk * delta_f) #calculation of the estimated SNR
        rate_tx[isim, nu+1] = np.sum(np.log2(1 + SNR_k_est[isim, nu+1,:] * marg_lin[isim, nu+1]), axis = 0) * delta_f         #calculate TX rate for the current instant 
        
        #RX
        SNR_k[isim, nu+1, :] = ( p_tx * p_loss * (np.abs(H_beamf_max_real[nu + 1, :]) **2) ) / ( N0_noise * Nk * delta_f ) #calculation of the SNR
        ach_rate[isim, nu + 1] = np.sum(np.log2(1 + SNR_k[isim, nu+1, :]), axis = 0) * delta_f                                      #Achievable Rate Calculation       
        spect_eff_k = ach_rate[isim, nu + 1] / (Nk * delta_f)
    E_dB[isim, Nu-1] = int(ach_rate[isim, Nu-1] <= rate_tx[isim, Nu-1])
    
    marg_ini = marg_lin[isim, Nu-1] #the initial margin is updated for the following simulation
    
    SNR_k_dB = 10*np.log10(SNR_k[isim])

    #luego introducir la dimension nMetodo y hacer las gráficas 
    #E_dB[nMetodo,isim, Nu-1]

plt.figure(1)
plt.plot(np.arange(Nu* Nsim),(10 * np.log10(np.mean(SNR_k, axis=2))).flatten(),'b')
plt.plot(np.arange(Nu* Nsim),(10 * np.log10(np.mean(SNR_k_est, axis=2))).flatten(),'r')
plt.xlabel('Trajectory point')
plt.ylabel('Mean SNR (dB)')
plt.legend(['True','Estimated'])

plt.figure(2)
plt.plot(np.arange(Nu * Nsim), ach_rate.flatten() * 1e-6, 'b')
plt.plot(np.arange(Nu * Nsim), rate_tx.flatten() * 1e-6, 'r')
plt.plot(np.arange(Nu * Nsim), (1 - E_dB.flatten()) * rate_tx.flatten() * 1e-6, 'g')
plt.xlabel('Trajectory point')
plt.ylabel('Rate (Mbps)')
plt.legend(['Achievable', 'Transmitted', 'Received (tx and no err.)'])

plt.figure(3)
plt.plot(np.arange(Nu * Nsim), marg_lin.flatten() , 'b')
plt.xlabel('Trajectory point')
plt.ylabel('Value')
plt.legend(['Lineal margin for Link Adaptation'])
plt.show()