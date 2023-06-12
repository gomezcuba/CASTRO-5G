#!/usr/bin/python

from progress.bar import Bar
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import cm
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
bar = Bar('Procesando', max=3000)

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
Nd=4
Na=4

Nxp=4
Nrft=1
Nrfr=4

Nsim = 1000      
NpathMethod = 3 #dimension para el número de Displaced paths que harán las simulaciones

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
E_dB = np.zeros((NpathMethod, Nsim, Nu))
SNR_k = np.zeros((NpathMethod, Nsim, Nu, Nk), dtype=np.float32)    #subcarrier SNR array
SNR_k_est = np.zeros((NpathMethod, Nsim, Nu,Nk), dtype=np.float32) #SNR estimated by tx
rate_tx = np.zeros((NpathMethod, Nsim, Nu),dtype = np.float32)     #tx rate 
ach_rate = np.zeros((NpathMethod, Nsim, Nu),dtype = np.float32)    #achievable rate

marg_ini = np.ones((NpathMethod))                      #initial adaptation margin    
marg_lin = np.zeros((NpathMethod, Nsim, Nu))

###################### Parameters for plotting ######################################
marg_perfectCSIT = np.zeros((Nu * Nsim))
marg_perfect1User = np.zeros((Nu * Nsim))
marg_10_pathDisplaced = np.zeros((Nu * Nsim))
marg_5_pathDisplaced = np.zeros((Nu * Nsim))
marg_2_pathDisplaced = np.zeros((Nu * Nsim))

cases = [1, 2, 3]
sizePaths = [10, 5, 2]

marg_perfectCSIT_file = 'marg_perfectCSIT.npy'
marg_perfect1User_file = 'marg_perfect1User.npy'
marg_10_pathDisplaced_file = 'marg_10_pathDisplaced.npy'
marg_5_pathDisplaced_file = 'marg_5_pathDisplaced.npy'
marg_2_pathDisplaced_file = 'marg_2_pathDisplaced.npy'

RUN_SIMULATOR = True

if RUN_SIMULATOR:

    for case in cases:
        marg_ini = np.ones((NpathMethod))
        
        if case == 1:
            method = 'perfectCSIT'
        elif case == 2:
            method = 'perfect1User'
        elif case == 3:
            method = 'NpathDisplaced'

    
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
            
            for nMethod in range(NpathMethod): ############
            
                NpathFeedback = sizePaths[nMethod]

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
                marg_lin[nMethod, isim, 0] = marg_ini[nMethod]   #the adaptation margin is initialized with that of the previous simulation

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
                mu = 0.01                         #(first version) coefficient of gradient descent. Is it possible to estimate it?
                epsy = 0.01                       #BLER       

                #calculation of parameters at instant 0
                #TX
                SNR_k_est[nMethod, isim, 0, :] = ( p_tx * p_loss * (np.abs(H_beamf_max_est[0, :]) **2) ) / ( N0_noise * Nk * delta_f ) #calculation of the estimated SNR
                rate_tx[nMethod, isim, 0] = np.sum(np.log2(1 + SNR_k_est[nMethod, isim,0,:] * marg_lin[nMethod, isim, 0]), axis = 0) * delta_f                      ##calculation of the TX rate
                
                #RX
                SNR_k[nMethod, isim, 0, :] = ( p_tx * p_loss * (np.abs(H_beamf_max_real[0, :]) **2) ) / ( N0_noise * Nk * delta_f )  #calculation of the SNR
                ach_rate[nMethod, isim, 0] = np.sum(np.log2(1 + SNR_k[nMethod, isim,0,:]), axis = 0) * delta_f                                       #calculation of the achievable rate
                
                #loop for parameter computation in the remaining instants and link adaptation algorithm
                for nu in range(Nu-1):                                   
                    
                    #compare previous TX rate with previous achievable rate RX to generate ACK 
                    #If ach_rate > rate_tx rate is considered a success (0), if it fails (1)
                    E_dB[nMethod, isim, nu] = int(ach_rate[nMethod, isim, nu] <= rate_tx[nMethod, isim, nu])

                    #update of the current margin using the previous one (calculated in dB and passed to u.n.)
                    marg_lin[nMethod, isim, nu + 1] = 10 ** (( 10*np.log10(marg_lin[nMethod, isim, nu]) - mu * (E_dB[nMethod, isim, nu] - epsy)) /10 ) 

                    #TX
                    SNR_k_est[nMethod, isim, nu+1, :] = (p_tx * p_loss * (np.abs(H_beamf_max_est[nu + 1, :]) ** 2)) / (N0_noise * Nk * delta_f) #calculation of the estimated SNR
                    rate_tx[nMethod, isim, nu+1] = np.sum(np.log2(1 + SNR_k_est[nMethod, isim, nu+1,:] * marg_lin[nMethod, isim, nu+1]), axis = 0) * delta_f         #calculate TX rate for the current instant 
                    
                    #RX
                    SNR_k[nMethod, isim, nu+1, :] = ( p_tx * p_loss * (np.abs(H_beamf_max_real[nu + 1, :]) **2) ) / ( N0_noise * Nk * delta_f ) #calculation of the SNR
                    ach_rate[nMethod, isim, nu + 1] = np.sum(np.log2(1 + SNR_k[nMethod, isim, nu+1, :]), axis = 0) * delta_f                                      #Achievable Rate Calculation       
                    spect_eff_k = ach_rate[nMethod, isim, nu + 1] / (Nk * delta_f)
                E_dB[nMethod, isim, Nu-1] = int(ach_rate[nMethod, isim, Nu-1] <= rate_tx[nMethod, isim, Nu-1])
                
                marg_ini[nMethod] = marg_lin[nMethod, isim, Nu-1] #the initial margin is updated for the following simulation
                
                SNR_k_dB = 10*np.log10(SNR_k[nMethod, isim])
            
            bar.next()

        if case == 1:
            #solo quedarse con la dimensión 0 de Nmethod
            marg_perfectCSIT = marg_lin[0, :, :].flatten()

            '''
            plt.figure(1)
            plt.bar(np.arange(Nsim), np.mean(ach_rate[0], axis=1) * 1e-6 )  # Crear el gráfico de barras
            plt.xlabel('Simulations')
            plt.ylabel('Mean Achievable rate (Mbps)')

            plt.figure(2)
            plt.bar(np.arange(Nsim), np.mean((1 - E_dB[0]) * rate_tx[0] * 1e-6, axis=1))  # Crear el gráfico de barras
            plt.xlabel('Simulations')
            plt.ylabel('Mean TX rate (Mbps)')
            '''
            
            '''
            plt.figure(1)
            plt.plot(np.arange(Nu * Nsim), ach_rate[0].flatten() * 1e-6, 'b')
            plt.plot(np.arange(Nu * Nsim), rate_tx[0].flatten() * 1e-6, 'r')
            plt.plot(np.arange(Nu * Nsim), (1 - E_dB[0].flatten()) * rate_tx[0].flatten() * 1e-6, 'g')
            plt.xlabel('Trajectory point')
            plt.ylabel('Rate (Mbps)')
            plt.legend(['Achievable', 'Transmitted', 'Received (tx and no err.)'])
            '''

        elif case == 2:
            #solo quedarse con la dimensión 0 de Nmethod
            marg_perfect1User = marg_lin[0, :, :].flatten()

            '''
            plt.figure(2)
            plt.plot(np.arange(Nu * Nsim), ach_rate[0].flatten() * 1e-6, 'b')
            plt.plot(np.arange(Nu * Nsim), rate_tx[0].flatten() * 1e-6, 'r')
            plt.plot(np.arange(Nu * Nsim), (1 - E_dB[0].flatten()) * rate_tx[0].flatten() * 1e-6, 'g')
            plt.xlabel('Trajectory point')
            plt.ylabel('Rate (Mbps)')
            plt.legend(['Achievable', 'Transmitted', 'Received (tx and no err.)'])
            '''

            datos = [np.mean(ach_rate[0], axis=1) * 1e-6, 
                    np.mean((1 - E_dB[0]) * rate_tx[0] * 1e-6, axis=1)]
            datos = np.array(datos).T

            plt.figure(1)
            Nbars=2
            barwidth=0.9/Nbars
            for nbar in range(Nbars):
                plt.bar(np.arange(Nsim)+(nbar-(Nbars-1)/2)*barwidth,  datos[:,nbar],  width=barwidth,color=cm.jet(nbar/(Nbars-1)))
            plt.xlabel('Number of simulations')
            plt.ylabel('Rate (Mbps)')
            plt.legend(['Mean AR', 'Mean TP'])
            

        elif case == 3:
            #guardar cada una de las dimensiones de Nmethod
            marg_10_pathDisplaced = marg_lin[0, :, :].flatten()
            marg_5_pathDisplaced = marg_lin[1, :, :].flatten()
            marg_2_pathDisplaced = marg_lin[2, :, :].flatten()

            '''
            plt.figure(3)
            plt.bar(np.arange(Nsim), np.mean(ach_rate[0], axis=1) * 1e-6, color='magenta', label='Mean Achievable rate (Mbps)')
            plt.bar(np.arange(Nsim), np.mean((1 - E_dB[0]) * rate_tx[0] * 1e-6, axis=1), color='blue', label='Mean TX rate (Mbps)')
            plt.xlabel('Simulations')
            plt.legend(['Mean Achievable rate (Mbps)', 'Mean TX rate (Mbps)'], loc='lower right')
            plt.ylabel('Mean rate (Mbps)')
            '''

            datos = [np.mean(ach_rate[0], axis=1) * 1e-6, 
                    np.mean((1 - E_dB[0]) * rate_tx[0] * 1e-6, axis=1)]
            datos = np.array(datos).T

            plt.figure(2)
            Nbars=2
            barwidth=0.9/Nbars
            for nbar in range(Nbars):
                plt.bar(np.arange(Nsim)+(nbar-(Nbars-1)/2)*barwidth,  datos[:,nbar],  width=barwidth,color=cm.jet(nbar/(Nbars-1)))
            plt.xlabel('Number of simulations')
            plt.ylabel('Rate (Mbps)')
            plt.legend(['Mean AR', 'Mean TP'])

            plt.figure(3)
            plt.plot(np.arange(Nu * Nsim), ach_rate[0].flatten() * 1e-6, 'b')
            plt.plot(np.arange(Nu * Nsim), rate_tx[0].flatten() * 1e-6, 'r')
            plt.plot(np.arange(Nu * Nsim), (1 - E_dB[0].flatten()) * rate_tx[0].flatten() * 1e-6, 'g')
            plt.xlabel('Trajectory point')
            plt.ylabel('Rate (Mbps)')
            plt.legend(['Achievable', 'Transmitted', 'Received (tx and no err.)'])

            plt.figure(4)
            plt.plot(np.arange(Nu * Nsim), ach_rate[2].flatten() * 1e-6, 'b')
            plt.plot(np.arange(Nu * Nsim), rate_tx[2].flatten() * 1e-6, 'r')
            plt.plot(np.arange(Nu * Nsim), (1 - E_dB[2].flatten()) * rate_tx[2].flatten() * 1e-6, 'g')
            plt.xlabel('Trajectory point')
            plt.ylabel('Rate (Mbps)')
            plt.legend(['Achievable', 'Transmitted', 'Received (tx and no err.)'])
            
            
    np.save(marg_perfectCSIT_file, marg_perfectCSIT)
    np.save(marg_perfect1User_file, marg_perfect1User)
    np.save(marg_10_pathDisplaced_file, marg_10_pathDisplaced)
    np.save(marg_5_pathDisplaced_file, marg_5_pathDisplaced)
    np.save(marg_2_pathDisplaced_file, marg_2_pathDisplaced)

    bar.finish()
    
    def moving_mean(mat,N,axis=0):
        return( np.apply_along_axis(lambda c: np.convolve(c,np.ones(N)/N,mode='same') ,axis=axis,arr=mat) )

    #in these plots we focus on the time-evolution of the simulation
    #for the sum-rate of all users

    plt.figure(5)
    mm_sum_arate = np.sum( moving_mean(ach_rate ,10,axis=1),axis=2) * 1e-6
    mm_sum_tput = np.sum( moving_mean((1 - E_dB) * rate_tx , 10,axis=1),axis=2) * 1e-6
    Ncases=3
    caseParam=[#name, npath if applicable
            # ('PerfectCSIT',0),
            # ('Perf. 1st User',0),
            ('Npath=',10),
            ('Npath=',5),
            ('Npath=',2),
        ]
    for case in range(Ncases):
        name,npath=caseParam[case]
        plt.plot(np.arange(Nsim), mm_sum_arate[case,:], ':',color=cm.jet(case/(Ncases-1)), label='sum AR %s %d'%(name,npath))
        plt.plot(np.arange(Nsim), mm_sum_tput[case,:] , '',color=cm.jet(case/(Ncases-1)), label='sum TP %s %d'%(name,npath))
    plt.legend()

    #for the average error prob of all users
    plt.figure(6)
    mm_BLER = np.mean( moving_mean(E_dB , 10,axis=1 ),axis=2)
    for case in range(Ncases):
        name,npath=caseParam[case]
        plt.plot(np.arange(Nsim), mm_BLER[case,:], '',color=cm.jet(case/(Ncases-1)), label='BLER %s %d'%(name,npath))
    plt.legend()

    #in these plots we focus on the mean performance of the simulation after convergence
    #for the per user rate
    plt.figure(7)
    barwidth=0.9/Ncases/2
    mean_arate = np.mean( ach_rate[:,Nsim//2:,:], axis=1) * 1e-6
    mean_tput = np.mean( (1 - E_dB[:,Nsim//2:,:]) * rate_tx[:,Nsim//2:,:], axis=1) * 1e-6
    for case in range(Ncases):    
        name,npath=caseParam[case]
        plt.bar(np.arange(Nu)+(2*case-1/2)*barwidth,mean_arate[case,:],color=cm.jet((2*case)/(Ncases*2-1)),width=barwidth, label='mean AR %s %d'%(name,npath))
        plt.bar(np.arange(Nu)+(2*case+1-1/2)*barwidth,mean_tput[case,:],color=cm.jet((2*case+1)/(Ncases*2-1)),width=barwidth, label='mean TP %s %d'%(name,npath))
    plt.legend()

    #for the per user error probability
    plt.figure(8)
    barwidth=0.9/Ncases
    mean_BLER = np.mean( E_dB[:,Nsim//2:,:], axis=1)
    for case in range(Ncases):    
        name,npath=caseParam[case]
        plt.bar(np.arange(Nu)+(case-1/2)*barwidth,mean_BLER[case,:],color=cm.jet((case)/(Ncases-1)),width=barwidth, label='mean BLER %s %d'%(name,npath))
    plt.legend()

    plt.figure(9)
    plt.plot(np.arange(Nu * Nsim), marg_perfectCSIT  , 'b')
    plt.plot(np.arange(Nu * Nsim), marg_perfect1User , 'g')
    plt.plot(np.arange(Nu * Nsim), marg_10_pathDisplaced , 'r')
    plt.plot(np.arange(Nu * Nsim), marg_5_pathDisplaced , 'c')
    plt.plot(np.arange(Nu * Nsim), marg_2_pathDisplaced , 'm')
    plt.xlabel('Trajectory point')
    plt.ylabel('Value')
    plt.legend(['Lineal margin for Perfect CSIT',
                'Lineal margin for Perfect CSIT of 1 user',
                'Lineal margin for 10 Displaced Paths',
                'Lineal margin for 5 Displaced Paths', 
                'Lineal margin for 2 Displaced Paths'])

    plt.figure(10)
    plt.plot(np.arange(Nu * Nsim), marg_perfectCSIT  , 'b')
    plt.plot(np.arange(Nu * Nsim), marg_10_pathDisplaced , 'r')
    plt.plot(np.arange(Nu * Nsim), marg_5_pathDisplaced , 'c')
    plt.plot(np.arange(Nu * Nsim), marg_2_pathDisplaced , 'm')
    plt.xlabel('Trajectory point')
    plt.ylabel('Value')
    plt.legend(['Lineal margin for Perfect CSIT',
                'Lineal margin for 10 Displaced Paths',
                'Lineal margin for 5 Displaced Paths', 
                'Lineal margin for 2 Displaced Paths'])

    plt.show()

else:
    marg_perfectCSIT = np.load(marg_perfectCSIT_file)
    marg_perfect1User = np.load(marg_perfect1User_file)
    marg_10_pathDisplaced = np.load(marg_10_pathDisplaced_file)
    marg_5_pathDisplaced = np.load(marg_5_pathDisplaced_file)
    marg_2_pathDisplaced = np.load(marg_2_pathDisplaced_file)


    plt.figure(10)
    plt.plot(np.arange(Nu * Nsim), marg_perfectCSIT  , 'b')
    plt.plot(np.arange(Nu * Nsim), marg_perfect1User , 'g')
    plt.plot(np.arange(Nu * Nsim), marg_10_pathDisplaced , 'r')
    plt.plot(np.arange(Nu * Nsim), marg_5_pathDisplaced , 'c')
    plt.plot(np.arange(Nu * Nsim), marg_2_pathDisplaced , 'm')
    plt.xlabel('Trajectory point')
    plt.ylabel('Value')
    plt.legend(['Lineal margin for Perfect CSIT',
                'Lineal margin for Perfect CSIT of 1 user',
                'Lineal margin for 10 Displaced Paths',
                'Lineal margin for 5 Displaced Paths', 
                'Lineal margin for 2 Displaced Paths'])

    plt.figure(11)
    plt.plot(np.arange(Nu * Nsim), marg_perfectCSIT  , 'b')
    plt.plot(np.arange(Nu * Nsim), marg_10_pathDisplaced , 'r')
    plt.plot(np.arange(Nu * Nsim), marg_5_pathDisplaced , 'c')
    plt.plot(np.arange(Nu * Nsim), marg_2_pathDisplaced , 'm')
    plt.xlabel('Trajectory point')
    plt.ylabel('Value')
    plt.legend(['Lineal margin for Perfect CSIT',
                'Lineal margin for 10 Displaced Paths',
                'Lineal margin for 5 Displaced Paths', 
                'Lineal margin for 2 Displaced Paths'])
    plt.show()




        
