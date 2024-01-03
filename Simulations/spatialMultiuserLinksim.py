#!/usr/bin/python

from progress.bar import Bar
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# import time
import os
import argparse
plt.close('all')
import sys
sys.path.append('../')
# from CASTRO5G import threeGPPMultipathGenerator as mpg
from CASTRO5G import multipathChannel as ch
# from CASTRO5G import OMPCachedRunner as oc
# import MIMOPilotChannel as pil


class UIMultipathChannelModel:
    def __init__(self,Ncp=1,Nr=1,Nt=1):
        self.Ncp=Ncp
        self.Nr=Nr
        self.Nt=Nt
        self.Nrtoms=Ncp*Nt*Nr        
    def zAWGN(self,shape):
        return ( np.random.normal(size=shape) + 1j*np.random.normal(size=shape) ) * np.sqrt( 1 / 2.0 )
    
    def computeDEC(self,delay,AoD,AoA,coefs):
        tpulse=np.sinc(np.arange(self.Ncp).reshape(self.Ncp,1,1,1)-delay)
        arrayA=np.exp(-1j*np.pi*np.sin(AoA)*np.arange(self.Nr).reshape(1,self.Nr,1,1))
        arrayD=np.exp(-1j*np.pi*np.sin(AoD)*np.arange(self.Nt).reshape(1,1,self.Nt,1))        
        return(np.sum(coefs*tpulse*arrayA*arrayD,axis=-1))
        
    def generateDEC(self,Npoints=1):
        delay=np.random.rand(Npoints)*self.Ncp
        AoD=np.random.rand(Npoints)*np.pi*2
        AoA=np.random.rand(Npoints)*np.pi*2
        coefs=self.zAWGN(Npoints)
        coefs=coefs/np.sqrt(np.sum(np.abs(coefs)**2))
        return (coefs,delay,AoD,AoA)
    
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
    



#TODO: make the script below the __main__() of a class that can be imported by other python programs
parser = argparse.ArgumentParser(description='Multiuser Multipath Channel Displacement and Link Evaluation Simulator')
#parameters that affect simulation 
parser.add_argument('-Ns', type=int,help='No. simulated drops')
#parameters that affect user fleet
parser.add_argument('-Nu', type=int,help='No. simulated users')
#parameters that affecto OFDM physical layer dimensions
parser.add_argument('-Nt', type=int,help='No. transmitter antennas')
parser.add_argument('-Nr', type=int,help='No. receiver antennas')
parser.add_argument('-Nrft', type=int,help='No. transmitter RF chains')
parser.add_argument('-Nrfr', type=int,help='No. receiver RF chains')
parser.add_argument('-Ncp', type=int,help='No. CP samples in OFDM')
parser.add_argument('-Nk', type=int,help='No. subcarriers in OFDM')

#parameters that affect the plots
parser.add_argument('--tracerate',help='plot rate traces for each case', action='store_true')
parser.add_argument('--tracemargin',help='plot margin traces for all cases', action='store_true')
parser.add_argument('--barrate',help='plot rate traces for all cases', action='store_true')
parser.add_argument('--barpout',help='plot Pout traces for all cases', action='store_true')
parser.add_argument('--userrate',help='plot user rates for all cases', action='store_true')
parser.add_argument('--userpout',help='plot user Pout for all cases', action='store_true')

#parameters that affect workflow
parser.add_argument('--label', type=str,help='str label appended to storage files')
parser.add_argument('--nosave', help='Do not save simulation data to new results file', action='store_true')
parser.add_argument('--nompg',help='Do not perform multipath generation, load existing file', action='store_true')
parser.add_argument('--nolinksim',help='Do not perform link simulation, load existing file', action='store_true')
parser.add_argument('--show', help='Open plot figures during execution', action='store_true')
parser.add_argument('--print', help='Save plot files in eps to results folder', action='store_true')


args = parser.parse_args("--nompg --nolinksim -Ns 100 -Nu 10 -Nt 16 -Nr 16 -Nrft 1 -Nrfr 4 -Nk 128 -Ncp 128 --show --print --label test --tracerate --tracemargin --barrate --barpout --userrate --userpout".split(' '))


Ns = args.Ns if args.Ns else 2
Nu = args.Nu if args.Nu else 3  

Nt= args.Nt if args.Nt else 4
Nr= args.Nr if args.Nr else 4
Ncp= args.Ncp if args.Ncp else 32
Nk= args.Nk if args.Nk else 32
Nrft= args.Nrft if args.Nrft else 1
Nrfr= args.Nrfr if args.Nrfr else 4

if args.label:
    outfoldername="../Results/MULinkAdaptresults%s"%(args.label)
else:
    outfoldername="../Results/MULinkAdaptresults-%d-%d"%(Ns,Nu)
if not os.path.isdir(outfoldername):
    os.mkdir(outfoldername)
    
xmax=100
ymax=100
dmax=10
chgen = UIMultipathChannelModel(Ncp,Nt,Nr)

###########################################################################################
p_tx = 500e-3        #(first version )transmitter power W [can be an input parameter].
numerology_mu = 0
delta_f = 15e3 * (2**numerology_mu)      #carrier spacing [Hz]
p_loss = 1e-12                           #pathloss
Ds=320e-9
Ts=Ds/Ncp
Npath=20

c=3e8
Temp = 290                       # Define temperature T (in Kelvin), 17ºC
k_boltz = 1.380649e-23           # Boltzmann's constant k [J/K]
N0_noise = k_boltz * Temp        # Potencia de ruido W/Hz

#LINK ADAPTATION parameters
mu = 0.05                         #(first version) coefficient of gradient descent. Is it possible to estimate it?
epsy = 0.05                       #BLER       

if args.nompg:
    #load location and channel files
    data=np.load(outfoldername+'/chanGenData.npz') 
    x=data["x"]         
    y=data["y"]
    coefs=data["coefs"]
    aod=data["aod"]
    aoa=data["aoa"]
    tdoa=data["tdoa"]
    clock_offset=data["clock_offset"]
    (Ns,Nu,Npath)=aod.shape
else:
    #generate location and channels
    # generate locations
    x1=(np.random.rand(Ns)-.5)*xmax*2
    y1=(np.random.rand(Ns)-.5)*ymax*2
    d1=np.sqrt(x1**2+y1**2)
    
    dist=np.random.rand(Ns,Nu-1)*dmax
    dire=np.random.rand(Ns,Nu-1)*np.pi*2
    xstep=dist*np.cos(dire)
    ystep=dist*np.sin(dire)
    x=np.cumsum(np.concatenate((x1[:,None],xstep),axis=1),axis=1)
    y=np.cumsum(np.concatenate((y1[:,None],ystep),axis=1),axis=1)
    
    # generate multipath channels
    tdoa=np.zeros((Ns,Nu,Npath))
    aod=np.zeros((Ns,Nu,Npath))
    aoa=np.zeros((Ns,Nu,Npath))
    clock_offset=np.zeros(Ns)
    bar = Bar('Generating channels', max=Ns)
    for isim in range(Ns):            
        coefs,delay1,aod1,aoa1=chgen.generateDEC(Npath)
        delay1-np.min(delay1)        
        ord_true=np.argsort(-np.abs(coefs)**2)
        coefs=coefs[ord_true]
        delay1=delay1[ord_true]
        aod1=aod1[ord_true]
        aoa1=aoa1[ord_true]
        
        tdoa[isim,0,:]=delay1*Ts
        aod[isim,0,:]=aod1
        aoa[isim,0,:]=aoa1   
        for nu in range(Nu-1):    
            tdoa[isim,nu+1,:],aod[isim,nu+1,:],aoa[isim,nu+1,:]=displaceMultipathChannel(tdoa[isim,nu,:],aod[isim,nu,:],aoa[isim,nu,:],xstep[isim,nu],ystep[isim,nu])
            #for 3GPP coefs must be updated too
        clock_offset[isim]=np.min(tdoa[isim,:,:])
        tdoa[isim,:,:] = tdoa[isim,:,:] - clock_offset[isim]
        bar.next()
    bar.finish()
    if not args.nosave: 
        np.savez(outfoldername+'/chanGenData.npz',
                 x=x,
                 y=y,
                 coefs=coefs,
                 aod=aod,
                 aoa=aoa,
                 tdoa=tdoa,
                 clock_offset=clock_offset)
        

# Nxp=4      
NcsitCases = 3 #dimension para el número de Displaced paths que se harán las simulaciones


NpathFeedback=10

method = 'NpathDisplaced'

NMSE=np.zeros((Ns,Nu))
NMSEdisp=np.zeros((Ns,Nu))
aodFlipErrors=np.zeros(Ns)
aoaFlipErrors=np.zeros(Ns)
aodFlipWeight=np.zeros(Ns)
aoaFlipWeight=np.zeros(Ns)

bGenRand=True

# pilgen = pil.MIMOPilotChannel("IDUV")
# omprunner = oc.OMPCachedRunner()

# if bGenRand:
    # (w,v)=pilgen.generatePilots((Nk,Nxp,Nrfr,Nr,Nt,Nrft),"IDUV")

csitConfig = [
    ("perfectCSIT",None),
    ("perfect1User",None),
    ("NpathDisplaced",10),
    ("NpathDisplaced",5),
    ("NpathDisplaced",2),
    ]
testLegends = []
NcsitCases = len(csitConfig)

################## Main parameters #########################


if args.nolinksim: 
    data=np.load(outfoldername+'/LinkAdaptResults.npz') 
    marg_epsilon=data["marg_epsilon"]
    marg_lin=data["marg_lin"]
    SNR_k=data["SNR_k"]
    ach_rate=data["ach_rate"]
    SNR_k_est=data["SNR_k_est"]
    rate_tx=data["rate_tx"]
    E_dB=data["E_dB"]
    for nCSIT in range(NcsitCases):
        method,methodConfig = csitConfig[nCSIT]        
        if method=='NpathDisplaced':
            NpathFeedback=methodConfig
            methodLegend="%d Path"%(NpathFeedback)
        # elif method=='NOMPDisplaced':
            # methodLegend=TBD
        else:
            methodLegend=method
        testLegends.append(methodLegend) 
else:        
    E_dB = np.zeros((NcsitCases, Ns, Nu))
    SNR_k = np.zeros((NcsitCases, Ns, Nu, Nk), dtype=np.float32)    #subcarrier SNR array
    ach_rate = np.zeros((NcsitCases, Ns, Nu),dtype = np.float32)    #achievable rate
    SNR_k_est = np.zeros((NcsitCases, Ns, Nu,Nk), dtype=np.float32) #SNR estimated by tx
    rate_tx = np.zeros((NcsitCases, Ns, Nu),dtype = np.float32)     #tx rate 
    marg_epsilon = np.empty((NcsitCases, Ns , Nu),dtype = np.float32) 
    marg_lin = np.empty((NcsitCases, Ns , Nu),dtype = np.float32) 
    for nCSIT in range(NcsitCases):
        method,methodConfig = csitConfig[nCSIT]
        marg_epsilon[nCSIT,:,0] = np.zeros(Ns)
        marg_lin[nCSIT,:,0] = np.ones(Ns)
        
        if method=='NpathDisplaced':
            NpathFeedback=methodConfig
            methodLegend="%d Path"%(NpathFeedback)
        # elif method=='NOMPDisplaced':
            # methodLegend=TBD
        else:
            methodLegend=method
        testLegends.append(methodLegend) 
    
        bar = Bar('Simulating LA with CSIT type %s'%(methodLegend), max=Ns*Nu)
        for isim in range(Ns):
            
            hall=np.zeros((Nu,Ncp,Nr,Nt),dtype=np.complex64)
            hall_est=np.zeros((Nu,Ncp,Nr,Nt),dtype=np.complex64)
                #for 3GPP coefs must be updated too
            
            for nu in range(Nu):    #note ht was generated without clock offset and must be modified
                hall[nu,:,:,:]=chgen.computeDEC(tdoa[isim,nu,:]/Ts,aod[isim,nu,:],aoa[isim,nu,:],coefs)
                hall_est[nu,:,:,:]=chgen.computeDEC(tdoa[isim,nu,0:NpathFeedback]/Ts,aod[isim,nu,0:NpathFeedback],aoa[isim,nu,0:NpathFeedback],coefs[0:NpathFeedback])
                
            #priNcp("NMSE: %s"%(  np.sum(np.abs(hall-hall_est)**2,axis=(1,2,3))/np.sum(np.abs(hall)**2,axis=(1,2,3)) ))
            hkall=np.fft.fft(hall,Nk,axis=1) 
            if method =='perfectCSIT':
                hkall_est=hkall
            elif method=='perfect1User':
                hkall_est = np.tile(hkall[0,:,:,:] ,[Nu,1,1,1])
            elif method=='NpathDisplaced':
                hkall_est=np.fft.fft(hall_est,Nk,axis=1)
            else:
                print("Method not supported")   
        
            #Beamforming calculation
            G_beamf_max_est = np.zeros((Nu, Nk))
            G_beamf_max_real = np.zeros((Nu, Nk))
            beams_table = np.fft.fft(np.eye(Nt))/np.sqrt(Nt)         #table of beam vectors
            beams_table_rx = np.fft.fft(np.eye(Nr))/np.sqrt(Nr)         #table of rx beam vectors
            
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
                G_beamf_max_est[nu,:] = np.abs(H_beamf_all[:,best_ind_rx,best_ind_tx])**2
                G_beamf_max_real[nu,:] = np.abs(W[:,best_ind_rx].T.conj() @ hkall[nu,:,:,:] @ V[:,best_ind_tx])**2
    
                #TX
                SNR_k_est[nCSIT, isim, nu, :] = ( p_tx * p_loss * G_beamf_max_est[nu, :] ) / ( N0_noise * Nk * delta_f ) #calculation of the estimated SNR
                rate_tx[nCSIT, isim, nu] = np.sum(np.log2(1 + SNR_k_est[nCSIT, isim,nu,:] * marg_lin[nCSIT, isim, nu]), axis = 0) * delta_f                      ##calculation of the TX rate
                
                #RX
                SNR_k[nCSIT, isim, nu, :] = ( p_tx * p_loss * G_beamf_max_real[nu, :] ) / ( N0_noise * Nk * delta_f )  #calculation of the SNR
                ach_rate[nCSIT, isim, nu] = np.sum(np.log2(1 + SNR_k[nCSIT, isim,nu,:]), axis = 0) * delta_f 
                
                #compare previous TX rate with previous achievable rate RX to generate ACK 
                #If ach_rate > rate_tx rate is considered a success (0), if it fails (1)
                E_dB[nCSIT, isim, nu] = int(ach_rate[nCSIT, isim, nu] < rate_tx[nCSIT, isim, nu])
    
                #update of the current margin using the previous one (calculated in dB and passed to u.n.)
                if nu<(Nu-1):
                    nextS=isim
                    nextU=nu + 1
                elif isim<Ns-1:
                    nextS=isim+1
                    nextU=0
                marg_epsilon[nCSIT, nextS, nextU] = marg_epsilon[nCSIT, isim, nu] - mu * (E_dB[nCSIT, isim, nu] - epsy)
                marg_lin[nCSIT, nextS, nextU] = 10 ** ( marg_epsilon[nCSIT, nextS, nextU] /10 )
                bar.next()
        bar.finish()
    
        if not args.nosave: 
            np.savez(outfoldername+'/LinkAdaptResults.npz',
                    marg_epsilon=marg_epsilon,
                    marg_lin=marg_lin,
                    SNR_k=SNR_k,
                    ach_rate=ach_rate,
                    SNR_k_est=SNR_k_est,
                    rate_tx=rate_tx,
                    E_dB=E_dB
                    )
            

plt.close('all')
fig_ctr=0
if args.tracerate:
    for nCSIT in range(NcsitCases):
        fig_ctr+=1
        plt.figure(fig_ctr)                
        plt.plot(np.arange(Nu * Ns), ach_rate[nCSIT,:,:].flatten() * 1e-6, 'b')
        if csitConfig[nCSIT][0] != 'perfectCSIT':
            plt.plot(np.arange(Nu * Ns), rate_tx[nCSIT,:,:].flatten() * 1e-6, 'r')
            plt.plot(np.arange(Nu * Ns), (1 - E_dB[nCSIT,:,:].flatten()) * rate_tx[nCSIT,:,:].flatten() * 1e-6, 'g')
            plt.legend(['Achievable', 'Transmitted', 'Received (tx and no err.)'])
        plt.xlabel('Trajectory point')
        plt.ylabel('Rate (Mbps)')
        plt.title(testLegends[nCSIT])

if args.tracemargin:
    fig_ctr+=1
    plt.figure(fig_ctr)            
    plt.plot(np.arange(Nu * Ns), marg_lin.reshape(NcsitCases,-1).T)
    plt.xlabel('Iteration')
    plt.ylabel('Linear Margin (1 is better)')
    plt.legend(testLegends)
    plt.show()

if args.barrate:
    fig_ctr+=1
    plt.figure(fig_ctr)   
    plt.bar(np.arange(NcsitCases), np.mean((1 - E_dB)*rate_tx, axis=(1,2)) * 1e-6 ) 
    plt.xlabel('Simulations')
    plt.ylabel('Mean Achievable rate (Mbps)')
    plt.xticks(ticks=np.arange(NcsitCases),labels=testLegends)
if args.barpout:
    fig_ctr+=1
    plt.figure(fig_ctr)   
    plt.bar(np.arange(NcsitCases), np.mean(E_dB, axis=(1,2) ) )
    plt.plot(np.arange(NcsitCases)-.5, epsy*np.ones(NcsitCases), ':k') 
    plt.xlabel('Simulations')
    plt.ylabel('Mean Pout')
    plt.xticks(ticks=np.arange(NcsitCases),labels=testLegends)
    
if args.userrate:
    for nCSIT in range(NcsitCases):
        fig_ctr+=1
        plt.figure(fig_ctr)       
        plt.bar(np.arange(Nu), np.mean((1 - E_dB[nCSIT,:,:])*rate_tx[nCSIT,:,:], axis=0 )* 1e-6 ) 
        plt.xlabel('User')
        plt.ylabel('Mean Achievable rate (Mbps)')
        plt.title(testLegends[nCSIT])
if args.userpout:
    for nCSIT in range(NcsitCases):
        fig_ctr+=1
        plt.figure(fig_ctr)       
        plt.bar(np.arange(Nu), np.mean(E_dB[nCSIT,:,:], axis=0 ) )
        plt.plot(np.arange(Nu)-.5, epsy*np.ones(Nu), ':k') 
        plt.xlabel('User')
        plt.ylabel('Mean Pout')
        plt.title(testLegends[nCSIT])
