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
import MultipathLocationEstimator as mploc
import threeGPPMultipathGenerator as mp3g
import multipathChannel as ch
import OMPCachedRunner as oc
import MIMOPilotChannel as pil

def fUPA( incidAngle , Nant = 16, dInterElement = .5):
    return np.exp( -2j * np.pi *  dInterElement * np.arange(Nant).reshape(Nant,1) * np.sin(incidAngle[...,None,None]) ) /np.sqrt(1.0*Nant)

def radUPA( Nangpoint, incidAngle , Nant = 16, dInterElement = .5):
    vUPA=fUPA(incidAngle,Nant,dInterElement)
    vAngle=fUPA(np.arange(0,2*np.pi,2*np.pi/Nangpoint),Nant,dInterElement)
    return (np.swapaxes(vUPA,vUPA.ndim-2,vUPA.ndim-1)[...,None,:,:]@vAngle.conj())[...,0,0]

def fitMmWaveChanAoAForLocation(x0,y0,phi0,AoD,dels):
    theta0=np.arctan(y0/x0)+np.pi*(x0<0)
    theta_dif=AoD-theta0
    l0=np.sqrt(x0**2+y0**2)
    l=l0+dels*c
    
    eta=l/l0
    SD=np.sin(theta_dif)
    CD=np.cos(theta_dif)
    
    A=(eta-CD)**2+SD**2
    B=-2*eta*SD*(eta-CD)
    C=(SD**2)*(eta**2-1)
    
    sol1= ( -B + np.sqrt( B**2 - 4*A*C  ) )/( 2*A )
    sol2= ( -B - np.sqrt( B**2 - 4*A*C  ) )/( 2*A )
    
    phi_dif=np.zeros((4,AoD.size))
    phi_dif[0,:]=np.arcsin(sol1)
    phi_dif[1,:]=np.arcsin(sol2)
    phi_dif[2,:]=np.pi-np.arcsin(sol1)
    phi_dif[3,:]=np.pi-np.arcsin(sol2)
    
    x=(y0+x0*np.tan(phi_dif-theta0))/(np.tan(AoD)+np.tan(phi_dif-theta0))
    y=x*np.tan(AoD) 
#    y=(x0-x)*np.tan(phi_dif-theta0)+y0
    dist=np.sqrt(x**2+y**2)+np.sqrt((x-x0)**2+(y-y0)**2)
    solIndx=np.argmin(np.abs(dist-l),0)
    phi_dif_final=phi_dif[solIndx,range(l.size)]
    AoA=np.mod(np.pi+theta0-phi_dif_final-phi0,2*np.pi)
    return (AoA,x[solIndx,range(l.size)],y[solIndx,range(l.size)])

def fitMmWaveChanDelForLocation(x0,y0,phi0,AoD,AoA):
    TA=np.tan(np.pi-(AoA+phi0))
    TD=np.tan(AoD)
    x=(y0+x0*TA)/(TD+TA)
    y=x*TD
    l=np.sqrt(x**2+y**2)+np.sqrt((x-x0)**2+(y-y0)**2)
    l0=np.sqrt(x0**2+y0**2)
    c=3e8
    exdel=(l-l0)/c
    return (exdel,x,y)

GEN_CHANS=False
GEN_PLOT=False
Nstrongest=40
Nmaxpaths=400
Nsims=100

EST_CHANS=False
#EST_PLOT=False
Nd=64
Na=64
Nt=256
Nxp=8
Nrft=1
Nrfr=4
K=256
Ts=300/Nt#2.5
Ds=Ts*Nt
sigma2=.01

MATCH_CHANS=False
#MATCH_PLOT=True

EST_LOCS=False
PLOT_LOCS=True

t_total_run_init=time.time()
fig_ctr=0

if GEN_CHANS:
    chgen = mp3g.ThreeGPPMultipathChannelModel()
    chgen.bLargeBandwidthOption=True

    #random locations in a 40m square
    x0=np.random.rand(Nsims)*100
    y0=np.random.rand(Nsims)*100-50
    
    #angles from locations
    theta0=np.arctan(y0/x0)+np.pi*(x0<0)
    phi0=np.random.rand(Nsims)*2*np.pi #receiver angular measurement offset
    
    #delays based on distance
    c=3e8
    l0=np.sqrt(np.abs(x0)**2+np.abs(y0)**2)
    tau0=l0/c
    
    t_start_g= time.time()
    bar = Bar("genchans", max=Nsims)
    bar.check_tty = False
    AoDbak=np.zeros((Nmaxpaths,Nsims))
    AoD=np.zeros((Nmaxpaths,Nsims))
    AoA=np.zeros((Nmaxpaths,Nsims))
    refPos=np.zeros((Nmaxpaths,Nsims,2))
    dels=np.zeros((Nmaxpaths,Nsims))
    coefs=np.zeros((Nmaxpaths,Nsims),dtype=np.complex)
    for nsim in range(Nsims):    
        mpch = chgen.create_channel((0,0,10),(x0[nsim],y0[nsim],1.5))
        amps = np.array([x.complexAmplitude[0] for x in mpch.channelPaths])
        allaoa_shifted = np.mod( np.array([x.azimutOfArrival[0] for x in mpch.channelPaths])-phi0[nsim] ,np.pi*2)
        allaod = np.mod( np.array([x.azimutOfDeparture[0] for x in mpch.channelPaths]) ,np.pi*2)
        alldelay = np.array([x.excessDelay[0] for x in mpch.channelPaths])*1e-9    
       
        (allaoa_shifted,xpos,ypos) = fitMmWaveChanAoAForLocation(x0[nsim],y0[nsim],phi0[nsim],allaod,alldelay) 
    #  
        
        Npaths=np.minimum(Nmaxpaths,amps.size)
        #the paths in backlobes are removed from the channel, receiver Sector may be 
        indbacklobeD=((allaod>np.pi/2)&(allaod<np.pi*3/2))
        indbacklobeA=((allaoa_shifted>np.pi/2)&(allaoa_shifted<np.pi*3/2))
        indbacklobeDir=indbacklobeD|indbacklobeA
        indbacklobeInv=indbacklobeD|~indbacklobeA
        if np.sum(np.abs(amps[~indbacklobeInv])**2)>np.sum(np.abs(amps[~indbacklobeDir])**2):
            allaoa_shifted = np.mod( allaoa_shifted + np.pi , np.pi*2)
            phi0[nsim] = np.mod( np.pi + phi0[nsim] , 2*np.pi)
            indbacklobe=indbacklobeInv
        else:
            indbacklobe=indbacklobeDir
        amps[indbacklobe]=0
            
    #    strongestInds=np.argpartition(-np.abs(amps),Npaths,axis=0)[0:Npaths]
        strongestInds=np.argsort(-np.abs(amps),axis=0)[0:Npaths]
        #get complex coefficients from model
        coefs[0:Npaths,nsim]=amps[strongestInds[0:Npaths]]
        #get delay in ns form channel model
        dels[0:Npaths,nsim] = alldelay[strongestInds[0:Npaths]]
        #get AoD form channel model
        AoD[0:Npaths,nsim] = allaod[strongestInds[0:Npaths]]
        #get AoA form channel model
        AoA[0:Npaths,nsim] = allaoa_shifted[strongestInds[0:Npaths]]
        refPos[0:Npaths,nsim,0] = xpos[strongestInds[0:Npaths]]
        refPos[0:Npaths,nsim,1] = ypos[strongestInds[0:Npaths]]
        
    #        #Assuming 2 sectors, we mirror-refleft the AoDs and AoAs associated with back lobe
    #    AoDbak[:,nsim]=AoD[:,nsim]
    #    indAbacklobeD=((AoD[:,nsim]>np.pi/2)&(AoD[:,nsim]<np.pi*3/2))
    #    AoD[indAbacklobeD,nsim]=np.mod( np.pi-AoD[indAbacklobeD,nsim]    , 2*np.pi)
    #    indAbacklobeA=((AoA[:,nsim]>np.pi/2)&(AoA[:,nsim]<np.pi*3/2))
    #    AoA[indAbacklobeA,nsim]=np.mod( np.pi-AoA[indAbacklobeA,nsim]    , 2*np.pi)
            
        #This assumes we cannot use the del form channel model, as this will violate the trigonometric equations of location and ray tracing
        #so we compute a new set of dels according to ray tracing trigonometric identities and the existing AoDs and AoAs
        #this WILL NOT violate the previous "no back lobes" claim in the AoA but not in the AoD
    #    (dels[:,nsim],refPos[:,nsim,0],refPos[:,nsim,1]) = fitMmWaveChanDelForLocation(x0[nsim],y0[nsim],phi0[nsim],AoD[:,nsim],AoA[:,nsim])    
        
        #This assumes we cannot use the AoA form channel model, as this will violate the trigonometric equations of location and ray tracing
        #so we compute a new set of AoAs according to ray tracing trigonometric identities and the existing AoDs and delays
        #this WILL violate the previous "no back lobes" claim in the AoA but not in the AoD
    #    (AoA[:,nsim],refPos[:,nsim,0],refPos[:,nsim,1]) = fitMmWaveChanAoAForLocation(x0[nsim],y0[nsim],phi0[nsim],AoD[:,nsim],dels[:,nsim]) 
    #        
        #finaly, if any of these was updated, store it back in the multipath channel object
        for pi in range(Npaths):
            chgen.dChansGenerated[(0,0,x0[nsim],y0[nsim])].channelPaths[strongestInds[pi]].azimutOfDeparture=AoD[pi,nsim]
            chgen.dChansGenerated[(0,0,x0[nsim],y0[nsim])].channelPaths[strongestInds[pi]].azimutOfArrival=AoA[pi,nsim]
            chgen.dChansGenerated[(0,0,x0[nsim],y0[nsim])].channelPaths[strongestInds[pi]].excessDelay=dels[pi,nsim]*1e9
            chgen.dChansGenerated[(0,0,x0[nsim],y0[nsim])].channelPaths[strongestInds[pi]].complexAmplitude=coefs[pi,nsim]
        
        bar.next()
    bar.finish() 
    t_run_g = time.time() - t_start_g

    np.savez('./mimochans-%d.npz'%(Nsims),
             x0=x0,
             y0=y0,
             phi0=phi0,
             refPos=refPos,
             AoA=AoA,
             AoD=AoD,
             dels=dels,
             coefs=coefs)
else:
    data=np.load('./mimochans-%d.npz'%(Nsims))
    x0=data["x0"]
    y0=data["y0"]
    phi0=data["phi0"]
    refPos=data["refPos"]
    AoA=data["AoA"]
    AoD=data["AoD"]
    dels=data["dels"]
    coefs=data["coefs"]
    
    chgen = mp3g.ThreeGPPMultipathChannelModel()
    chgen.bLargeBandwidthOption=True
    
    for nsim in range(Nsims):
        Npaths=np.sum(np.abs(coefs[:,nsim])>0,axis=0)
        lp=[]
        for npath in range(Npaths): 
            lp.append( ch.ParametricPath(coefs[npath,nsim],1e9*dels[npath,nsim],AoD[npath,nsim],AoA[npath,nsim],0,0,0) )
            
        chgen.dChansGenerated[(0,0,x0[nsim],y0[nsim])]=ch.MultipathChannel((0,0,10),(x0[nsim],y0[nsim],1.5),lp)
    

if GEN_PLOT:
    
    fig_ctr+=1
    plt.figure(fig_ctr)
    plt.plot(np.vstack((np.zeros_like(refPos[:,0,0]),refPos[:,0,0],x0[0]*np.ones_like(refPos[:,0,0]))),np.vstack((np.zeros_like(refPos[:,0,1]),refPos[:,0,1],y0[0]*np.ones_like(refPos[:,0,1]))),':xb')
    plt.plot(x0[0],y0[0],'or')
    plt.plot(0,0,'sr')
    
    plt.plot(x0[0]+np.sin(phi0[0])*5*np.array([-1,1]),y0[0]-np.cos(phi0[0])*5*np.array([-1,1]),'r-')
    plt.plot(x0[0]+np.cos(phi0[0])*5*np.array([0,1]),y0[0]+np.sin(phi0[0])*5*np.array([0,1]),'g-')
    
    plt.axis([0,120,-60,60])
    
    bfil=(coefs[:,0]==0)#((AoA[:,0])>np.pi/2)&((AoA[:,0])<3*np.pi/2)
    plt.plot(np.vstack((np.zeros_like(refPos[bfil,0,0]),refPos[bfil,0,0],x0[0]*np.ones_like(refPos[bfil,0,0]))),np.vstack((np.zeros_like(refPos[bfil,0,1]),refPos[bfil,0,1],y0[0]*np.ones_like(refPos[bfil,0,1]))),':sk')#print(((AoA-phi0)>np.pi)&((AoA-phi0)<3*np.pi))

if EST_CHANS:
    #channel multipath estimation outputs with error
    
#    angleNoiseMSE=(2*np.pi/64)**2
#    AoD_err = np.mod(AoD+np.random.randn(Nmaxpaths,Nsims)*np.sqrt(angleNoiseMSE),2*np.pi)
#    AoA_err = np.mod(AoA+np.random.randn(Nmaxpaths,Nsims)*np.sqrt(angleNoiseMSE),2*np.pi)
    #AoD_err = np.mod(AoD+np.random.rand(Nstrongest,Nsims)*2*np.pi/128,2*np.pi)
    #AoA_err = np.mod(AoA+np.random.rand(Nstrongest,Nsims)*2*np.pi/128,2*np.pi)
#    clock_error=(40/c)*np.random.rand(1,Nsims) #delay estimation error
#    dels_err = dels+clock_error
    
    t_start_cs = time.time()
    Nchan=Nsims
    omprunner = oc.OMPCachedRunner()
    pilgen = pil.MIMOPilotChannel("UPhase")
    AoD_est=np.zeros((Nstrongest,Nsims))
    AoA_est=np.zeros((Nstrongest,Nsims))
    dels_est=np.zeros((Nstrongest,Nsims))
    coef_est=np.zeros((Nstrongest,Nsims),dtype=np.complex)
    bar = Bar("estchans", max=Nsims)
    bar.check_tty = False
    MSEOMP=np.zeros(Nsims)
    for nsim in range(Nsims):    
        mpch=chgen.dChansGenerated[(0,0,x0[nsim],y0[nsim])]
        (w,v)=pilgen.generatePilots((K,Nxp,Nrfr,Na,Nd,Nrft),"UPhase")
        zp=(np.random.randn(K,Nxp,Na,1)+1j*np.random.randn(K,Nxp,Na,1))/np.sqrt(2)
        ht=mpch.getDEC(Na,Nd,Nt,Ts)*np.sqrt(Nd*Na)#mpch uses normalized matrices of gain 1
        hk=np.fft.fft(ht.transpose([2,0,1]),K,axis=0)
        yp=pilgen.applyPilotChannel(hk,w,v,zp*np.sqrt(sigma2))
        Pfa=1e-5
        factor_Pfa=np.log(1/(1-(Pfa)**(1/(Nd*Na*Nt))))
        ( hest, Isupp )=omprunner.OMPBR(yp,sigma2*K*Nxp*Nrfr*factor_Pfa,nsim,v,w, Xt=1.0, Xd=1.0, Xa=1.0, Xmu=5.0,  accelDel = True)
        MSEOMP[nsim]=np.sum(np.abs(hest-hk)**2)/np.sum(np.abs(hk)**2)
    #    a_est = np.linalg.lstsq(Isupp.observations,yp.reshape(-1,1),rcond=None)[0]
        a_est = Isupp.coefs
        if Nstrongest<a_est.size:
    #        indStrongest=np.argpartition(-np.abs(a_est[:,0]),Nstrongest,axis=0)[0:Nstrongest]
            indStrongest=np.argsort(-np.abs(a_est[:,0]),axis=0)[0:Nstrongest]
            AoD_est[:,nsim]=np.array(Isupp.AoDs)[indStrongest]
            AoA_est[:,nsim]=np.array(Isupp.AoAs)[indStrongest]
            dels_est[:,nsim]=np.array(Isupp.delays)[indStrongest]*Ts*1e-9
            coef_est[:,nsim]=np.array(Isupp.coefs)[indStrongest].reshape(-1)/np.sqrt(Na*Nd*Nt)
        else:
            Nelems=a_est.size
            indStrongest=np.argsort(-np.abs(a_est[:,0]),axis=0)
            AoD_est[0:Nelems,nsim]=np.array(Isupp.AoDs)[indStrongest]
            AoA_est[0:Nelems,nsim]=np.array(Isupp.AoAs)[indStrongest]
            dels_est[0:Nelems,nsim]=np.array(Isupp.delays)[indStrongest]*Ts*1e-9
            coef_est[0:Nelems,nsim]=np.array(Isupp.coefs).reshape(-1)[indStrongest]/np.sqrt(Na*Nd*Nt)
        omprunner.freeCacheOfPilot(nsim,Nt,Nd,Na,Xt=1.0,Xd=1.0,Xa=1.0)
        bar.next()
    bar.finish()
    
    AoA_est=np.mod(AoA_est,np.pi*2)
    AoD_est=np.mod(AoD_est,np.pi*2)
    
    t_run_cs = time.time() - t_start_cs
    
    print("OMP MSEs: %s"%(MSEOMP))

    MSE_partial=np.zeros(Isupp.coefs.size)
    #MSE_partial2=np.zeros(Isupp.coefs.size)
    for ns in range(0,Isupp.coefs.size):
        hest_partial=Isupp.outputs[:,0:(ns+1)]@Isupp.coefs[0:(ns+1),:]
        MSE_partial[ns]=np.sum(np.abs(hest_partial.reshape(Nt,Na,Nd)-hk)**2)/np.sum(np.abs(hk)**2)

    np.savez('./mimoestimschans-%d-%d-%d-%d-%d.npz'%(Nd,Na,Nt,Nxp,Nsims),
             AoA_est=AoA_est,
             AoD_est=AoD_est,
             dels_est=dels_est,
             coef_est=coef_est)
else:
    data=np.load('./mimoestimschans-%d-%d-%d-%d-%d.npz'%(Nd,Na,Nt,Nxp,Nsims))
    AoA_est=data["AoA_est"]
    AoD_est=data["AoD_est"]
    dels_est=data["dels_est"]
    coef_est=data["coef_est"]

#REFINAMENTO
#AoD_ref = sage(AOD_est)
AoD_ref=AoD_est

#CALCULO POSICIOM CON REFINAMENTO EXIP

if MATCH_CHANS:
    pdist=np.zeros((Nsims,Nmaxpaths,Nstrongest))
    for nsim in range(Nsims):
        pdist[nsim,:,:]= ( 
                np.abs(np.sin(AoD_est[:,nsim:nsim+1]).T-np.sin(AoD[:,nsim:nsim+1]))/2 +
                np.abs(np.sin(AoA_est[:,nsim:nsim+1]).T-np.sin(AoA[:,nsim:nsim+1]))/2 +
                np.abs(dels_est[:,nsim:nsim+1].T-dels[:,nsim:nsim+1])*1e9/Ts/Nt +
                np.abs(coef_est[:,nsim:nsim+1].T-coefs[:,nsim:nsim+1])**2*(1/np.abs(coef_est[:,nsim:nsim+1].T)**2+1/np.abs(coefs[:,nsim:nsim+1])**2)/np.sum(np.abs(coef_est[:,nsim:nsim+1])>0)
                )
    pathMatchTable = np.argmin(pdist,axis=1)
    AoA_diff=np.zeros((Nstrongest,Nsims))
    AoD_diff=np.zeros((Nstrongest,Nsims))
    dels_diff=np.zeros((Nstrongest,Nsims))
    coef_diff=np.zeros((Nstrongest,Nsims),dtype=np.complex)
    for nsim in range(Nsims):
        AoA_diff[:,nsim]=np.mod(AoA[pathMatchTable[nsim,:],nsim],2*np.pi)-np.mod(AoA_est[:,nsim],2*np.pi)
        AoD_diff[:,nsim]=np.mod(AoD[pathMatchTable[nsim,:],nsim],2*np.pi)-np.mod(AoD_est[:,nsim],2*np.pi)
        dels_diff[:,nsim]=dels[pathMatchTable[nsim,:],nsim]-dels_est[:,nsim]
        coef_diff[:,nsim]=coefs[pathMatchTable[nsim,:],nsim]-coef_est[:,nsim]
    Nlines=10
    fig_ctr+=1
    plt.figure(fig_ctr)
    plt.plot(10*np.log10(np.sort(np.abs(coef_diff[0:Nlines,:])**2/np.abs(coef_est[0:Nlines,:])**2,axis=1).T),np.arange(0,1,1/Nsims))
    
    fig_ctr+=1
    plt.figure(fig_ctr)
    plt.plot(10*np.log10(np.sort(np.abs(AoA_diff[0:Nlines,:]),axis=1).T),np.arange(0,1,1/Nsims))
    
    fig_ctr+=1
    plt.figure(fig_ctr)
    plt.plot(10*np.log10(np.sort(np.abs(AoD_diff[0:Nlines,:]),axis=1).T),np.arange(0,1,1/Nsims))
    
    fig_ctr+=1
    plt.figure(fig_ctr)
    plt.plot(10*np.log10(np.sort(np.abs(dels_diff[0:Nlines,:]*1e9/Ts),axis=1).T),np.arange(0,1,1/Nsims))
    
    fig_ctr+=1
    fig=plt.figure(fig_ctr)
    ax = Axes3D(fig)
    t=np.linspace(0,2*np.pi,100)
    for dbref in range(6):
        ax.plot3D(60*(1-dbref/6)*np.cos(t),60*(1-dbref/6)*np.sin(t),np.zeros_like(t),color='k')
        ax.text3D(0,-60*(1-dbref/6),0,'%s dB'%(-10*dbref),color='k')        
    t=np.linspace(0,62,100)
    for labeltheta in np.arange(0,2*np.pi,2*np.pi/8):
        ax.plot3D(t*np.cos(labeltheta),t*np.sin(labeltheta),-np.ones_like(t),color='k')
        ax.text3D(62*np.cos(labeltheta),62*np.sin(labeltheta),-1,'%.2f pi'%(labeltheta/np.pi),color='k')    
    ax.text3D(62*np.cos(np.pi/16),62*np.sin(np.pi/16),-1,'AoA',color='k')
#    maxdel=np.max(dels)*1e9
    ax.plot3D([0,0],[0,0],[0,Nt],color='k')
    ax.text3D(0,0,Nt,"delay [ns]",color='k')
    Nang=250
    angbase=np.arange(0,2*np.pi,2*np.pi/Nang)
    nsim=0
    for pind in range(0,np.sum(np.abs(coef_est[:,nsim])>0)):
        ang=AoA[pathMatchTable[nsim,pind],nsim]
        delay=dels[pathMatchTable[nsim,pind],nsim]*1e9/Ts
        gain=np.abs(coefs[pathMatchTable[nsim,pind],nsim])**2    
        x=np.maximum(10*np.log10(gain)+60,0)*np.cos(ang)
        y=np.maximum(10*np.log10(gain)+60,0)*np.sin(ang)
        p=ax.plot3D([0,x],[0,y],[delay,delay])
        ax.scatter3D(x,y,delay,marker='o',color=p[-1].get_color())    
        gainrad=np.abs(radUPA(Nang,ang,Na,.5)*coefs[pathMatchTable[nsim,pind],nsim])**2
        xrad=np.maximum(10*np.log10(gainrad)+60,0)*np.cos(angbase)
        yrad=np.maximum(10*np.log10(gainrad)+60,0)*np.sin(angbase)
        ax.plot3D(xrad,yrad,delay*np.ones_like(xrad),'-.',color=p[-1].get_color())
        
        ang=AoA_est[pind,nsim]
        delay=dels_est[pind,nsim]*1e9/Ts
        gain=np.abs(coef_est[pind,nsim])**2    
        x=np.maximum(10*np.log10(gain)+60,0)*np.cos(ang)
        y=np.maximum(10*np.log10(gain)+60,0)*np.sin(ang)
        ax.plot3D([0,x],[0,y],[delay,delay],color=p[-1].get_color())
        ax.scatter3D(x,y,delay,marker='x',color=p[-1].get_color())      
        gainrad=np.abs(radUPA(Nang,ang,Na,.5)*coef_est[pind,nsim])**2
        xrad=np.maximum(10*np.log10(gainrad)+60,0)*np.cos(angbase)
        yrad=np.maximum(10*np.log10(gainrad)+60,0)*np.sin(angbase)
        ax.plot3D(xrad,yrad,delay*np.ones_like(xrad),':',color=p[-1].get_color())   

if EST_LOCS:
    
    configTable=[
            #location method, user phi0 coarse hint initialization
            ('CS','bisec','',':','*','b'),
            ('CS','fsolve','No Hint',':','x','g'),
            ('CS','fsolve_linear','No hint','--','s','g'),
            ('CS','fsolve','Hint',':','+','m'),
            ('CS','fsolve_linear','Hint','--','d','m'),
            ('CS','oracle','Pure','-','p','r'),            
            ('CS','oracle','Hint','-.','p','y'),            
            ('true','bisec','',':','*','b'),
            ('true','fsolve','No Hint',':','x','g'),
            ('true','fsolve_linear','No hint','--','s','g'),
            ('true','fsolve','Hint',':','+','m'),
            ('true','fsolve_linear','Hint','--','d','m'),
            ('true','oracle','Pure','-','p','r'),            
            ('true','oracle','Hint','-.','p','y'),            
    ]
    Nconfigs=len(configTable)
    NpathsRetrieved=np.sum(np.abs(coef_est)**2>0,axis=0)    
    
#    NpathsRetrieved=np.zeros(Nsims,dtype=np.int)
#    for nsim in range(Nsims):
#        NpathsRetrieved[nsim]=np.sum(np.abs(coef_est[:,nsim])**2>sigma2,axis=0)
#        NpathsRetrieved[nsim]=np.where(np.cumsum(np.abs(coef_est[:,nsim])**2,axis=0)/np.sum(np.abs(coef_est[:,nsim])**2,axis=0)>.75)[0][0]
#        NpathsRetrieved[nsim]=np.sum(coef_est[:,nsim]!=0)
    loc=mploc.MultipathLocationEstimator(Npoint=1000,Nref=20,Ndiv=2,RootMethod='lm')
    t_start_all=np.zeros(Nconfigs)
    t_end_all=np.zeros(Nconfigs)
    phi0_est=np.zeros((Nconfigs,Nstrongest-3,Nsims))
    x0_est=np.zeros((Nconfigs,Nstrongest-3,Nsims))
    y0_est=np.zeros((Nconfigs,Nstrongest-3,Nsims))
    x_est=np.zeros((Nconfigs,Nstrongest-3,Nstrongest,Nsims))
    y_est=np.zeros((Nconfigs,Nstrongest-3,Nstrongest,Nsims))
    phi0_coarse=np.round(phi0*32/np.pi/2)*np.pi*2/32
    for ncfg in range(Nconfigs):
        cfg = configTable[ncfg]
        t_start_all[ncfg]=time.time()
        bar = Bar("%s %s %s"%(cfg[0:3]), max=Nsims)
        bar.check_tty = False
        for nsim in range(Nsims):
            for Nstimpaths in range(3,NpathsRetrieved[nsim]):
                if cfg[0]=='CS':
                    if cfg[1]=='oracle':
                        if cfg[2]=='Hint':
                            phi0_est[ncfg,Nstimpaths-3,nsim]=phi0_coarse[nsim]
                        else:
                            phi0_est[ncfg,Nstimpaths-3,nsim]=phi0[nsim]
                        (
                                x0_est[ncfg,Nstimpaths-3,nsim],
                                y0_est[ncfg,Nstimpaths-3,nsim],
                                _,
                                x_est[ncfg,Nstimpaths-3,0:Nstimpaths,nsim],
                                y_est[ncfg,Nstimpaths-3,0:Nstimpaths,nsim]
                        ) = loc.computeAllPathsLinear(
                                AoD_est[0:Nstimpaths,nsim],
                                AoA_est[0:Nstimpaths,nsim],
                                dels_est[0:Nstimpaths,nsim],
                                phi0_est[ncfg,Nstimpaths-3,nsim])
                    else:
                        if (cfg[1]!='fsolve_linear') or (nsim!=47):
                            (
                                    phi0_est[ncfg,Nstimpaths-3,nsim],
                                    x0_est[ncfg,Nstimpaths-3,nsim],
                                    y0_est[ncfg,Nstimpaths-3,nsim],
                                    x_est[ncfg,Nstimpaths-3,0:Nstimpaths,nsim],
                                    y_est[ncfg,Nstimpaths-3,0:Nstimpaths,nsim]
                            ) = loc.computeAllLocationsFromPaths(
                                    AoD_est[0:Nstimpaths,nsim],
                                    AoA_est[0:Nstimpaths,nsim],
                                    dels_est[0:Nstimpaths,nsim],
                                    method=cfg[1],
                                    hint_phi0= ( phi0_coarse[nsim] if (cfg[2]=='Hint') else None ) )
                else:
                    if cfg[1]=='oracle':
                        if cfg[2]=='Hint':
                            phi0_est[ncfg,Nstimpaths-3,nsim]=phi0_coarse[nsim]
                        else:
                            phi0_est[ncfg,Nstimpaths-3,nsim]=phi0[nsim]
                        (
                                x0_est[ncfg,Nstimpaths-3,nsim],
                                y0_est[ncfg,Nstimpaths-3,nsim],
                                _,
                                x_est[ncfg,Nstimpaths-3,0:Nstimpaths,nsim],
                                y_est[ncfg,Nstimpaths-3,0:Nstimpaths,nsim]
                        ) = loc.computeAllPathsLinear(
                                AoD[0:Nstimpaths,nsim],
                                AoA[0:Nstimpaths,nsim],
                                dels[0:Nstimpaths,nsim],
                                phi0_est[ncfg,Nstimpaths-3,nsim])
                    else:
                        if (cfg[1]!='fsolve_linear') or (nsim!=47):
                            (
                                    phi0_est[ncfg,Nstimpaths-3,nsim],
                                    x0_est[ncfg,Nstimpaths-3,nsim],
                                    y0_est[ncfg,Nstimpaths-3,nsim],
                                    x_est[ncfg,Nstimpaths-3,0:Nstimpaths,nsim],
                                    y_est[ncfg,Nstimpaths-3,0:Nstimpaths,nsim]
                            ) = loc.computeAllLocationsFromPaths(
                                    AoD[0:Nstimpaths,nsim],
                                    AoA[0:Nstimpaths,nsim],
                                    dels[0:Nstimpaths,nsim],
                                    method=cfg[1],
                                    hint_phi0= ( phi0_coarse[nsim] if (cfg[2]=='Hint') else None ) )
            if NpathsRetrieved[nsim]<Nstrongest:
                phi0_est[ncfg,NpathsRetrieved[nsim]-2:,nsim]=phi0_est[ncfg,NpathsRetrieved[nsim]-3,nsim]
                x0_est[ncfg,NpathsRetrieved[nsim]-2:,nsim]=x0_est[ncfg,NpathsRetrieved[nsim]-3,nsim]
                y0_est[ncfg,NpathsRetrieved[nsim]-2:,nsim]=y0_est[ncfg,NpathsRetrieved[nsim]-3,nsim]
                x_est[ncfg,NpathsRetrieved[nsim]-2:,0:Nstimpaths,nsim]=x_est[ncfg,NpathsRetrieved[nsim]-3,0:Nstimpaths,nsim]
                x_est[ncfg,NpathsRetrieved[nsim]-2:,0:Nstimpaths,nsim]=y_est[ncfg,NpathsRetrieved[nsim]-3,0:Nstimpaths,nsim]
            bar.next()
        bar.finish()
        t_end_all[ncfg]=time.time()
        np.savez('./mimoestimslocs-%d-%d-%d-%d-%d.npz'%(Nd,Na,Nt,Nxp,Nsims),
                t_start_all=t_start_all,
                t_end_all=t_end_all,
                phi0_est=phi0_est,
                x0_est=x0_est,
                y0_est=y0_est,
                x_est=x_est,
                y_est=y_est,
                configTable=configTable)
else:
    data=np.load('./mimoestimslocs-%d-%d-%d-%d-%d.npz'%(Nd,Na,Nt,Nxp,Nsims))
    t_start_all=data["t_start_all"]
    t_end_all=data["t_end_all"]
    phi0_est=data["phi0_est"]
    x0_est=data["x0_est"]
    y0_est=data["y0_est"]
    x_est=data["x_est"]
    y_est=data["y_est"]
    configTable=data["configTable"]
    Nconfigs=len(configTable)
    NpathsRetrieved=np.sum(np.abs(coef_est)**2>0,axis=0)    
    phi0_est=np.mod(phi0_est,2*np.pi)
    phi0_coarse=np.round(phi0*32/np.pi/2)*np.pi*2/32

if PLOT_LOCS:
    
    fig_ctr+=1
    plt.figure(fig_ctr)
    error_dist=np.sqrt( (x0_est - x0)**2 + (y0_est - y0)**2 )
    npathbest=np.argmin(error_dist,axis=1)
    error_min=np.min(error_dist,axis=1)
    for ncfg in range(Nconfigs):
        cfg = configTable[ncfg]
        plt.semilogx(np.sort(error_min[ncfg,:]),np.linspace(0,1,Nsims),cfg[3]+cfg[5])
    
    x0_guess=np.random.rand(Nsims)*100
    y0_guess=np.random.rand(Nsims)*100-50
    
    error_guess=np.sqrt(np.abs(x0-x0_guess)**2+np.abs(y0-y0_guess))
    plt.semilogx(np.sort(error_guess).T,np.linspace(0,1,error_guess.size),':k')
    plt.legend(["%s %s %s"%(x[0],x[1].replace("_"," "),x[2]) for x in configTable]+['random guess'])
    1
    fig_ctr+=1
    plt.figure(fig_ctr)
    labstrFilter=["bisec ", "fsolve linear No hint", "fsolve linear Hint", "oracle Hint"]
    for ncfg in range(Nconfigs):
        cfg = configTable[ncfg]
        labstr = "%s %s %s"%(cfg[0],cfg[1].replace("_"," "),cfg[2])
        if labstr in labstrFilter:
            pcatch = plt.plot(np.vstack((x0,x0_est[ncfg,npathbest[ncfg,:],range(Nsims)])),np.vstack((y0,y0_est[ncfg,npathbest[ncfg,:],range(Nsims)])),':'+cfg[4]+cfg[5], mfc='none', label=labstr)
            plt.setp(pcatch[1:],label="_")

    plt.plot(x0,y0,'ok')
    plt.axis([0, 100, -50, 50])
    plt.legend()
    
    fig_ctr+=1
    plt.figure(fig_ctr)
    labstrFilter=["CS bisec ", "CS fsolve linear No hint", "CS fsolve linear Hint"]
    phi0_err = np.minimum(
            np.mod(np.abs(phi0-phi0_est),np.pi*2),
            np.mod(np.abs(phi0+phi0_est-2*np.pi),np.pi*2)
            )
    phi0_eatmin = np.zeros((Nconfigs,Nsims))
    for ncfg in range(Nconfigs):
        cfg = configTable[ncfg]
        phi0_eatmin[ncfg,:]=phi0_err[ncfg,npathbest[ncfg,:],range(Nsims)]
        labstr = "%s %s %s"%(cfg[0],cfg[1].replace("_"," "),cfg[2])
        if labstr in labstrFilter:
            plt.loglog(phi0_eatmin[ncfg,:],error_min[ncfg,:],cfg[4]+cfg[5])
    plt.axis([0,np.pi,0,150])
    plt.legend(labstrFilter)

    def noiseThreshold(coef_est,param):
        return np.sum(np.abs(coef_est)**2>param[0],axis=0)
    def boostedNoiseThreshold(coef_est,param):
        Npathfrac=1-NpathsRetrieved/Nt#TODO: make these variables accessible without being global
        AvgNoBoost = -Npathfrac*np.log(1-Npathfrac)
        return np.sum(np.abs(coef_est)**2>AvgNoBoost*param[0],axis=0)
    def percentTotPowerThreshold(coef_est,param):
        return np.sum(np.cumsum(np.abs(coef_est)**2/np.sum(np.abs(coef_est)**2,axis=0),axis=0)<param[0],axis=0)
    def percentMaxPowerThreshold(coef_est,param):
        return np.sum(np.abs(coef_est)**2/np.max(np.abs(coef_est)**2,axis=0)>param[0],axis=0)
    def useAllThreshold(coef_est,param):
        return NpathsRetrieved
    def usePhi0Err(coef_est,param):
        return np.sum(np.abs(phi0_est[4,:,:]-phi0_coarse)<2*np.pi/32,axis=0)
#        return 37-np.argmax(np.abs(phi0_est[4,::-1,:]-phi0_coarse)<2*np.pi/32,axis=0)
    
#       fig_ctr+=1    
#    plt.figure(fig_ctr)
#    for perct in np.arange(0.64,.96,0.02):
#        npaths_method=np.maximum(np.minimum( percentTotPowerThreshold(coef_est,[perct]) -3,36),0)
#        plt.semilogx(np.sort(error_dist[4,npaths_method,range(Nsims)]),np.linspace(0,1,Nsims),'--x',color=(.2,(perct-.64)/.32,1-(perct-.64)/.32),label='Lin %.2f'%perct)
#        plt.semilogx(np.sort(error_dist[5,npaths_method,range(Nsims)]),np.linspace(0,1,Nsims),'-.*',color=(.2,(perct-.64)/.32,1-(perct-.64)/.32),label='Ora %.2f'%perct)
#                
#    plt.semilogx(np.sort(error_min[4,:]),np.linspace(0,1,Nsims),'--sk',label="lin min")
#    plt.semilogx(np.sort(error_min[5,:]),np.linspace(0,1,Nsims),'-.ok',label="ora min")
#    plt.semilogx(np.sort(error_guess).T,np.linspace(0,1,Nsims),':k',label="guess")
#    plt.legend()    
#    
#    fig_ctr+=1    
#    plt.figure(fig_ctr)
#    for perct in np.arange(0.04,.40,0.04):
#        npaths_method=np.maximum(np.minimum( percentMaxPowerThreshold(coef_est,[perct]) -3,36),0)
#        plt.semilogx(np.sort(error_dist[4,npaths_method,range(Nsims)]),np.linspace(0,1,Nsims),'--x',color=(.2,(perct-0)/.4,1-(perct-0)/.4),label='Lin %.2f'%perct)
#        plt.semilogx(np.sort(error_dist[5,npaths_method,range(Nsims)]),np.linspace(0,1,Nsims),'-.*',color=(.2,(perct-0)/.4,1-(perct-0)/.4),label='Ora %.2f'%perct)
#                
#    plt.semilogx(np.sort(error_min[4,:]),np.linspace(0,1,Nsims),'--sk',label="lin min")
#    plt.semilogx(np.sort(error_min[5,:]),np.linspace(0,1,Nsims),'-.ok',label="ora min")
#    plt.semilogx(np.sort(error_guess).T,np.linspace(0,1,Nsims),':k',label="guess")
#    plt.legend()
        
    #using the above commented code we determined that 0.8 and 0.25 are the best parameters for the power methods
    
    
    lMethods=[(noiseThreshold,[sigma2],'b','Nth'),
             (boostedNoiseThreshold,[sigma2],'c','Nbo'),
             (percentTotPowerThreshold,[.8],'r','Ptp'),
             (percentMaxPowerThreshold,[.25],'m','Pmp'),
             (usePhi0Err,[.8],'g','P0'),
             (useAllThreshold,[],'y','All')
            ]
    Nmethods=len(lMethods)
      
    fig_ctr+=1
    plt.figure(fig_ctr)
    plt.plot(npathbest.T,error_min.T,'x')
    plt.legend(["%s %s %s"%(x[0],x[1].replace("_"," "),x[2]) for x in configTable])
    plt.xlabel("Npaths min error")
    plt.ylabel("Location error")
    fig_ctr+=1
    plt.figure(fig_ctr)
    plt.plot(NpathsRetrieved.T,error_min.T,'x')
    plt.legend(["%s %s %s"%(x[0],x[1].replace("_"," "),x[2]) for x in configTable])
    plt.xlabel("Npaths retrieved in CS")
    plt.ylabel("Location error")
    
    fig_ctr+=1    
    fig_num_bar=fig_ctr
    fig_ctr+=1    
    fig_num_cdf=fig_ctr
    for nmethod in range(Nmethods):
        method=lMethods[nmethod]
        npaths_method=np.maximum(np.minimum( method[0](coef_est,method[1])-3,36),0)
        CovVsN_method=np.cov(npathbest,npaths_method)
        CorrN_method=CovVsN_method[-1,:-1] / CovVsN_method[range(Nconfigs),range(Nconfigs)] / CovVsN_method[-1,-1]        
        plt.figure(fig_num_bar)        
        plt.bar(np.arange(Nconfigs)+nmethod*1/(Nmethods+1),CorrN_method,1/(Nmethods+1),color=method[2])
        
        plt.figure(fig_num_cdf)  
        plt.semilogx(np.sort(error_dist[4,npaths_method,range(Nsims)]),np.linspace(0,1,Nsims),'--x',color=method[2])
        plt.semilogx(np.sort(error_dist[5,npaths_method,range(Nsims)]),np.linspace(0,1,Nsims),'-.*',color=method[2])
                
#        fig_ctr+=1
#        plt.figure(fig_ctr)
#        plt.plot(npaths_method,npathbest.T,'x')
#        plt.legend(["%s %s"%(x[0].replace("_"," "),x[1]) for x in configTable])
        fig_ctr+=1
        plt.figure(fig_ctr)
        plt.plot(npaths_method,error_min.T,'x')
        plt.legend(["%s %s %s"%(x[0],x[1].replace("_"," "),x[2]) for x in configTable])
    plt.figure(fig_num_bar)     
    plt.legend(["%s"%(x[2]) for x in lMethods])
    plt.figure(fig_num_cdf)  
    plt.semilogx(np.sort(error_min[4,:]),np.linspace(0,1,Nsims),'--sk')
    plt.semilogx(np.sort(error_min[5,:]),np.linspace(0,1,Nsims),'-.ok')
    plt.semilogx(np.sort(error_guess).T,np.linspace(0,1,Nsims),':k')
    plt.legend(["%s %s"%(y,x[3]) for x in lMethods+[(0,0,0,'Min')] for y in ['Linear','Oracle']]+['Guess'])
    
    npaths_method=np.maximum(np.minimum( usePhi0Err(coef_est,[.8])-3,36),0)
    fig_ctr+=1
    plt.figure(fig_ctr)
    for ncfg in range(Nconfigs):
        cfg = configTable[ncfg]
        er_data=error_dist[ncfg,npaths_method,np.arange(Nsims)]
        plt.semilogx(np.sort(er_data),np.linspace(0,1,Nsims),cfg[3]+cfg[5])
        print("""
Scheme %s error...
          Mean: %.3f
        Median: %.3f
    Worst 25-p: %.3f
          10-p: %.3f
           5-p: %.3f
              """%(
                  cfg[0]+" "+cfg[1]+" "+cfg[2],
                  np.mean(er_data),
                  np.median(er_data),                 
                  np.percentile(er_data,75),       
                  np.percentile(er_data,90),         
                  np.percentile(er_data,95),         
                  ))
    
    x0_guess=np.random.rand(Nsims)*100
    y0_guess=np.random.rand(Nsims)*100-50
    
    plt.semilogx(np.sort(error_guess).T,np.linspace(0,1,error_guess.size),':k')
    plt.legend(["%s %s %s"%(x[0],x[1].replace("_"," "),x[2]) for x in configTable]+['random guess'])
    
#    fig_ctr+=1
#    plt.figure(fig_ctr)
#    for nsim in range(Nsims):
#        if NpathsRetrieved[nsim]>3:
#            plt.plot(np.arange(0,1,1/(NpathsRetrieved[nsim]-3)),error_dist[4,0:(NpathsRetrieved[nsim]-3),nsim])
#    plt.axis([0,1,0,100])
    
 
    
    
    #NAnglenoises=11
    #t_start_ba= time.time()
    #bar = Bar("angle error", max=Nsims*NAnglenoises)
    #bar.check_tty = False
    #error_root=np.zeros((NAnglenoises,Nsims))
    #angleNoiseMSE=np.logspace(0,-6,NAnglenoises)
    #for ian in range( NAnglenoises):
    #    AoD_err = np.mod(AoD+np.random.randn(Nstrongest,Nsims)*np.sqrt(angleNoiseMSE[ian]),2*np.pi)
    #    AoA_err = np.mod(AoA+np.random.randn(Nstrongest,Nsims)*np.sqrt(angleNoiseMSE[ian]),2*np.pi)
    #    clock_error=(40/c)*np.random.rand(1,Nsims) #delay estimation error
    #    dels_err = dels+clock_error
    #    phi0_r=np.zeros((1,Nsims))
    #    x0_r=np.zeros((1,Nsims))
    #    y0_r=np.zeros((1,Nsims))
    #    x_r=np.zeros((Nstrongest,Nsims))
    #    y_r=np.zeros((Nstrongest,Nsims))
    #    for nsim in range(Nsims):
    #        (phi0_r[:,nsim],x0_r[:,nsim],y0_r[:,nsim],x_r[:,nsim],y_r[:,nsim])= loc.computeAllLocationsFromPaths(AoD_err[:,nsim],AoA_err[:,nsim],dels_err[:,nsim],method='fsolve')
    #        bar.next()
    #    error_root[ian,:]=np.sqrt(np.abs(x0-x0_r)**2+np.abs(y0-y0_r))
    #bar.finish()
    #t_run_ba = time.time() - t_start_ba
    #fig=
#    fig_ctr+=1
#    plt.figure(fig_ctr)
    #plt.loglog(angleNoiseMSE,np.percentile(error_root,90,axis=1))
    #plt.loglog(angleNoiseMSE,np.percentile(error_root,75,axis=1))
    #plt.loglog(angleNoiseMSE,np.percentile(error_root,50,axis=1))
    #fig=
#    fig_ctr+=1
#    plt.figure(fig_ctr)
    #ax = Axes3D(fig)
    #ax.plot_surface(np.log10(np.sort(error_root,axis=1)),np.tile(np.log10(angleNoiseMSE),[Nsims,1]).T,np.tile(np.arange(Nsims)/Nsims,[NAnglenoises,1]))
    #fig=
#    fig_ctr+=1
#    plt.figure(fig_ctr)
    #plt.semilogx(np.sort(error_root,axis=1).T,np.tile(np.arange(Nsims)/Nsims,[NAnglenoises,1]).T)
    
print("Total run time %d seconds"%(time.time()-t_total_run_init))