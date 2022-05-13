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
GEN_PLOT=True
Nstrongest=40
Nmaxpaths=400
Nsims=100

EST_CHANS=False
EST_PLOT=True
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

DIST_CHANS=True
EST_LOCS=True

t_total_run_init=time.time()

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
            allaoa_shifted = allaoa_shifted + np.pi
            phi0[nsim]=np.pi+phi0[nsim]
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
        Pfa=1e-10
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

if DIST_CHANS:
    pdist=np.zeros((Nsims,Nmaxpaths,Nstrongest))
    for nsim in range(Nsims):
        pdist[nsim,:,:]= ( 
                np.abs(np.sin(AoD_est[:,nsim:nsim+1]).T-np.sin(AoD[:,nsim:nsim+1]))/2 +
                np.abs(np.sin(AoA_est[:,nsim:nsim+1]).T-np.sin(AoA[:,nsim:nsim+1]))/2 +
                np.abs(dels_est[:,nsim:nsim+1].T-dels[:,nsim:nsim+1])*1e9/Nt +
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
    plt.figure(4)
    plt.plot(10*np.log10(np.sort(np.abs(coef_diff[0:Nlines,:])**2/np.abs(coef_est[0:Nlines,:])**2,axis=1).T),np.arange(0,1,1/Nsims))
    plt.figure(5)
    plt.plot(10*np.log10(np.sort(np.abs(AoA_diff[0:Nlines,:]),axis=1).T),np.arange(0,1,1/Nsims))
    plt.figure(6)
    plt.plot(10*np.log10(np.sort(np.abs(AoD_diff[0:Nlines,:]),axis=1).T),np.arange(0,1,1/Nsims))
    plt.figure(7)
    plt.plot(10*np.log10(np.sort(np.abs(dels_diff[0:Nlines,:]*1e9/Ts),axis=1).T),np.arange(0,1,1/Nsims))
    
    fig = plt.figure(8)
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
            ('bisec',False,':b'),
            ('fsolve',False,':g'),
            ('fsolve_linear',False,'--g'),
            ('fsolve',True,':m'),
            ('fsolve_linear',True,'--m'),
            ('oracle',False,'r'),            
    ]
    Nconfigs=len(configTable)
    NstimpathsMax=np.sum(np.abs(coef_est)**2>sigma2,axis=0)
#    NstimpathsMax=np.zeros(Nsims,dtype=np.int)
#    for nsim in range(Nsims):
#        NstimpathsMax[nsim]=np.sum(np.abs(coef_est[:,nsim])**2>sigma2,axis=0)
#        NstimpathsMax[nsim]=np.where(np.cumsum(np.abs(coef_est[:,nsim])**2,axis=0)/np.sum(np.abs(coef_est[:,nsim])**2,axis=0)>.75)[0][0]
#        NstimpathsMax[nsim]=np.sum(coef_est[:,nsim]!=0)
    loc=mploc.MultipathLocationEstimator(Npoint=1000,Nref=20,Ndiv=2,RootMethod='lm')
    t_start_all=np.zeros(Nconfigs)
    t_end_all=np.zeros(Nconfigs)
    phi0_est=np.zeros((Nconfigs,Nstrongest-3,Nsims))
    x0_est=np.zeros((Nconfigs,Nstrongest-3,Nsims))
    y0_est=np.zeros((Nconfigs,Nstrongest-3,Nsims))
    x_est=np.zeros((Nconfigs,Nstrongest-3,Nstrongest,Nsims))
    y_est=np.zeros((Nconfigs,Nstrongest-3,Nstrongest,Nsims))
    phi0_coarse=np.round(phi0*256/np.pi/2)*np.pi*2/256
    for ncfg in range(Nconfigs):
        cfg = configTable[ncfg]
        t_start_all[ncfg]=time.time()
        bar = Bar("%s %s"%(cfg[0:2]), max=Nsims)
        bar.check_tty = False
        for nsim in range(Nsims):
            for Nstimpaths in range(3,NstimpathsMax[nsim]):
                if cfg[0]=='oracle':
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
                            phi0[nsim])
                else:
                    if (cfg[0]!='fsolve_linear') or (nsim!=47):
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
                                method=cfg[0],
                                hint_phi0= ( phi0_coarse[nsim] if (cfg[1]) else None ) )
            bar.next()
        bar.finish()
        t_end_all[ncfg]=time.time()
        
        
    plt.figure(3)
    error_dist=np.sqrt( (x0_est - x0)**2 + (y0_est - y0)**2 )
    for ncfg in range(Nconfigs):
        cfg = configTable[ncfg]
        plt.semilogx(np.sort(np.min(error_dist[ncfg,:,:],axis=0)),np.linspace(0,1,Nsims),cfg[2])
    
    x0_guess=np.random.rand(Nsims)*100
    y0_guess=np.random.rand(Nsims)*100-50
    
    error_guess=np.sqrt(np.abs(x0-x0_guess)**2+np.abs(y0-y0_guess))
    plt.semilogx(np.sort(error_guess).T,np.linspace(0,1,error_guess.size),':k')
    plt.legend(["%s %s"%(x[0].replace("_"," "),x[1]) for x in configTable]+['random guess'])
#    plt.figure(4)
##    plt.plot(np.vstack((x0,x0_b)),np.vstack((y0,y0_b)),':sb', mfc='none')
#    npathbest_b=np.argmin(error_bisec,axis=0)
#    npathbest_r=np.argmin(error_root,axis=0)
#    npathbest_r2=np.argmin(error_root2,axis=0)
#    npathbest_h=np.argmin(error_hint,axis=0)
#    npathbest_o=np.argmin(error_oracle,axis=0)
##    plt.plot(np.vstack((x0,x0_r[npathbest_r,range(Nsims)])),np.vstack((y0,y0_r[npathbest_r,range(Nsims)])),':xr')
##    plt.plot(np.vstack((x0,x0_o[npathbest_o,range(Nsims)])),np.vstack((y0,y0_o[npathbest_o,range(Nsims)])),':^g')
#    plt.plot(np.vstack((x0,x0_h[npathbest_h,range(Nsims)])),np.vstack((y0,y0_h[npathbest_h,range(Nsims)])),':pm')
#    plt.plot(x0,y0,'ok')
#    plt.axis([0, 100, -50, 50])
##    
#    plt.figure(9)
#    plt.loglog(np.abs(np.mod(phi0-phi0_r[np.argmin(error_root,axis=0),range(Nsims)]+np.pi,2*np.pi)-np.pi),np.min(error_root,axis=0),'o')
#    plt.axis([0,np.pi,0,150])
#
#    plt.figure(10)
#    plt.plot(np.sum(np.abs(coef_est)**2>sigma2,axis=0),np.argmin(error_hint,axis=0),'ob')
#    plt.plot(np.sum(np.abs(coef_est)**2>sigma2,axis=0),np.argmin(error_root,axis=0),'xg')
#    plt.plot(np.sum(np.abs(coef_est)**2>sigma2,axis=0),np.argmin(error_root2,axis=0),'sk')
#    plt.plot(np.sum(np.abs(coef_est)**2>sigma2,axis=0),np.argmin(error_bisec,axis=0),'^r')
    
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
    #fig=plt.figure(5)
    #plt.loglog(angleNoiseMSE,np.percentile(error_root,90,axis=1))
    #plt.loglog(angleNoiseMSE,np.percentile(error_root,75,axis=1))
    #plt.loglog(angleNoiseMSE,np.percentile(error_root,50,axis=1))
    #fig=plt.figure(4)
    #ax = Axes3D(fig)
    #ax.plot_surface(np.log10(np.sort(error_root,axis=1)),np.tile(np.log10(angleNoiseMSE),[Nsims,1]).T,np.tile(np.arange(Nsims)/Nsims,[NAnglenoises,1]))
    #fig=plt.figure(5)
    #plt.semilogx(np.sort(error_root,axis=1).T,np.tile(np.arange(Nsims)/Nsims,[NAnglenoises,1]).T)
    
print("Total run time %d seconds"%(time.time()-t_total_run_init))