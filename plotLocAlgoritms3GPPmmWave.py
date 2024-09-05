#!/usr/bin/python
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

import sys
sys.path.append('../')
from CASTRO5G import  MultipathLocationEstimator as mploc
from CASTRO5G import  threeGPPMultipathGenerator as mp3g
from CASTRO5G import multipathChannel as mc
from CASTRO5G import  compressedSensingTools as cs

plt.close('all')

GEN_CHANS=False
GEN_PLOT=False
Nstrongest=40
Nmaxpaths=50
Nsims=100

EST_CHANS=True
EST_PLOT=False
Nd=16
Na=16
Ncp=128
Nframe=7
Nrft=1
Nrfr=4
K=128
Ds=300e-9
Ts=Ds/Ncp #2.5 ns
sigma2=.01

MATCH_CHANS=False
#MATCH_PLOT=True

EST_LOCS=True
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
    AoD0=np.arctan(y0/x0)+np.pi*(x0<0)
    AoA0=np.random.rand(Nsims)*2*np.pi #receiver angular measurement offset
    
    #delays based on distance
    c=3e8
    l0=np.sqrt(np.abs(x0)**2+np.abs(y0)**2)
    tau0=l0/c
    
    t_start_g= time.time()
    Npath=np.zeros(Nsims,dtype=int)
    AoD=np.zeros((Nsims,Nmaxpaths))
    DAoA=np.zeros((Nsims,Nmaxpaths))
    refLocs=np.zeros((Nsims,Nmaxpaths,2))
    TDoA=np.zeros((Nsims,Nmaxpaths))
    coefs=np.zeros((Nsims,Nmaxpaths),dtype=complex)
    for nsim in tqdm(range(Nsims),desc="Generating multipath channels"):
        #chamar ao xenerador de canle do 3GPP
        plinfo,macro,clustersNAD,subpathsNAD = chgen.create_channel((0,0,10),(x0[nsim],y0[nsim],1.5))        
        (_,_,plinfo,clustersAD,subpathsAD)  = chgen.fullFitAOA((0,0,10),(x0[nsim],y0[nsim],1.5),plinfo,clustersNAD,subpathsNAD,mode3D=False)
        (_,_,plinfo,clustersAD,subpathsAD)  = chgen.fullDeleteBacklobes((0,0,10),(x0[nsim],y0[nsim],1.5),plinfo,clustersAD,subpathsAD,tAOD=0,rAOA=AoA0[nsim]*180/np.pi)   
       # TDoA,P,AoA,AoD,ZoA,ZoD,Xs,Ys = clustersAD.T.to_numpy()
        # los, PLfree, SF = plinfo
        # TDoA_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp,_,phase00,_,_,_,xpos,ypos  = subpathsAD.T.to_numpy
        allCoefs = np.sqrt( subpathsAD.P.to_numpy() )*np.exp(2j*np.pi*subpathsAD.phase00.to_numpy())
        allDAoA = np.mod( subpathsAD.AoA.to_numpy()*np.pi/180-AoA0[nsim] ,np.pi*2)
        allAoD = np.mod( subpathsAD.AoD.to_numpy()*np.pi/180 ,np.pi*2)
        allTDoA = subpathsAD.TDoA.to_numpy()
        Npath[nsim]=np.minimum(Nmaxpaths,allCoefs.size)
            
    #    strongestInds=np.argpartition(-np.abs(amps),Npath[nsim],axis=0)[0:Npath[nsim]]
        strongestInds=np.argsort(-np.abs(allCoefs),axis=0)[0:Npath[nsim]]
        coefs[nsim,0:Npath[nsim]]=allCoefs[strongestInds[0:Npath[nsim]]]
        TDoA[nsim,0:Npath[nsim]] = allTDoA[strongestInds[0:Npath[nsim]]]
        AoD[nsim,0:Npath[nsim]] = allAoD[strongestInds[0:Npath[nsim]]]
        DAoA[nsim,0:Npath[nsim]] = allDAoA[strongestInds[0:Npath[nsim]]]
        refLocs[nsim,0:Npath[nsim],0] = subpathsAD.Xs.to_numpy()[strongestInds[0:Npath[nsim]]]
        refLocs[nsim,0:Npath[nsim],1] = subpathsAD.Ys.to_numpy()[strongestInds[0:Npath[nsim]]]
            
    t_run_g = time.time() - t_start_g

    np.savez('../Results/mimochans-%d.npz'%(Nsims),
             Npath=Npath,
             x0=x0,
             y0=y0,
             AoA0=AoA0,
             refLocs=refLocs,
             DAoA=DAoA,
             AoD=AoD,
             TDoA=TDoA,
             coefs=coefs)
else:
    data=np.load('../Results/mimochans-%d.npz'%(Nsims))
    Npath=data["Npath"]
    x0=data["x0"]
    y0=data["y0"]
    AoA0=data["AoA0"]
    refLocs=data["refLocs"]
    DAoA=data["DAoA"]
    AoD=data["AoD"]
    TDoA=data["TDoA"]
    coefs=data["coefs"]    

if GEN_PLOT:
    
    fig_ctr+=1
    plt.figure(fig_ctr)
    pcatch = plt.plot(np.vstack((np.zeros_like(refLocs[:,0,0]),refLocs[:,0,0],x0[0]*np.ones_like(refLocs[:,0,0]))),np.vstack((np.zeros_like(refLocs[:,0,1]),refLocs[:,0,1],y0[0]*np.ones_like(refLocs[:,0,1]))),':xb',label='Front-of-array Paths (detected)')
    plt.setp(pcatch[1:],label="_")
    bfil=(coefs[:,0]==0)#((AoA[:,0])>np.pi/2)&((AoA[:,0])<3*np.pi/2)1
    pcatch = plt.plot(np.vstack((np.zeros_like(refLocs[bfil,0,0]),refLocs[bfil,0,0],x0[0]*np.ones_like(refLocs[bfil,0,0]))),np.vstack((np.zeros_like(refLocs[bfil,0,1]),refLocs[bfil,0,1],y0[0]*np.ones_like(refLocs[bfil,0,1]))),':sk',label='Back-of-array paths (shielded)')
    #print(((AoA-AoA0)>np.pi)&((AoA-AoA0)<3*np.pi))
    plt.setp(pcatch[1:],label="_")
    plt.plot(x0[0],y0[0],'or',label='Receiver')
    plt.plot(0,0,'sr',label='Transmitter1')
    
    plt.plot(x0[0]+np.sin(AoA0[0])*5*np.array([-1,1]),y0[0]-np.cos(AoA0[0])*5*np.array([-1,1]),'r-',label='Array Plane')
    plt.plot(x0[0]+np.cos(AoA0[0])*5*np.array([0,1]),y0[0]+np.sin(AoA0[0])*5*np.array([0,1]),'g-',label='Array Face')
    plt.plot([0.01,0.01],[-1,-1],'r-')
    plt.plot([0,0],[0,1],'g-')
    
    plt.axis([0,120,-60,60])
    
    plt.title('mmWave multipath environment')
    plt.legend()
    
if EST_CHANS:
    #channel multipath estimation outputs with error
    
#    angleNoiseMSE=(2*np.pi/64)**2
#    AoD_err = np.mod(AoD+np.random.randn(Nsims,Nmaxpaths)*np.sqrt(angleNoiseMSE),2*np.pi)
#    AoA_err = np.mod(AoA+np.random.randn(Nsims,Nmaxpaths)*np.sqrt(angleNoiseMSE),2*np.pi)
#    clock_error=(40/c)*np.random.rand(1,Nsims) #delay estimation error
#    TDoA_err = TDoA+clock_error
    
    t_start_cs = time.time()
    Nchan=Nsims
    omprunner = cs.CSDictionaryRunner()
    dicFast=cs.CSMultiFFTDictionary()
    omprunner.setDictionary(dicFast)
    pilgen = mc.MIMOPilotChannel("IDUV")
    AoD_est=np.zeros((Nsims,Nmaxpaths))
    DAoA_est=np.zeros((Nsims,Nmaxpaths))
    TDoA_est=np.zeros((Nsims,Nmaxpaths))
    coef_est=np.zeros((Nsims,Nmaxpaths),dtype=complex)
    wall=np.zeros((Nsims,Nframe,K,Nrfr,Na),dtype=complex)
    vall=np.zeros((Nsims,Nframe,K,Nd,Nrft),dtype=complex)
    zall=np.zeros((Nsims,Nframe,K,Na,1),dtype=complex)
    hestall=np.zeros((Nsims,K,Na,Nd),dtype=complex)
    hkall=np.zeros((Nsims,K,Na,Nd),dtype=complex)
    IsuppAll=[]
    MSEOMP=np.zeros(Nsims)
    for nsim in tqdm(range(Nsims),desc="Estimating channels"):    
        mpch = mc.MultipathDEC((0,0,10),(x0[nsim],y0[nsim], 1.5),[])
        mpch.insertPathsFromListParameters(coefs[nsim,0:Npath[nsim]],TDoA[nsim,0:Npath[nsim]],DAoA[nsim,0:Npath[nsim]],AoD[nsim,0:Npath[nsim]],np.zeros(Npath[nsim]),np.zeros(Npath[nsim]),np.zeros(Npath[nsim]))
        ht=mpch.getDEC(Na,Nd,Ncp,Ts)*np.sqrt(Nd*Na)#mpch uses normalized matrices of gain 1
        hk=np.fft.fft(ht.transpose([2,0,1]),K,axis=0)
    
        (w,v)=pilgen.generatePilots(Nframe*K*Nrft,Na,Nd,Npr=Nframe*K*Nrfr,rShape=(Nframe,K,Nrfr,Na),tShape=(Nframe,K,Nd,Nrft))
        zp=mc.AWGN((Nframe,K,Na,1))
        zp_bb=np.matmul(w,zp)
        
        yp_noiseless=pilgen.applyPilotChannel(hk,w,v,None)
        yp=yp_noiseless+zp_bb*np.sqrt(sigma2)
        Pfa=1e-5
        factor_Pfa=np.log(1/(1-(Pfa)**(1/(Nd*Na*Ncp))))
        factor_Pfa=1
        ( hest, paths_est, Rsup, Hsup)=omprunner.OMP(yp,sigma2*K*Nframe*Nrfr*factor_Pfa,nsim,v,w, Xt=1.0, Xd=1.0, Xa=1.0, Xrefine=10.0,  Ncp = Ncp)
        #    a_est = np.linalg.lstsq(Isupp.observations,yp.reshape(-1,1),rcond=None)[0]
        a_est = paths_est.coefs.to_numpy()
        if Nmaxpaths<a_est.size:
    #        indStrongest=np.argpartition(-np.abs(a_est[:,0]),Nmaxpaths,axis=0)[0:Nmaxpaths]
            indStrongest=np.argsort(-np.abs(a_est),axis=0)[0:Nmaxpaths]
            AoD_est[nsim,:]=paths_est.AoD.to_numpy()[indStrongest]
            DAoA_est[nsim,:]=paths_est.AoA.to_numpy()[indStrongest]
            TDoA_est[nsim,:]=paths_est.TDoA.to_numpy()[indStrongest]*Ts
            coef_est[nsim,:]=paths_est.coefs.to_numpy()[indStrongest].reshape(-1)/np.sqrt(Na*Nd*Ncp)
        else:
            Nelems=a_est.size
            indStrongest=np.argsort(-np.abs(a_est),axis=0)
            AoD_est[nsim,0:Nelems]=paths_est.AoD.to_numpy()[indStrongest]
            DAoA_est[nsim,0:Nelems]=paths_est.AoA.to_numpy()[indStrongest]
            TDoA_est[nsim,0:Nelems]=paths_est.TDoA.to_numpy()[indStrongest]*Ts
            coef_est[nsim,0:Nelems]=paths_est.coefs.to_numpy()[indStrongest].reshape(-1)/np.sqrt(Na*Nd*Ncp)
        omprunner.dictionaryEngine.freeCacheOfPilot(nsim,(K,Ncp,Na,Nd),(Ncp,Na,Nd))
        MSEOMP[nsim]=np.sum(np.abs(hest-hk)**2)/np.sum(np.abs(hk)**2)
        IsuppAll.append(paths_est)
        wall[nsim,:]=w
        zall[nsim,:]=zp
        vall[nsim,:]=v
        hkall[nsim,:]=hk
        hestall[nsim,:]=hest
    
    DAoA_est=np.mod(DAoA_est,np.pi*2)
    AoD_est=np.mod(AoD_est,np.pi*2)
    
    t_run_cs = time.time() - t_start_cs
    
    print("OMP MSEs: %s"%(MSEOMP))
    
    NpathsRetrieved=np.sum(np.abs(coef_est)**2>0,axis=1)
    NpathsRetrievedMax=np.max(NpathsRetrieved)
    DAoA_est=DAoA_est[:,0:NpathsRetrievedMax]
    AoD_est=AoD_est[:,0:NpathsRetrievedMax]
    TDoA_est=TDoA_est[:,0:NpathsRetrievedMax]
    coef_est=coef_est[:,0:NpathsRetrievedMax]
    np.savez('../Results/mimoestimschans-%d-%d-%d-%d-%d-%d-%d.npz'%(Nrft,Nd,Na,Nrfr,Ncp,Nframe,Nsims),
             DAoA_est=DAoA_est,
             AoD_est=AoD_est,
             TDoA_est=TDoA_est,
             coef_est=coef_est,
             wall=wall,
             zall=zall,
             vall=vall,
             hkall=hkall,
             hestall=hestall,
             MSEOMP=MSEOMP)
else:
    data=np.load('../Results/mimoestimschans-%d-%d-%d-%d-%d-%d-%d.npz'%(Nrft,Nd,Na,Nrfr,Ncp,Nframe,Nsims),allow_pickle=True)
    DAoA_est=data["DAoA_est"]
    AoD_est=data["AoD_est"]
    TDoA_est=data["TDoA_est"]
    coef_est=data["coef_est"]
    wall=data["wall"]
    zall=data["zall"]
    vall=data["vall"]
    hkall=data["hkall"]
    hestall=data["hestall"]
    MSEOMP=data["MSEOMP"]
    NpathsRetrieved=np.sum(np.abs(coef_est)**2>0,axis=1)
    NpathsRetrievedMax=np.max(NpathsRetrieved)

if EST_PLOT:
    fig_ctr+=1
    plt.figure(fig_ctr)
    plt.semilogx(np.sort(MSEOMP),np.linspace(0,1,Nsims),'b')
    plt.xlabel("e $ = \\frac{\\|\\mathbf{h}-\\hat{\\mathbf{h}}\\|^2}{\\|\\mathbf{h}\\|^2}$")
    plt.ylabel("$F(e)$")
    fig_ctr+=1
    plt.figure(fig_ctr)
    Npathretcount=np.sum(np.arange(NpathsRetrievedMax+1)==NpathsRetrieved[:,np.newaxis],axis=0)
    plt.bar(np.arange(NpathsRetrievedMax+1),Npathretcount)
    plt.xlabel("n paths")
    plt.ylabel("$F(n)$")

#REFINAMENTO
#AoD_ref = sage(AOD_est)
#AoD_ref=AoD_est

#CALCULO POSICIOM CON REFINAMENTO EXIP

if MATCH_CHANS:
    
    def radUPA( Nangpoint, incidAngle , Nant = 16, dInterElement = .5):
        vUPA=mc.fULA(incidAngle,Nant,dInterElement)[...,None]#column vector
        vAngle=mc.fULA(np.arange(0,2*np.pi,2*np.pi/Nangpoint),Nant,dInterElement)[...,None]#column vector
        return (np.swapaxes(vUPA,vUPA.ndim-2,vUPA.ndim-1)[...,None,:,:]@vAngle.conj())[...,0,0]
    pdist=np.zeros((Nsims,Nmaxpaths,Nmaxpaths))
    for nsim in range(Nsims):
        pdist[nsim,:,:]= ( 
                np.abs(np.sin(AoD_est[nsim:nsim+1,:]).T-np.sin(AoD[nsim:nsim+1,0:]))/2 +
                np.abs(np.sin(DAoA_est[nsim:nsim+1,:]).T-np.sin(DAoA[nsim:nsim+1,:]))/2 +
                np.abs(TDoA_est[nsim:nsim+1,:].T-TDoA[nsim:nsim+1,:])*1e9/Ts/Ncp +
                np.abs(coef_est[nsim:nsim+1,:].T-coefs[nsim:nsim+1,:])**2*(1/np.abs(coef_est[nsim:nsim+1,:].T)**2+1/np.abs(coefs[nsim:nsim+1,:])**2)/np.sum(np.abs(coef_est[nsim:nsim+1,:])>0)
                )
    pathMatchTable = np.argmin(pdist,axis=1)
    DAoA_diff=np.zeros((Nsims,Nstrongest))
    AoD_diff=np.zeros((Nsims,Nstrongest))
    TDoA_diff=np.zeros((Nsims,Nstrongest))
    coef_diff=np.zeros((Nsims,Nstrongest),dtype=complex)
    for nsim in range(Nsims):
        DAoA_diff[nsim,:]=np.mod(DAoA[nsim,pathMatchTable[nsim,0:Nstrongest]],2*np.pi)-np.mod(DAoA_est[nsim,0:Nstrongest],2*np.pi)
        AoD_diff[nsim,:]=np.mod(AoD[nsim,pathMatchTable[nsim,0:Nstrongest]],2*np.pi)-np.mod(AoD_est[nsim,0:Nstrongest],2*np.pi)
        TDoA_diff[nsim,:]=TDoA[nsim,pathMatchTable[nsim,0:Nstrongest]]-TDoA_est[nsim,0:Nstrongest]
        coef_diff[nsim,:]=coefs[nsim,pathMatchTable[nsim,0:Nstrongest]]-coef_est[nsim,0:Nstrongest]
    Nlines=10
    fig_ctr+=1
    plt.figure(fig_ctr)
    plt.plot(10*np.log10(np.sort(np.abs(coef_diff[0:Nlines,:])**2/np.abs(coef_est[0:Nlines,:])**2,axis=1).T),np.arange(0,1,1/Nsims))
    
    fig_ctr+=1
    plt.figure(fig_ctr)
    plt.plot(10*np.log10(np.sort(np.abs(DAoA_diff[0:Nlines,:]),axis=1).T),np.arange(0,1,1/Nsims))
    
    fig_ctr+=1
    plt.figure(fig_ctr)
    plt.plot(10*np.log10(np.sort(np.abs(AoD_diff[0:Nlines,:]),axis=1).T),np.arange(0,1,1/Nsims))
    
    fig_ctr+=1
    plt.figure(fig_ctr)
    plt.plot(10*np.log10(np.sort(np.abs(TDoA_diff[0:Nlines,:]*1e9/Ts),axis=1).T),np.arange(0,1,1/Nsims))
    
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
#    maxdel=np.max(TDoA)*1e9
    ax.plot3D([0,0],[0,0],[0,Ncp],color='k')
    ax.text3D(0,0,Ncp,"delay [ns]",color='k')
    Nang=250
    angbase=np.arange(0,2*np.pi,2*np.pi/Nang)
    nsim=0
    for pind in range(0,np.sum(np.abs(coef_est[nsim,:])>0)):
        ang=DAoA[nsim,pathMatchTable[nsim,pind]]
        delay=TDoA[nsim,pathMatchTable[nsim,pind]]*1e9/Ts
        gain=np.abs(coefs[nsim,pathMatchTable[nsim,pind]])**2    
        x=np.maximum(10*np.log10(gain)+60,0)*np.cos(ang)
        y=np.maximum(10*np.log10(gain)+60,0)*np.sin(ang)
        p=ax.plot3D([0,x],[0,y],[delay,delay])
        ax.scatter3D(x,y,delay,marker='o',color=p[-1].get_color())    
        gainrad=np.abs(radUPA(Nang,ang,Na,.5)*coefs[nsim,pathMatchTable[nsim,pind]])**2
        xrad=np.maximum(10*np.log10(gainrad)+60,0)*np.cos(angbase)
        yrad=np.maximum(10*np.log10(gainrad)+60,0)*np.sin(angbase)
        ax.plot3D(xrad,yrad,delay*np.ones_like(xrad),'-.',color=p[-1].get_color())
        
        ang=DAoA_est[nsim,pind]
        delay=TDoA_est[nsim,pind]*1e9/Ts
        gain=np.abs(coef_est[nsim,pind])**2    
        x=np.maximum(10*np.log10(gain)+60,0)*np.cos(ang)
        y=np.maximum(10*np.log10(gain)+60,0)*np.sin(ang)
        ax.plot3D([0,x],[0,y],[delay,delay],color=p[-1].get_color())
        ax.scatter3D(x,y,delay,marker='x',color=p[-1].get_color())      
        gainrad=np.abs(radUPA(Nang,ang,Na,.5)*coef_est[nsim,pind])**2
        xrad=np.maximum(10*np.log10(gainrad)+60,0)*np.cos(angbase)
        yrad=np.maximum(10*np.log10(gainrad)+60,0)*np.sin(angbase)
        ax.plot3D(xrad,yrad,delay*np.ones_like(xrad),':',color=p[-1].get_color())   

if EST_LOCS:
    
    configTable=[
            #multipath data source, location method, user AoA0 coarse hint initialization
#            ('CS','brute','3path',''),
#            ('CS','root','3path','No Hint'),
            # ('CS','root','drop1','No hint'),
#            ('CS','root','3path','Hint'),
            # ('CS','root','drop1','Hint'),
            ('CS','oracle','','Pure'),
            # ('CS','oracle','','Hint'),            
#            ('true','brute','3path',''),
#            ('true','root','3path','No Hint'),
            # ('true','root','drop1','No hint'),
            # ('true','root','3path','Hint'),
            # ('true','root','drop1','Hint'),
            ('true','oracle','','Pure'),            
            ('true','oracle','','Hint'),            
    ]
    Nconfigs=len(configTable)
    
#    NpathsRetrieved=np.zeros(Nsims,dtype=np.int)
#    for nsim in range(Nsims):
#        NpathsRetrieved[nsim]=np.sum(np.abs(coef_est[nsim,:])**2>sigma2,axis=0)
#        NpathsRetrieved[nsim]=np.where(np.cumsum(np.abs(coef_est[nsim,:])**2,axis=0)/np.sum(np.abs(coef_est[nsim,:])**2,axis=0)>.75)[0][0]
#        NpathsRetrieved[nsim]=np.sum(coef_est[nsim,:]!=0)
    loc=mploc.MultipathLocationEstimator(nPoint=100,orientationMethod='lm')
    t_start_all=np.zeros(Nconfigs)
    t_end_all=np.zeros(Nconfigs)
    AoA0_est=np.zeros((Nconfigs,Nsims,NpathsRetrievedMax-2))
    d0_est=np.zeros((Nconfigs,Nsims,NpathsRetrievedMax-2,2))
    d_est=np.zeros((Nconfigs,Nsims,NpathsRetrievedMax-2,NpathsRetrievedMax,2))
    AoA0_coarse=np.round(AoA0*32/np.pi/2)*np.pi*2/32
    AoA0_cov=np.inf*np.ones((Nconfigs,Nsims,NpathsRetrievedMax-2))
    for ncfg in range(Nconfigs):
        cfg = configTable[ncfg]
        t_start_all[ncfg]=time.time()
        for nsim in tqdm(range(Nsims),desc=f'Estimating location with alg {cfg}'):
            for Nstimpaths in range(3, NpathsRetrieved[nsim] ):
                # try:
                if cfg[0]=='CS':
                    subpathValues = pd.DataFrame({
                        "AoD" : AoD_est[nsim,0:Nstimpaths],
                        "DAoA" : DAoA_est[nsim,0:Nstimpaths],
                        "TDoA" : TDoA_est[nsim,0:Nstimpaths],
                        })
                else:
                    subpathValues = pd.DataFrame({
                        "AoD" : AoD[nsim,0:Nstimpaths],
                        "DAoA" : DAoA[nsim,0:Nstimpaths],
                        "TDoA" : TDoA[nsim,0:Nstimpaths],
                        })
                if cfg[1]=='oracle':
                    if cfg[3]=='Hint':
                        AoA0_est[ncfg,nsim,Nstimpaths-3]=AoA0_coarse[nsim]
                    else:
                        AoA0_est[ncfg,nsim,Nstimpaths-3]=AoA0[nsim]
                    (d0_est[ncfg,nsim,Nstimpaths-3,:],_,d_est[ncfg,nsim,Nstimpaths-3,0:Nstimpaths,:]) = loc.computeAllPaths(subpathValues,AoA0_est[ncfg,nsim,Nstimpaths-3])
                else:
                    if cfg[3]=='Hint':                            
                        (d0_est[ncfg,nsim,Nstimpaths-3,:],_,d_est[ncfg,nsim,Nstimpaths-3,0:Nstimpaths,:],AoA0_est[ncfg,nsim,Nstimpaths-3],_)= loc.computeAllLocationsFromPaths(subpathValues,orientationMethod='lm',orientationMethodArgs={'groupMethod':cfg[2],'hitRotation':AoA0_coarse[nsim] })
                    else:
                        (d0_est[ncfg,nsim,Nstimpaths-3,:],_,d_est[ncfg,nsim,Nstimpaths-3,0:Nstimpaths,:],AoA0_est[ncfg,nsim,Nstimpaths-3],_)= loc.computeAllLocationsFromPaths(subpathValues,orientationMethod='lm',orientationMethodArgs={'groupMethod':cfg[2]})                            
                # except Exception as e:
                #     print("Omitting other error in nsim %d npaths %d"%(nsim,Nstimpaths))
                #     print(str(e))
            if NpathsRetrieved[nsim]<NpathsRetrievedMax:
                AoA0_est[ncfg,nsim,-3:NpathsRetrieved[nsim]]=AoA0_est[ncfg,nsim,NpathsRetrieved[nsim]-4]
                d0_est[ncfg,nsim,-3:NpathsRetrieved[nsim],:]=d0_est[ncfg,nsim,NpathsRetrieved[nsim]-4,:]
                d_est[ncfg,nsim,NpathsRetrieved[nsim]-3:,0:Nstimpaths,:]=d_est[ncfg,nsim,NpathsRetrieved[nsim]-4,0:Nstimpaths,:]
        t_end_all[ncfg]=time.time()
        np.savez('../Results/mimoestimslocs-%d-%d-%d-%d-%d.npz'%(Nd,Na,Ncp,Nframe,Nsims),
                t_start_all=t_start_all,
                t_end_all=t_end_all,
                AoA0_est=AoA0_est,
                d0_est=d0_est,
                d_est=d_est,
                configTable=configTable)
        configTable=np.array(configTable)
else:
    loc=mploc.MultipathLocationEstimator(nPoint=100,orientationMethod='lm')
    data=np.load('../Results/mimoestimslocs-%d-%d-%d-%d-%d.npz'%(Nd,Na,Ncp,Nframe,Nsims))
    t_start_all=data["t_start_all"]
    t_end_all=data["t_end_all"]
    AoA0_est=data["AoA0_est"]
    d0_est=data["d0_est"]
    d_est=data["d_est"]
    configTable=data["configTable"]
    Nconfigs=len(configTable)
    NpathsRetrieved=np.sum(np.abs(coef_est)**2>0,axis=1)
    NpathsRetrievedMax=np.max(NpathsRetrieved)
    AoA0_est=np.mod(AoA0_est,2*np.pi)
    AoA0_coarse=np.round(AoA0*32/np.pi/2)*np.pi*2/32

if PLOT_LOCS:
    
    fig_ctr+=1
    plt.figure(fig_ctr)
    d0=np.column_stack([x0,y0])
    error_dist=np.linalg.norm( d0_est - d0[None,:,None,:],axis=-1)
    error_dist[np.isnan(error_dist)]=np.inf
    npathbest=np.argmin(error_dist,axis=2)
    error_min=np.min(error_dist,axis=2)
    
    configTablePlot=[
            (':','x','c'),
            (':','+','y'),
#            (':','*','k'),
            ('-.','+','g'),
            ('-.','*','b'),
            ('-.','v','r'),            
            ('-.','^','m'),            
            ('--','s','c'),
            ('--','d','y'),
#            ('--','p','k'),
            ('-','d','g'),
            ('-','p','b'),
            ('-','>','r'),            
            ('-','<','m'),            
    ]
    for ncfg in range(Nconfigs):
        cfg = configTable[ncfg]
        (lncfg,mkcfg,clcfg) = configTablePlot[ncfg]
        plt.semilogx(np.sort(error_min[ncfg,:]),np.linspace(0,1,Nsims),linestyle=lncfg,color=clcfg)
    
    x0_guess=np.random.rand(Nsims)*100
    y0_guess=np.random.rand(Nsims)*100-50
    
    error_guess=np.sqrt(np.abs(x0-x0_guess)**2+np.abs(y0-y0_guess))
    plt.semilogx(np.sort(error_guess).T,np.linspace(0,1,error_guess.size),':k')
    plt.legend(["%s %s %s %s"%(x[0],x[1],x[2],x[3]) for x in configTable]+['random guess'])
    plt.xlabel("Position error (m)")
    plt.ylabel("C.D.F.")
    plt.title("Assuming optimum number of paths")
    
    fig_ctr+=1
    plt.figure(fig_ctr)
    labstrFilter=["CS root linear Hint", "true root linear Hint", "CS oracle Hint", "true oracle Hint"]
    plt.plot(x0,y0,'ok')
    for ncfg in range(Nconfigs):
        cfg = configTable[ncfg]
        (lncfg,mkcfg,clcfg) = configTablePlot[ncfg]
        labstr = "%s %s %s %s"%(cfg[0],cfg[1],cfg[2],cfg[3])
        if labstr in labstrFilter:
            pcatch = plt.plot(np.vstack((x0,d0_est[ncfg,range(Nsims),npathbest[ncfg,:],0])),np.vstack((y0,d0_est[ncfg,range(Nsims),npathbest[ncfg,:],1])),linestyle=':',color=clcfg,marker=mkcfg, mfc='none', label=labstr)
            plt.setp(pcatch[1:],label="_")

    plt.axis([0, 100, -50, 50])
    plt.legend()
    plt.title("All device positions and their error drift")
    
    fig_ctr+=1
    plt.figure(fig_ctr)
    labstrFilter=["CS bisec ", "CS root No hint", "CS root Hint", "CS root linear Hint"]
    AoA0_err = np.minimum(
            np.mod(np.abs(AoA0[None,:,None]-AoA0_est),np.pi*2),
            np.mod(np.abs(AoA0[None,:,None]+AoA0_est-2*np.pi),np.pi*2)
            )
    AoA0_eatmin = np.zeros((Nconfigs,Nsims))
    for ncfg in range(Nconfigs):
        cfg = configTable[ncfg]
        AoA0_eatmin[ncfg,:]=AoA0_err[ncfg,range(Nsims),npathbest[ncfg,:]]
        labstr = "%s %s %s %s"%(cfg[0],cfg[1],cfg[2],cfg[3])
        if labstr in labstrFilter:
            (lncfg,mkcfg,clcfg) = configTablePlot[ncfg]
            plt.loglog(AoA0_eatmin[ncfg,:],error_min[ncfg,:],linestyle='',marker=mkcfg,color=clcfg)
            plt.loglog(AoA0_err[ncfg,:,:].reshape(-1),error_dist[ncfg,:,:].reshape(-1),linestyle='',marker=mkcfg,color='r')
    plt.axis([0,np.pi,0,150])
    plt.legend(labstrFilter)
    plt.xlabel("$\\phi_o$ error (rad)")
    plt.ylabel("Position error (m)")
      
    fig_ctr+=1
    plt.figure(fig_ctr)
    for ncfg in range(Nconfigs):
        cfg = configTable[ncfg]
        (lncfg,mkcfg,clcfg) = configTablePlot[ncfg]
        plt.plot(npathbest[ncfg,:],error_min[ncfg,:],linestyle='',marker=mkcfg,color=clcfg)
    plt.legend(["%s %s %s %s"%(x[0],x[1],x[2],x[3]) for x in configTable])
    plt.xlabel("Npaths at min error")
    plt.ylabel("Location error")
    plt.axis([0,NpathsRetrievedMax,0,50])
    
    fig_ctr+=1
    plt.figure(fig_ctr)
    for ncfg in range(Nconfigs):
        cfg = configTable[ncfg]
        (lncfg,mkcfg,clcfg) = configTablePlot[ncfg]
        plt.plot(NpathsRetrieved,error_min[ncfg,:],linestyle='',marker=mkcfg,color=clcfg)
    plt.legend(["%s %s %s %s"%(x[0],x[1],x[2],x[3]) for x in configTable])
    plt.xlabel("Npaths retrieved total")
    plt.ylabel("Location error")
    plt.axis([0,NpathsRetrievedMax,0,50])

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
                         
                      
############################
#define method to determine the FIM lower bound to noise vs number of paths
    
#     def beta(w,ang):
#         Nant=w.shape[-1]
#         return(w@np.exp(-1j*np.pi*np.arange(Nant)[:,np.newaxis]*np.sin(ang)))
#     def d_beta(w,ang):
#         Nant=w.shape[-1]
#         return(w@(-1j*np.pi*np.arange(Nant)[:,np.newaxis]*np.cos(ang)*np.exp(-1j*np.pi*np.arange(Nant)[:,np.newaxis]*np.sin(ang))))    
#     def diffYtoParamGeneral(coef,delay,AoD,AoA,w,v,dAxis='None'):
#         K=w.shape[0]
#         if dAxis=='delay':
#             tau_term=-2j*np.pi* np.arange(K)[...,np.newaxis,np.newaxis,np.newaxis] *np.exp(-2j*np.pi* np.arange(K)[...,np.newaxis,np.newaxis,np.newaxis] * delay[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]*1e9/Ncp/Ts)
#         else:
#             tau_term=np.exp(-2j*np.pi* np.arange(K)[...,np.newaxis,np.newaxis,np.newaxis] * delay[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]*1e9/Ncp/Ts)
#         if dAxis=='AoD':
#             theta_term=d_beta( np.transpose(v,(0,1,3,2)) ,AoD[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])
#         else:
#             theta_term=beta( np.transpose(v,(0,1,3,2)) ,AoD[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])
#         if dAxis=='AoA':
#             phi_term=beta(w,AoA[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])
#         else:
#             phi_term=d_beta(w,AoA[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])
#         return(coef[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] * tau_term * theta_term * phi_term)
        
#     def getFIMYToParam(sigma2,coef,delay,AoD,AoA,w,v,dAxes=('delay','AoD','AoA') ):
#         npth=AoD.shape[0]
#         K=w.shape[0]
#         Nframe=w.shape[1]
#     #    Nrfr=w.shape[2]
#         Na=w.shape[3]
#         Nd=v.shape[2]
#     #    Nrft=v.shape[3]
#         listOfpartials = [ diffYtoParamGeneral(coef,delay,AoD,AoA,w,v, term ).reshape(npth,-1) for term in dAxes]
#         partialD=np.concatenate( listOfpartials ,axis=0)
#         J=2*np.real(partialD@partialD.conj().T)*sigma2*K*Nd*Na*Nframe
#         return( J )
    
#     def diffTau(p0,pP):
#         c=3e-1#in m/ns
#         return (p0/np.linalg.norm(p0-pP,axis=1)[:,np.newaxis]/c)
    
#     def diffTheta(p0,pP):
#         return (np.zeros((pP.shape[0],p0.shape[0])))
    
#     def diffPhi(p0,pP):
#         g=(p0[1]-pP[:,1])/(p0[0]-pP[:,0])
#         dgx= -(p0[1]-pP[:,1])/((p0[0]-pP[:,0])**2)
#         dgy= 1/(p0[0]-pP[:,0])
#         return np.concatenate([dgx[:,np.newaxis],dgy[:,np.newaxis]],axis=1) * 1/(1+g[:,np.newaxis]**2)
        
#     def getTParamToLoc(p0,pis,dAxes):
#         dfun={
#             'delay':diffTau,
#             'AoD':diffTheta,
#             'AoA':diffPhi
#               }
#         listOfPartials= [dfun[term](p0,pis) for term in dAxes]
#         T=np.concatenate(listOfPartials,axis=0)
#         return(T)

#     #for nsim in range(Nsims):
#     #    npath=npathbest[5,nsim]+3
#     #    J=getFIMYToParam(sigma2,coefs[0:npath,nsim],TDoA[0:npath,nsim],AoD[0:npath,nsim],AoA[0:npath,nsim],wall[nsim],vall[nsim],('delay','AoD','AoA'))
#     #    pathMSE=np.trace(np.linalg.inv(J))/3/npath
#     #    print("Sim best (%d Path) param MSE: %f"%(nsim,pathMSE))
    
#     def getFIMYToLoc(p0,pis,sigma2,coef,delay,AoD,AoA,w,v):
#         Jparam_notheta=getFIMYToParam(sigma2,coef,delay,AoD,AoA,w,v,('delay','AoA'))
#         T=getTParamToLoc(p0,pis,('delay','AoA'))
#         Jloc = T.conj().T @ Jparam_notheta @ T
#         return(Jloc)    
    
       
#     def getBestPathsFIMErr():#TODO add proper input arguments
#         posMSE=np.inf*np.ones((Nsims,NpathsRetrievedMax-3))
#         bar = Bar("FIM finding best num. paths", max=Nsims)
#         bar.check_tty = False
#         for nsim in range(Nsims):
#             for npath in range(3,NpathsRetrieved[nsim]):
#                 J2=getFIMYToLoc(p0[nsim,:], refLocs[0:npath,nsim,:] , sigma2,coefs[0:npath,nsim],TDoA[0:npath,nsim],AoD[0:npath,nsim],AoA[0:npath,nsim],wall[nsim],vall[nsim])
#                 posMSE[nsim,npath-3]=np.trace(np.linalg.inv(J2))
#             bar.next()
#         bar.finish()
#         return np.argmin(posMSE,axis=1)+3
    
#     def getBestPathsEstFIMErr():#TODO add proper input arguments
#         p0_est = np.stack((x0_est,y0_est),axis=-1) 
#         pAll_est = np.stack((x_est,y_est),axis=-1)   
#         posMSE_est=np.inf*np.ones((Nsims,NpathsRetrievedMax-3)) 
#         bar = Bar("Estimated FIM finding best num. paths", max=Nsims)
#         bar.check_tty = False 
#         for nsim in range(Nsims):
#             for npath in range(3,NpathsRetrieved[nsim]):
#                  J2_est=getFIMYToLoc(
#                     p0_est[Ncfglinhint,npath-3,nsim,:], 
#                     pAll_est[Ncfglinhint,npath-3,0:npath,nsim,:], 
#                     sigma2,
#                     coef_est[0:npath,nsim],
#                     TDoA_est[0:npath,nsim],
#                     AoD_est[0:npath,nsim],
#                     AoA_est[0:npath,nsim],
#                     wall[nsim],
#                     vall[nsim])
#                  posMSE_est[nsim,npath-3]=np.trace(np.linalg.inv(J2_est))
#             bar.next()
#         bar.finish()
#         return np.argmin(posMSE_est,axis=1)+3
   
#     p0 = np.stack((x0,y0),axis=-1)  
#     p0_est = np.stack((x0_est,y0_est),axis=-1) 
#     pAll_est = np.stack((x_est,y_est),axis=-1)     
  
#     Npathfrac=1-NpathsRetrieved/Ncp#TODO: make these variables accessible without being global
#     AvgNoBoost = -Npathfrac*np.log(1-Npathfrac)
#     Ncfglinhint = np.argmax(np.all(configTable==('CS','root','drop1','Hint'),axis=1))
#     Ncfgoracle = np.argmax(np.all(configTable==('CS','oracle','','Pure'),axis=1))
    
#     lMethods=[
#              #oracle reveals npaths with true best distance error
#              (np.argmin(error_dist[Ncfglinhint,:],axis=0) + 3 ,'b','Oracle'),
#              #total num of paths retrieved
# #             ( NpathsRetrieved,'r','All'),
#              #number of paths with magnitude above typical noise
#              (np.sum(np.abs(coef_est)**2>sigma2,axis=0) ,'r','Ncph'),
#              ( np.minimum(9,NpathsRetrieved),'g','fix'),
#              #number of paths with magnitude above boosted noise (max Noath out of Ncp)
# #             (np.sum(np.abs(coef_est)**2>AvgNoBoost*sigma2,axis=0),'c','Nbo'),
#              #num paths that represent 80% total channel power, threshold adjusted by trial and error
# #             ( np.sum(np.cumsum(np.abs(coef_est)**2/np.sum(np.abs(coef_est)**2,axis=0),axis=0)<.8,axis=0) ,'r','Ptp'),
#              #num paths with power greater than fraction of largest MPC peak power, threshold adjusted by trial and error
# #             (np.sum(np.abs(coef_est)**2/np.max(np.abs(coef_est)**2,axis=0)>.25,axis=0) ,'m','Pmp'),
#              #num paths with AoA0 estimation greater than AoA0 coarse hint resolution
# #             ( np.sum(np.abs(AoA0_est[Ncfglinhint,:,:]-AoA0_coarse)<2*np.pi/32,axis=0) ,'g','P0'),
#              #num paths with minimum AoA0 estimation variance in lstsq
# #             ( np.argmin(AoA0_cov[Ncfglinhint,:,:],axis=0) ,'y','V0'),
# #             #num paths with minimum actual channel FIM (oracle-ish)
# #             (getBestPathsFIMErr(),'k','FIM'),
#              #num paths with minimum estimated channel FIM
# #             (getBestPathsEstFIMErr(),'k','eFIM'),
#             ]
#     Nmethods=len(lMethods)
    
    
#     fig_ctr+=1    
#     fig_num_bar=fig_ctr
#     fig_ctr+=1    
#     fig_num_cdf=fig_ctr
#     lPercentilesPlot = [50,75,90,95]
#     er_data_pctls=np.zeros((Nmethods,Nconfigs,len(lPercentilesPlot)+1))
#     for nmethod in range(Nmethods):
#         method=lMethods[nmethod]
#         npaths_method=np.maximum(method[0] - 3 ,0)
#         CovVsN_method=np.cov(npathbest,npaths_method)
#         CorrN_method=CovVsN_method[-1,:-1] / CovVsN_method[range(Nconfigs),range(Nconfigs)] / CovVsN_method[-1,-1]        
#         plt.figure(fig_num_bar)        
#         plt.bar(np.arange(Nconfigs)+nmethod*1/(Nmethods+2),CorrN_method,1/(Nmethods+2),color=method[1])
        
#         plt.figure(fig_num_cdf)
#         for ncfg in range(Nconfigs):
#             cfg = configTable[ncfg]
#             (lncfg,mkcfg,clcfg) = configTablePlot[ncfg]
#             plt.semilogx(np.sort(error_dist[ncfg,npaths_method,range(Nsims)]),np.linspace(0,1,Nsims),linestyle=lncfg,marker=mkcfg,color=method[1])
                
# #        fig_ctr+=1
# #        plt.figure(fig_ctr)
# #        plt.plot(npaths_method,npathbest.T,'x')
# #        plt.legend(["%s %s"%(x[0].replace("_"," "),x[1]) for x in configTable])
# #        fig_ctr+=1
# #        plt.figure(fig_ctr)
# #        for ncfg in range(Nconfigs):
# #            cfg = configTable[ncfg]
# #            (lncfg,mkcfg,clcfg) = configTablePlot[ncfg]
# #            plt.plot(NpathsRetrieved, error_dist[ncfg,npaths_method,range(Nsims)],linestyle='',marker=mkcfg,color=clcfg)
# #        plt.legend(["%s %s %s %s"%(x[0],x[1],x[2],x[3]) for x in configTable])
# #        plt.axis([0,NpathsRetrievedMax,0,50])
# #        plt.xlabel("npaths selected method")
# #        plt.ylabel("Location error")
# #        plt.title("method = %s"%method[3])
        
# #        fig_ctr+=1
# #        plt.figure(fig_ctr)
#         for ncfg in range(Nconfigs):
#             cfg = configTable[ncfg]
#             (lncfg,mkcfg,clcfg) = configTablePlot[ncfg]
#             er_data=error_dist[ncfg,npaths_method,np.arange(Nsims)]
# #            plt.semilogx(np.sort(er_data),np.linspace(0,1,Nsims),linestyle=lncfg,color=clcfg)
# #            print("""
# #    Criterion %s, Scheme %s error...
# #              Mean: %.3f
# #            Median: %.3f
# #        Worst 25-p: %.3f
# #              10-p: %.3f
# #               5-p: %.3f
# #                  """%(
# #                      method[3],
# #                      cfg[0]+" "+cfg[1]+" "+cfg[2],
# #                      np.mean(er_data),
# #                      np.median(er_data),                 
# #                      np.percentile(er_data,75),
# #                      np.percentile(er_data,90),
# #                      np.percentile(er_data,95),
# #                      ))
#             er_data_pctls[nmethod,ncfg,:]=[
#                       np.mean(er_data),
#                       np.percentile(er_data,50),                 
#                       np.percentile(er_data,75),
#                       np.percentile(er_data,90),
#                       np.percentile(er_data,95),
#                       ]
# #        plt.semilogx(np.sort(error_guess).T,np.linspace(0,1,Nsims),':k')        
# #        plt.title("Assuming %s bound min number of paths"%method[3])
# #        plt.legend(["%s %s %s %s"%(x[0],x[1],x[2],x[3]) for x in configTable]+['random guess'])
# #        plt.xlabel("Position error (m)")
# #        plt.ylabel("C.D.F.")
        
#     plt.figure(fig_num_bar)
#     plt.legend([x[2] for x in lMethods])
#     plt.xlabel('config')
#     plt.xticks(ticks=np.arange(Nconfigs),labels=["%s %s %s %s"%(x[0],x[1],x[2],x[3]) for x in configTable],rotation=45)
#     plt.ylabel('Corr coefficient')
# #    plt.legend(["%s %s %s %s"%(x[0],x[1],x[2],x[3]) for x in configTable])
   
    
# #    npaths_method=np.maximum(np.minimum( usePhi0Err(coef_est,[.8])-3,36),0)
# #    fig_ctr+=1
# #    plt.figure(fig_ctr)
# #    for ncfg in range(Nconfigs):
# #        cfg = configTable[ncfg]
# #        er_data=error_dist[ncfg,npaths_method,np.arange(Nsims)]
# #        plt.semilogx(np.sort(er_data),np.linspace(0,1,Nsims),linestyle=lncfg,color=clcfg)
# #        print("""
# #Scheme %s error...
# #          Mean: %.3f
# #        Median: %.3f
# #    Worst 25-p: %.3f
# #          10-p: %.3f
# #           5-p: %.3f
# #              """%(
# #                  cfg[0]+" "+cfg[1]+" "+cfg[2],
# #                  np.mean(er_data),
# #                  np.median(er_data),                 
# #                  np.percentile(er_data,75),       
# #                  np.percentile(er_data,90),         
# #                  np.percentile(er_data,95),         
# #                  ))
#     plt.figure(fig_num_cdf)  
#     plt.semilogx(np.sort(error_guess).T,np.linspace(0,1,Nsims),':k')
#     plt.legend(["%s %s %s %s"%(y[0],y[1].replace("_"," "),y[2],x[2]) for x in lMethods for y in configTable]+['Random Guess'])
#     plt.xlabel("Position error (m)")
#     plt.ylabel("C.D.F.")
   
#     fig_ctr+=1    
#     plt.figure(fig_ctr)
#     for nmethod in range(Nmethods):
#         method=lMethods[nmethod]
#         plt.bar(np.arange(Nconfigs)+nmethod*1/(Nmethods+2),er_data_pctls[nmethod,:,0],1/(Nmethods+2),color=method[1])
#     plt.xlabel('config')
#     plt.xticks(ticks=np.arange(Nconfigs),labels=["%s %s %s %s"%(x[0],x[1],x[2],x[3]) for x in configTable],rotation=45)
#     plt.ylabel('Mean error (m)')
#     plt.legend([x[2] for x in lMethods])
# #  
#     for npct in range(len(lPercentilesPlot)):
#         fig_ctr+=1    
#         plt.figure(fig_ctr)
#         for nmethod in range(Nmethods):
#             method=lMethods[nmethod]
#             plt.bar(np.arange(Nconfigs)+nmethod*1/(Nmethods+2),er_data_pctls[nmethod,:,1+npct],1/(Nmethods+2),color=method[1])
#         plt.xlabel('config')
#         plt.xticks(ticks=np.arange(Nconfigs),labels=["%s %s %s %s"%(x[0],x[1],x[2],x[3]) for x in configTable],rotation=45)
#         plt.ylabel('%d pct error (m)'%lPercentilesPlot[npct])
#         plt.legend([x[2] for x in lMethods])
        
    
# #    npaths_method=np.maximum(lMethods[NmethodBest90pctl][0] - 3 ,0)
    
    
#     #NAnglenoises=11
#     #t_start_ba= time.time()
#     #bar = Bar("angle error", max=Nsims*NAnglenoises)
#     #bar.check_tty = False
#     #error_root=np.zeros((NAnglenoises,Nsims))
#     #angleNoiseMSE=np.logspace(0,-6,NAnglenoises)
#     #for ian in range( NAnglenoises):
#     #    AoD_err = np.mod(AoD+np.random.randn(Nsims,Nstrongest)*np.sqrt(angleNoiseMSE[ian]),2*np.pi)
#     #    AoA_err = np.mod(AoA+np.random.randn(Nsims,Nstrongest)*np.sqrt(angleNoiseMSE[ian]),2*np.pi)
#     #    clock_error=(40/c)*np.random.rand(1,Nsims) #delay estimation error
#     #    TDoA_err = TDoA+clock_error
#     #    AoA0_r=np.zeros((1,Nsims))
#     #    x0_r=np.zeros((1,Nsims))
#     #    y0_r=np.zeros((1,Nsims))
#     #    x_r=np.zeros((Nsims,Nstrongest))
#     #    y_r=np.zeros((Nsims,Nstrongest))
#     #    for nsim in range(Nsims):
#     #        (AoA0_r[:,nsim],x0_r[:,nsim],y0_r[:,nsim],x_r[:,nsim],y_r[:,nsim])= loc.computeAllLocationsFromPaths(AoD_err[:,nsim],AoA_err[:,nsim],TDoA_err[:,nsim],method='root')
#     #        bar.next()
#     #    error_root[ian,:]=np.sqrt(np.abs(x0-x0_r)**2+np.abs(y0-y0_r))
#     #bar.finish()
#     #t_run_ba = time.time() - t_start_ba
#     #fig=
# #    fig_ctr+=1
# #    plt.figure(fig_ctr)
#     #plt.loglog(angleNoiseMSE,np.percentile(error_root,90,axis=1))
#     #plt.loglog(angleNoiseMSE,np.percentile(error_root,75,axis=1))
#     #plt.loglog(angleNoiseMSE,np.percentile(error_root,50,axis=1))
#     #fig=
# #    fig_ctr+=1
# #    plt.figure(fig_ctr)
#     #ax = Axes3D(fig)
#     #ax.plot_surface(np.log10(np.sort(error_root,axis=1)),np.tile(np.log10(angleNoiseMSE),[Nsims,1]).T,np.tile(np.arange(Nsims)/Nsims,[NAnglenoises,1]))
#     #fig=
# #    fig_ctr+=1
# #    plt.figure(fig_ctr)
#     #plt.semilogx(np.sort(error_root,axis=1).T,np.tile(np.arange(Nsims)/Nsims,[NAnglenoises,1]).T)
    

# #(AoA0_aux, x0_aux, y0_aux, x_aux, y_aux,AoA0_var) = loc.computeAllLocationsFromPaths( AoD_est[0:10,0], AoA_est[0:10,0], TDoA_est[0:10,0],method='root_linear',hint_AoA0= AoA0_coarse[0])
    
print("Total run time %d seconds"%(time.time()-t_total_run_init))
