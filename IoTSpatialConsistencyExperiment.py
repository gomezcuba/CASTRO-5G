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
        return(np.sum(coefs*tpulse*arrayA*arrayD,axis=3))
        
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
    
def disambiguateAngles(de0,ad0,aa0,de1,ad1,aa1,deltax,deltay):
    c=3e8
    path_length=c*de0    
    newdelay_c=de0+(np.cos(aa0)*deltax+np.sin(aa0)*deltay)/c
    newdelay_i=de0+(np.cos(np.pi-aa0)*deltax+np.sin(np.pi-aa0)*deltay)/c
    #from ad1=ad0-np.arctan(delta_distD/path_length)
    delta_distD_ex=path_length*np.tan(ad0-ad1)
    delta_distA_ex=path_length*np.tan(aa0-aa1)
    # assuming the angles are correctly identified fom -pi/2 to pi/2
    delta_distD_c=-np.sin(ad0)*deltax+np.cos(ad0)*deltay
    delta_distA_c=+np.sin(aa0)*deltax-np.cos(aa0)*deltay 
    # assuming the angles are incorrectly mirror-flipped, the correct angle is pi-x from pi/2 to 3*pi/2
    # sin is the same but cos is opposite
    delta_distD_i=-np.sin(ad0)*deltax-np.cos(ad0)*deltay
    delta_distA_i=+np.sin(aa0)*deltax+np.cos(aa0)*deltay
    
    bFlipAoA = ( np.abs(newdelay_c-de1) > np.abs(newdelay_i-de1) )
    bFlipAoA_alt = ( np.abs(delta_distA_c-delta_distA_ex) > np.abs(delta_distA_i-delta_distA_ex) )
    bFlipAoD = ( np.abs(delta_distD_c-delta_distD_ex) > np.abs(delta_distD_i-delta_distD_ex) )
    
##      predict future angles form angles at position 0
##     assuming the angles are correctly identified fom -pi/2 to pi/2
#    delta_distD_c=-np.sin(ad0)*deltax+np.cos(ad0)*deltay
#    newaod_c=ad0-np.arctan(delta_distD_c/path_length)
#    delta_distA_c=+np.sin(aa0)*deltax-np.cos(aa0)*deltay
#    newaoa_c=aa0-np.arctan(delta_distA_c/path_length)
#    #  predict future angles form angles at position 0
#    # assuming the angles are incorrectly mirror-flipped, the correct angle is pi-x from pi/2 to 3*pi/2
#    # sin is the same but cos is opposite
#    delta_distD_i=-np.sin(ad0)*deltax-np.cos(ad0)*deltay
#    newaod_i=np.pi-ad0-np.arctan(delta_distD_i/path_length)
#    delta_distA_i=+np.sin(aa0)*deltax+np.cos(aa0)*deltay
#    newaoa_i=np.pi-aa0-np.arctan(delta_distA_i/path_length)
#    #if predicted angles with first hypothesis are closer to true angles at position 1, the flipping is not needed
#    bFlipAoA = ( np.abs(newaoa_c-aa1) > np.abs(newaoa_i-np.pi+aa1) )
#    bFlipAoD = ( np.abs(newaod_c-ad1) > np.abs(newaod_i-np.pi+ad1) )
    return(bFlipAoA,bFlipAoD)

def fixDegenAngles(de0,ad0,aa0,de1,ad1,aa1,deltax,deltay):
    c=3e8
    path_length=c*de0    
    newdelay_c=de0+(np.cos(aa0)*deltax+np.sin(aa0)*deltay)/c
    newdelay_i=de0+(np.cos(-aa0)*deltax+np.sin(-aa0)*deltay)/c
    #from ad1=ad0-np.arctan(delta_distD/path_length)
    delta_distD_ex=path_length*np.tan(ad0-ad1)
    delta_distA_ex=path_length*np.tan(aa0-aa1)
    # assuming the angles are correctly identified fom -pi/2 to pi/2
    delta_distD_c=-np.sin(ad0)*deltax+np.cos(ad0)*deltay
    delta_distA_c=+np.sin(aa0)*deltax-np.cos(aa0)*deltay 
    # assuming the angles are incorrectly mirror-flipped, the correct angle is pi-x from pi/2 to 3*pi/2
    # sin is the same but cos is opposite
    delta_distD_i=-np.sin(-ad0)*deltax+np.cos(-ad0)*deltay
    delta_distA_i=+np.sin(-aa0)*deltax-np.cos(-aa0)*deltay
    
    bFixDgAoA = ( np.abs(newdelay_c-de1) > np.abs(newdelay_i-de1) )
    bFixDgAoA_alt = ( np.abs(delta_distA_c-delta_distA_ex) > np.abs(delta_distA_i-delta_distA_ex) )
    bFixDgAoD = ( np.abs(delta_distD_c-delta_distD_ex) > np.abs(delta_distD_i-delta_distD_ex) )
    
    return(bFixDgAoA,bFixDgAoD)

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
    delay1=delay1-np.min(delay1)
    
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
    
    if bGenRand:
        zp=(np.random.randn(Nu,Nk,Nxp,Na,1)+1j*np.random.randn(Nu,Nk,Nxp,Na,1))/np.sqrt(2)
    yp=np.zeros((Nu,Nk,Nxp,Nrfr,1),dtype=np.complex64)
    sigma2=.01
    hestall=np.zeros((Nu,Nk,Na,Nd),dtype=np.complex64)
    Isupall=[]
    for nu in range(Nu):
        yp[nu,...]=pilgen.applyPilotChannel(hkall[nu,...],w,v,zp[nu,...]*np.sqrt(sigma2))    
        ( hestall[nu,...], Isupp )=omprunner.OMPBR(yp[nu,...],sigma2*Nk*Nxp*Nrfr*1.2,0,v,w, Xt=1.0, Xd=1.0, Xa=1.0, Xmu=10.0)
        Isupall.append(Isupp)
    NMSE[isim,:]=np.sum(np.abs(hestall-hkall)**2)/np.sum(np.abs(hkall)**2,axis=(1,2,3))
    print(NMSE[isim,:])
    NestPaths=[len(Isupall[x].AoDs) for x in range(Nu)]
    maxNestPaths=np.max([len(Isupall[x].AoDs) for x in range(Nu)])
    aod_est=np.zeros((Nu,maxNestPaths))
    aoa_est=np.zeros((Nu,maxNestPaths))
    del_est=np.zeros((Nu,maxNestPaths))
    coef_est=np.zeros((Nu,maxNestPaths),dtype=np.complex)
    for nu in range(Nu):
        aod_est[nu,0:NestPaths[nu]]=Isupall[nu].AoDs.T
        aoa_est[nu,0:NestPaths[nu]]=Isupall[nu].AoAs.T
        del_est[nu,0:NestPaths[nu]]=(Isupall[nu].delays.T+clock_offset)*Ts+d1/c
        coef_est[nu,0:NestPaths[nu]]=Isupall[nu].coefs.T/np.sqrt(Nk*Na*Nd)
    
    ord_est=np.argsort(-np.abs(coef_est)**2)
    fancy_aux=np.tile(np.arange(coef_est.shape[0]).reshape(-1,1),[1,coef_est.shape[1]])
    aod_est=aod_est[fancy_aux,ord_est]
    aoa_est=aoa_est[fancy_aux,ord_est]
    del_est=del_est[fancy_aux,ord_est]
    coef_est=coef_est[fancy_aux,ord_est]
    
    #NdisambPaths=np.minimum(NestPaths[0],NestPaths[1])
    #NdisambPaths=np.minimum(np.sum(np.cumsum(np.abs(coef_est[0,:])**2/np.sum(np.abs(coef_est[0,:])**2))<.99),NestPaths[1])
    #NdisambPaths=Npath
    NdisambPaths=NestPaths[0]
    aod_disp=np.zeros((Nu,NdisambPaths))
    aoa_disp=np.zeros((Nu,NdisambPaths))
    del_disp=np.zeros((Nu,NdisambPaths))
    hdispall=np.zeros((Nu,Nt,Na,Nd),dtype=np.complex64)
    #process first user, just pilot CS estimation
    del_disp[0,:]=del_est[0,0:NdisambPaths]
    aod_disp[0,:]=aod_est[0,0:NdisambPaths]
    aoa_disp[0,:]=aoa_est[0,0:NdisambPaths]
    hdispall[0,...]=np.fft.ifft(hestall[0,...],Nk,axis=0)
    #process second user, use pilot CS estimation to disambiguate angles of first user in range (aod[0,:]>np.pi/2)&(aod[0,:]<np.pi*3/2)
     
    corr=np.zeros((NdisambPaths,NestPaths[1]))
    for p0 in range(NdisambPaths):#find the most similar path in second user for each path of first user
        for p1 in range(NestPaths[1]):
            corr[p0,p1]=np.abs(del_est[0,p0]-del_est[1,p1])/Ts/Nt+np.abs(aod_est[0,p0]-aod_est[1,p1])/np.pi+np.abs(aoa_est[0,p0]-aoa_est[1,p1])/np.pi+np.abs(coef_est[0,p0]-coef_est[1,p1])**2*(1/np.abs(coef_est[0,p0])**2+1/np.abs(coef_est[1,p1])**2)
    closest_path=np.argmin(corr,axis=1)
    aod_matched=aod_est[1,closest_path]
    aoa_matched=aoa_est[1,closest_path]
    del_matched=del_est[1,closest_path]
    
    #special case in degenerated lobe
    bDegenA = np.abs(aoa_disp[0,:])>np.arcsin(1-1/Na)
    bDegenD = np.abs(aod_disp[0,:])>np.arcsin(1-1/Nd)
    #in some rare cases degenerated lobes get estimated with opposite sign
    bOpSgA=(aoa_matched-aoa_disp[0,:])>np.arcsin(1-1/Na)
    bOpSgD=(aod_matched-aod_disp[0,:])>np.arcsin(1-1/Nd)
    #this just makes the sign equal in both degenerated lobes, but they may be both inverted
    aoa_matched[bDegenA&bOpSgA]=-aoa_matched[bDegenA&bOpSgA]
    aoa_matched[bDegenA&bOpSgA]=-aoa_matched[bDegenA&bOpSgA]
    
#    #fix degenerate angles that may be inverted
#    
#    (bFixDgAoA,bFixDgAoD)= fixDegenAngles(del_est[0,0:NdisambPaths],aod_est[0,0:NdisambPaths],aoa_est[0,0:NdisambPaths],del_matched,aod_matched,aoa_matched,xstep[0],ystep[0])    
#    bFixDgAoA=bFixDgAoA&bDegenA
#    bFixDgAoD=bFixDgAoD&bDegenD
#    bFixDgAoDext=np.concatenate((bFixDgAoD,np.zeros(maxNestPaths-bFixDgAoD.size, dtype=bool)))
#    bFixDgAoAext=np.concatenate((bFixDgAoA,np.zeros(maxNestPaths-bFixDgAoA.size, dtype=bool)))
#    aod_disp[0,bFixDgAoD]=-aod_est[0,bFixDgAoDext]
#    aoa_disp[0,bFixDgAoA]=-aoa_est[0,bFixDgAoAext]
#    aoa_matched[bFixDgAoA]=-aoa_matched[bFixDgAoA]
#    aoa_matched[bFixDgAoA]=-aoa_matched[bFixDgAoA]
    
    #compute the "mirrored" angles
    (bFlipAoA,bFlipAoD)= disambiguateAngles(del_est[0,0:NdisambPaths],aod_est[0,0:NdisambPaths],aoa_est[0,0:NdisambPaths],del_matched,aod_matched,aoa_matched,xstep[0],ystep[0])    
#    bFlipAoA=bFlipAoA&~bDegenA
#    bFlipAoD=bFlipAoD&~bDegenD
    bFlipAoDext=np.concatenate((bFlipAoD,np.zeros(maxNestPaths-bFlipAoD.size, dtype=bool)))
    bFlipAoAext=np.concatenate((bFlipAoA,np.zeros(maxNestPaths-bFlipAoA.size, dtype=bool)))
    aod_disp[0,bFlipAoD]=np.pi-aod_est[0,bFlipAoDext]
    aoa_disp[0,bFlipAoA]=np.pi-aoa_est[0,bFlipAoAext]
    
    #finally estimate all channels using only first user channel estimation and trigonometry
    for nu in range(0,Nu-1):
        del_disp[nu+1,:],aod_disp[nu+1,:],aoa_disp[nu+1,:]=displaceMultipathChannel(del_disp[nu,0:NdisambPaths],aod_disp[nu,0:NdisambPaths],aoa_disp[nu,0:NdisambPaths],xstep[nu],ystep[nu])
        hdispall[nu+1,...]=chgen.computeDEC((del_disp[nu+1,:]-d1/c)/Ts-clock_offset,aod_disp[nu+1,:],aoa_disp[nu+1,:],coef_est[0,0:NdisambPaths])
        
    hkdispall=np.fft.fft(hdispall,Nk,axis=1)
    NMSEdisp[isim,:]=np.sum(np.abs(hkdispall-hkall)**2,axis=(1,2,3))/np.sum(np.abs(hkall)**2,axis=(1,2,3))
    print(NMSEdisp[isim,:])
    
    #verify the disambiguation flipping behavior
    corrTrue=np.zeros((NdisambPaths,Npath))
    for p0 in range(NdisambPaths):
        for p1 in range(Npath):
            corrTrue[p0,p1]=np.abs(del_est[0,p0]-tdelay[0,p1])/Ts/Nt+np.abs(aod_est[0,p0]-np.arcsin(np.sin(aod[0,p1])))/np.pi+np.abs(aoa_est[0,p0]-np.arcsin(np.sin(aoa[0,p1])))/np.pi#+...
    closest_truePath=np.argmin(corrTrue,axis=1)
    bFlipAoD_true=(aod[0,closest_truePath]>np.pi/2)&((aod[0,closest_truePath]<np.pi*3/2))
    bFlipAoA_true=(aoa[0,closest_truePath]>np.pi/2)&((aoa[0,closest_truePath]<np.pi*3/2))

    aodFlipWeight[isim]=np.sum((bFlipAoD_true!=bFlipAoD)*np.abs(coef_est[0,0:NdisambPaths])**2)
    aoaFlipWeight[isim]=np.sum((bFlipAoA_true!=bFlipAoA)*np.abs(coef_est[0,0:NdisambPaths])**2)
    aodFlipErrors[isim]=np.sum((bFlipAoD_true!=bFlipAoD))
    aoaFlipErrors[isim]=np.sum((bFlipAoA_true!=bFlipAoA))

plt.plot(np.sort(NMSEdisp[:,1],axis=0),np.linspace(0,1,Nsim))