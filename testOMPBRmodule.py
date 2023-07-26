#!/usr/bin/python

from CASTRO5G import OMPCachedRunner as oc
import MIMOPilotChannel as pil
import csProblemGenerator as prb

import matplotlib.pyplot as plt
import numpy as np

import time
from progress.bar import Bar

plt.close('all')

Nchan=5
Nd=4
Na=4
Nt=64
Nxp=3
Nrft=1
Nrfr=2
K=64
Ts=2.5
Ds=Ts*Nt
SNRs=10**(np.arange(-1,2.01,1.0))
#SNRs=10**(np.arange(1,1.01,1.0))
# SNRs=np.array([100])
MSE=[]
Npaths=[]
totTimes=[]

def pseudoOMP(v,xi):
    #assumes v is explicitly sparse and returns its N largest coefficients where rho(N)<xi
    (Nt,Nd,Na)=np.shape(v)
    r=v
    et=1j*np.zeros(np.shape(r))
    #at least one iteration
    ind=np.unravel_index(np.argmax(np.abs(r)),np.shape(r))
    et[ind]=r[ind]
    r=v-et
    ctr=1
    while np.sum(np.abs(r)**2)>xi:
        ind=np.unravel_index(np.argmax(np.abs(r)),np.shape(r))
        et[ind]=r[ind]
        r=v-et
        ctr=ctr+1
#    print('OMP ctr %d'%ctr)
    e=et
    return(e)
    
def pseudoISTA(v,xi,Niter,horig):
    #assumes v is explicitly sparse and returns its N largest coefficients where rho(N)<xi
    (Nt,Nd,Na)=np.shape(v)
    r=v
    et=np.zeros(np.shape(r))
    beta=.5
    for n in range(Niter):
        c=et+beta*r        
        et=np.exp(1j*np.angle(c))*np.maximum(np.abs(c)-xi,0)
        r=v-et
#        print(np.mean(np.abs(et-horig)**2)/np.mean(np.abs(horig)**2))
    e=et
    return(e)
    
def pseudoFISTA(v,xi,Niter,horig):
    #assumes v is explicitly sparse and returns its N largest coefficients where rho(N)<xi
    (Nt,Nd,Na)=np.shape(v)
    r=v
    et=np.zeros(np.shape(r))
    old_et=et
    beta=.5
    for n in range(Niter):
        c=et+beta*r+(et-old16_et)*(n-2.0)/(n+1.0)
        old_et=et
        et=np.exp(1j*np.angle(c))*np.maximum(np.abs(c)-xi,0)
        r=v-et
#        print(np.mean(np.abs(et-horig)**2)/np.mean(np.abs(horig)**2))
    e=et
    return(e)

def pseudoAMP(v,xi,Niter,horig):
    #assumes v is explicitly sparse and returns its N largest coefficients where rho(N)<xi
    (Nt,Nd,Na)=np.shape(v)
    r=v
    et=np.zeros(np.shape(r))
    old_r=r
    for n in range(Niter):
        bt=np.sum(np.abs(et)>0)/(Nt*Nd*Na)
        c=et+r
        ldt = xi*np.sqrt(np.sum(np.abs(r)**2)/(Nt*Nd*Na))
        et=np.exp(1j*np.angle(c))*np.maximum(np.abs(c)-ldt,0)
        old_r=r
        r=v-et+bt*old_r
#        print(np.mean(np.abs(et-horig)**2)/np.mean(np.abs(horig)**2))
    e=et
    return(e)

omprunner = oc.OMPCachedRunner()
pilgen = pil.MIMOPilotChannel("UPhase")
probgen = prb.csProblemGenerator(Nt,Nd,Na,Nrft,Nrfr,Nxp,Ts,pilotType ="UPhase", chanModel = "simple")
probgen.pregenerate(Nchan)

legStrAlgs=[
#        'LS',
#        'pseudoOMP',
#        'OMPx1',
#        'OMPx2',
        'OMPBR',
        'OMPBR accel',
#        'pseudoISTA',
#        'ISTAx1',
#        'pseudoFISTA',
#        'FISTAx1',
#        'pseudoAMP',
#        'AMP',
#        'VAMP'
        ]

bar = Bar("CS sims", max=Nchan)
bar.check_tty = False
for ichan in range(0,Nchan):
    (hk,zp,wp,vp)=probgen.listPreparedProblems[ichan]
        
    h=np.fft.ifft(hk,axis=0)
    hsparse=np.fft.ifft(h,axis=1)*np.sqrt(Nd)
    hsparse=np.fft.ifft(hsparse,axis=2)*np.sqrt(Na)
    # print(hk)
    MSE.append([])
    totTimes.append([])
    Npaths.append([])

    zh=probgen.zAWGN((Nt,Na,Nd))

    for isnr in range(0,len(SNRs)):
        sigma2=1.0/SNRs[isnr]
        totTimes[-1].append([])
        MSE[-1].append([])
        Npaths[-1].append([])

        #direct observation with noise
#        hest=hsparse*np.sqrt(Nt)+zh*np.sqrt(sigma2)
#        MSE[-1][-1].append(np.mean(np.abs(hsparse-hest/np.sqrt(Nt))**2)/np.mean(np.abs(hsparse)**2))
#
#        #pseudoOMP sparse component selection
#        t0 = time.time()
#        hest2=pseudoOMP(hest,sigma2*Nt*Na*Nd)
#        MSE[-1][-1].append(np.mean(np.abs(hsparse-hest2/np.sqrt(Nt))**2)/np.mean(np.abs(hsparse)**2))
#        totTimes[-1][-1].append(time.time()-t0)
#        Npaths[-1].append([])        
#        Npaths[-1][-1].append(np.sum(hest2!=0))

        yp=pilgen.applyPilotChannel(hk,wp,vp,zp*np.sqrt(sigma2))
        
#        t0 = time.time()
#        hest3,paths=omprunner.OMPBR(yp,sigma2*Nt*Nxp*Nrfr,ichan,vp,wp,1.0,1.0,1.0,1.0)
#        MSE[-1][-1].append(np.mean(np.abs(hk-hest3)**2)/np.mean(np.abs(hk)**2))
#        totTimes[-1][-1].append(time.time()-t0)
#        Npaths[-1][-1].append(len(paths.delays))
#
#        t0 = time.time()
#        hest4,paths=omprunner.OMPBR(yp,sigma2*Nt*Nxp*Nrfr,ichan,vp,wp,2.0,2.0,2.0,1.0)
#        MSE[-1][-1].append(np.mean(np.abs(hk-hest4)**2)/np.mean(np.abs(hk)**2))
#        totTimes[-1][-1].append(time.time()-t0)
#        Npaths[-1][-1].append(len(paths.delays))

        t0 = time.time()
        hest5,paths=omprunner.OMPBR(yp,sigma2*Nt*Nxp*Nrfr,ichan,vp,wp,4.0,1.0,1.0,1.0)
        MSE[-1][-1].append(np.mean(np.abs(hk-hest5)**2)/np.mean(np.abs(hk)**2))
        totTimes[-1][-1].append(time.time()-t0)
        Npaths[-1][-1].append(len(paths.delays))
        
        
        t0 = time.time()
        hest6,paths2=omprunner.OMPBR(yp,sigma2*Nt*Nxp*Nrfr,ichan,vp,wp,4.0,1.0,1.0,1.0, accelDel = True)
        MSE[-1][-1].append(np.mean(np.abs(hk-hest6)**2)/np.mean(np.abs(hk)**2))
        totTimes[-1][-1].append(time.time()-t0)
        Npaths[-1][-1].append(len(paths.delays))
#        
#        t0 = time.time()
#        hest6=pseudoISTA(hest,.5*np.sqrt(sigma2),15,hsparse*np.sqrt(Nt))
#        MSE[-1][-1].append(np.mean(np.abs(hsparse-hest6/np.sqrt(Nt))**2)/np.mean(np.abs(hsparse)**2))
#        totTimes[-1][-1].append(time.time()-t0)
#        Npaths[-1][-1].append(np.sum(hest6!=0))
        
#        t0 = time.time()
#        hest7,paths=omprunner.Shrinkage(yp, (.5*np.sqrt(sigma2), .5) ,15,ichan,vp,wp,1.0,1.0,1.0,'ISTA')
#        MSE[-1][-1].append(np.mean(np.abs(hk-hest7)**2)/np.mean(np.abs(hk)**2))
#        totTimes[-1][-1].append(time.time()-t0)
#        Npaths[-1][-1].append(len(paths.delays))
#        
#        t0 = time.time()
#        hest8,paths=omprunner.Shrinkage(yp, (.5*np.sqrt(sigma2), .5) ,15,ichan,vp,wp,2.0,2.0,2.0,'ISTA')
#        MSE[-1][-1].append(np.mean(np.abs(hk-hest8)**2)/np.mean(np.abs(hk)**2))
#        totTimes[-1][-1].append(time.time()-t0)
#        Npaths[-1][-1].append(len(paths.delays))
        
#        t0 = time.time()
#        hest9=pseudoFISTA(hest,.5*np.sqrt(sigma2),15,hsparse*np.sqrt(Nt))
#        MSE[-1][-1].append(np.mean(np.abs(hsparse-hest9/np.sqrt(Nt))**2)/np.mean(np.abs(hsparse)**2))
#        totTimes[-1][-1].append(time.time()-t0)
#        Npaths[-1][-1].append(np.sum(hest6!=0))
#        
#        t0 = time.time()
#        hest10,paths=omprunner.Shrinkage(yp, (.5*np.sqrt(sigma2),.5),15,ichan,vp,wp,1.0,1.0,1.0,'FISTA')
#        MSE[-1][-1].append(np.mean(np.abs(hk-hest10)**2)/np.mean(np.abs(hk)**2))
#        totTimes[-1][-1].append(time.time()-t0)
#        Npaths[-1][-1].append(len(paths.delays))
#        
#        t0 = time.time()
#        hest11=pseudoAMP(hest,.5*np.sqrt(sigma2),15,hsparse*np.sqrt(Nt))
#        MSE[-1][-1].append(np.mean(np.abs(hsparse-hest11/np.sqrt(Nt))**2)/np.mean(np.abs(hsparse)**2))
#        totTimes[-1][-1].append(time.time()-t0)
#        Npaths[-1][-1].append(np.sum(hest6!=0))
#        
#        t0 = time.time()
#        hest12,paths=omprunner.Shrinkage(yp,(.5*np.sqrt(sigma2),),15,ichan,vp,wp,1.0,1.0,1.0,'AMP')
#        MSE[-1][-1].append(np.mean(np.abs(hk-hest12)**2)/np.mean(np.abs(hk)**2))
#        totTimes[-1][-1].append(time.time()-t0)
#        Npaths[-1][-1].append(len(paths.delays))
#        
#        t0 = time.time()
#        hest13,paths=omprunner.Shrinkage(yp,(sigma2,5),5,ichan,vp,wp,1.0,1.0,1.0,'VAMP')
#        MSE[-1][-1].append(np.mean(np.abs(hk-hest13)**2)/np.mean(np.abs(hk)**2))
#        totTimes[-1][-1].append(time.time()-t0)
#        Npaths[-1][-1].append(len(paths.delays))
    bar.next()
bar.finish()

MSE=np.array(MSE)
totTimes=np.array(totTimes)
print(np.mean(MSE,axis=0))
print(np.mean(totTimes,axis=0))
plt.semilogy(10*np.log10(SNRs),np.mean(MSE,axis=0))
plt.legend(legStrAlgs)
plt.xlabel('SNR(dB)')
plt.ylabel('MSE')
plt.figure()
plt.bar(range(np.size(totTimes)//Nchan),np.mean(totTimes,axis=0).reshape(np.size(totTimes)//Nchan,))
plt.xlabel('alg')
plt.ylabel('runtime')
plt.figure()
plt.bar(range(np.size(totTimes)//Nchan),np.mean(Npaths,axis=0).reshape(np.size(Npaths)//Nchan,))
plt.xlabel('alg')
plt.ylabel('N paths')
plt.show()
