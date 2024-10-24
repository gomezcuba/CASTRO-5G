#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

import bisect

import sys
sys.path.append('../')
from CASTRO5G import multipathChannel as mc
pilgen = mc.MIMOPilotChannel("IDUV")
channelResponseFunctions = {
    # "TDoA" : mc.pSinc,
    "TDoA" : lambda t,M: np.fft.ifft(mc.pCExp(t,M)),  
    "AoA" : mc.fULA,
    "AoD" : mc.fULA,
    }
K=2048
Ncp=256
Na=16
Nd=16
Nframe=3
Tcp=570e-9 #mu=2
Ts=Tcp/Ncp
mpch = mc.MultipathDEC((0,0,10),(40,0,1.5),customResponse=channelResponseFunctions)
chgen=mc.UniformMultipathChannelModel(Npath=3,Ds=0.9*Tcp,mode3D=False)
pathsData=chgen.create_channel(1)
mpch.insertPathsFromDF(pathsData.loc[0,:])
ht=mpch.getDEC(Na,Nd,Ncp,Ts)*np.sqrt(Nd*Na)#mpch uses normalized matrices of gain 1
hk=np.fft.fft(ht,K,axis=0)
wp,vp=pilgen.generatePilots(Nframe*K,Na,Nd,Npr=Nframe*K,rShape=(Nframe,K,1,Na),tShape=(Nframe,K,Nd,1))
zp=mc.AWGN((K,Na,1))
yp=pilgen.applyPilotChannel( hk,wp,vp,None)
Hsizes=[K,Ncp,Na,Nd]
Lsizes=[4*Ncp,4*Na,4*Nd]

def projYSphere(yp,pilots,dimH,dimPhi):
    wp,vp=pilots
    qp=np.matmul(wp.transpose([0,1,3,2]).conj(),np.matmul(yp,vp.transpose([0,1,3,2]).conj()))
    qp = np.sum(qp,axis=0)
    
    Uini = np.sum(np.abs(qp)**2)
    # Uini = np.linalg.norm(qp)
    mp = {(-1,-1,-1):(0,Uini,qp)}
    mpr = { -Uini:(-1,-1,-1)}
    lU = [-Uini]       
    K,Ncp,Na,Nd = dimH
    Lt,La,Ld = dimPhi
    # print(mp)
    stop=False
    while not stop:
        # it=max(mp,key=lambda x: mp.get(x)[1])
        it=mpr[lU[0]]
        d,Uprev,Qprev = mp[it]
        if d==3:
            stop=True
            break;
        elif d==0:
            Kexpand=int(K*Lt/Ncp)
            val = np.fft.ifft(Qprev,Kexpand,axis=0,norm="forward")[0:Lt,:,:]/np.sqrt(K)
            Unext = np.sum(np.abs(val)**2,axis=(1,2))
            # Unext = np.linalg.norm(val,axis=(1,2))
        elif d==1:
            val = np.fft.ifft(Qprev,La,axis=0,norm="forward")/np.sqrt(Na)
            Unext = np.sum(np.abs(val)**2,axis=1)
            # Unext = np.linalg.norm(val,axis=1)
        elif d==2:
            val = np.fft.ifft(Qprev,Ld,axis=0,norm="forward")/np.sqrt(Nd)
            Unext = np.abs(val)**2
            # Unext = np.abs(val)
        
        mp.pop(it)
        mpr.pop(-Uprev)
        lU=lU[1:]
        for n in range(dimPhi[d]):
            it_next=list(it)
            it_next[d]=n
            mp[tuple(it_next)]=(d+1,Unext[n],val[n,...])
            mpr[-Unext[n]]=tuple(it_next)
            bisect.insort(lU,-Unext[n])
    # print(f"""ITER
          
    # #       {mp}""")
    C=np.zeros((Lt,La,Ld),dtype=np.complex128)
    for a in mp.keys():
        d,U,c=mp[a]      
        ind_list=int(np.ravel_multi_index(a[0:d],dimPhi[0:d])*np.prod(C.shape[d:]))
        if d<3:
            ind_list=ind_list+np.arange(np.prod(C.shape[d:]),dtype=int)
            C[np.unravel_index(ind_list,dimPhi)]=np.sqrt(U)
        else:
            C[np.unravel_index(ind_list,dimPhi)]=c
    return(it,np.sqrt(Uprev),C.reshape(-1),mp)

# %timeit projYSphere(yp,(wp,vp),Hsizes,Lsizes)
it,U,c,mp=projYSphere(yp,(wp,vp),Hsizes,Lsizes)

print(len(mp.keys()),8*Ncp*Na*Nd)

def projYFull(yp,pilots,dimH,dimPhi):
    qp=np.matmul(wp.transpose([0,1,3,2]).conj(),np.matmul(yp,vp.transpose([0,1,3,2]).conj()))
    qp = np.sum(qp,axis=0)
    K,Ncp,Na,Nd = dimH
    Lt,La,Ld = dimPhi
    Kexpand=int(K*Lt/Ncp)
    v2=np.fft.ifft(qp,Kexpand,axis=0,norm="forward")[0:Lt,:,:]
    v3=np.fft.ifft(v2,La,axis=1,norm="forward")
    v4=np.fft.ifft(v3,Ld,axis=2,norm="forward")
    c=v4/np.sqrt(K*Na*Nd)
    it=np.unravel_index(np.argmax(np.abs(c)),shape=dimPhi)
    return(it,np.max(np.abs(c)),c.reshape(-1,1))


# %timeit projYFull(yp,(wp,vp),Hsizes,Lsizes)
it2,U2,c2=projYFull(yp,(wp,vp),Hsizes,Lsizes)
print(it,it2)
print(U,U2)
plt.close('all')
plt.plot(np.abs(c2),'r')
plt.plot(np.abs(c),'b:')