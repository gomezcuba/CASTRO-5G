#!/usr/bin/python

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
mpch = mc.MultipathDEC((0,0,10),(40,0,1.5),customResponse=channelResponseFunctions)
chgen=mc.UniformMultipathChannelModel(Npath=3,Ds=570e-9,mode3D=False)
pathsData=chgen.create_channel(1)
mpch.insertPathsFromDF(pathsData.loc[0,:])
K=2048
Ncp=256
Na=16
Nd=16
Ts=570e-9/Ncp
ht=mpch.getDEC(Na,Nd,Ncp,Ts)*np.sqrt(Nd*Na)#mpch uses normalized matrices of gain 1
hk=np.fft.fft(ht,K,axis=0)
wp,vp=pilgen.generatePilots(K,Na,Nd,Npr=K,rShape=(K,1,Na),tShape=(K,Nd,1))
zp=mc.AWGN((K,Na,1))
yp=pilgen.applyPilotChannel( hk,wp,vp,zp)
Nsizes=[K,Na,Nd]
Lsizes=[2*K,2*Na,2*Nd]

def projYSphere(yp,pilots,dimH,dimPhi):
    wp,vp=pilots
    qp=np.matmul(wp.transpose([0,2,1]).conj(),np.matmul(yp,vp.transpose([0,2,1]).conj()))

    mp = {(-1,-1,-1):(0,np.sum(np.abs(qp)**2),qp)}
    K,Ncp,Na,Nd = dimH
    Lt,La,Ld = dimPhi
    # print(mp)
    stop=False
    while not stop:
        it=max(mp,key=lambda x: mp.get(x)[1])
        d,Uprev,Qprev = mp[it]
        if d==3:
            stop=True
            break;
        elif d==0:
            Kexpand=int(K*Lt/Ncp)
            val = np.fft.ifft(Qprev,Kexpand,axis=0,norm="forward")[0:Lt,:,:]/np.sqrt(K)
            # Unext = np.sum(np.abs(val)**2,axis=(1,2))
            Unext = np.linalg.norm(val,axis=(1,2))
        elif d==1:
            val = np.fft.ifft(Qprev,La,axis=0,norm="forward")/np.sqrt(Na)
            # Unext = np.sum(np.abs(val)**2,axis=1)
            Unext = np.linalg.norm(val,axis=1)
        elif d==2:
            val = np.fft.ifft(Qprev,Ld,axis=0,norm="forward")/np.sqrt(Nd)
            # Unext = np.abs(val)**2
            Unext = np.abs(val)
        
        for n in range(dimPhi[d]):
            it_next=list(it)
            it_next[d]=n
            mp[tuple(it_next)]=(d+1,Unext[n],val[n,...])
        mp.pop(it)
    # print(f"""ITER
          
    # #       {mp}""")
    C=np.zeros((Lt,La,Ld),dtype=np.complex128)
    for a in mp.keys():
        d,U,c=mp[a]      
        ind_list=int(np.ravel_multi_index(a[0:d],dimPhi[0:d])*np.prod(C.shape[d:]))
        if d<3:
            ind_list=ind_list+np.arange(np.prod(C.shape[d:]),dtype=int)  
            C[np.unravel_index(ind_list,dimPhi)]=U
        else:
            C[np.unravel_index(ind_list,dimPhi)]=c
    return(it,Uprev,C.reshape(-1),mp)
    
%timeit projYSphere(yp,(wp,vp),(K,Ncp,Na,Nd),(2*Ncp,2*Na,2*Nd))
it,U,c,mp=projYSphere(yp,(wp,vp),(K,Ncp,Na,Nd),(2*Ncp,2*Na,2*Nd))

print(len(mp.keys()),8*Ncp*Na*Nd)

def projYFull(yp,pilots,dimH,dimPhi):
    qp=np.matmul(wp.transpose([0,2,1]).conj(),np.matmul(yp,vp.transpose([0,2,1]).conj()))
    K,Ncp,Na,Nd = dimH
    Lt,La,Ld = dimPhi
    v2=np.fft.ifft(qp,La,axis=1,norm="forward")
    v3=np.fft.ifft(v2,Ld,axis=2,norm="forward")
    Kexpand=int(K*Lt/Ncp)
    v4=np.fft.ifft(v3,Kexpand,axis=0,norm="forward")[0:Lt,:,:]
    c=v4/np.sqrt(K*Na*Nd)
    it=np.unravel_index(np.argmax(c),shape=dimPhi)
    return(it,np.max(c),c.reshape(-1,1))


%timeit projYFull(yp,(wp,vp),(K,Ncp,Na,Nd),(2*Ncp,2*Na,2*Nd))
it2,U2,c2=projYFull(yp,(wp,vp),(K,Ncp,Na,Nd),(2*Ncp,2*Na,2*Nd))
print(it,it2)
print(U,U2)
plt.close('all')
plt.plot(np.abs(c2),'r')
plt.plot(np.abs(c),'b:')