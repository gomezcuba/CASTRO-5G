#!/usr/bin/python

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
plt.close('all')

from CASTRO5G import multipathChannel as mc
import MIMOPilotChannel as pil

fig_ctr=0

Nant = 32
Nrf = 1
mindBpolar = -30
Nplotpoint =1000
angles = np.arange(0,2*np.pi,2*np.pi/Nplotpoint)
h_array = mc.fULA(angles,Nant,.5)

pilGntr = pil.MIMOPilotChannel()


fig_ctr+=1
fig = plt.figure(fig_ctr)
w_eye,v_eye = pil.MIMOPilotChannel.generatePilotsEye(None,(1,1,Nrf,Nant,Nant,Nrf))
v_eye = v_eye[0,0,:,:]
g_eye= h_array.transpose([0,2,1]).conj() @ v_eye
plt.polar(angles,np.maximum(10*np.log10(np.abs(g_eye[:,0,0])**2),mindBpolar),label='1-hot (omni)')

V_fft = np.fft.fft(np.eye(Nant))/np.sqrt(Nant)
G_fft= h_array.transpose([0,2,1]).conj() @ V_fft
# G_fft = np.fft.fft(h_array.transpose([0,2,1]).conj(),axis=2)/np.sqrt(Nant)#direct equivalent
plt.polar(angles,np.maximum(10*np.log10(np.abs(G_fft[:,0,0])**2),mindBpolar),label='DFT $k=0$')
plt.polar(angles,np.maximum(10*np.log10(np.abs(G_fft[:,0,2])**2),mindBpolar),label='DFT $k=2$')

Ndesiredgains = 100
angles_design = np.arange(-np.pi/2,np.pi/2,np.pi/Ndesiredgains)
Nsectors = 8
desired_G = np.zeros((Ndesiredgains,Nsectors))
for sec in range(Nsectors):
    k = np.mod(sec+Nsectors/2,Nsectors)
    mask1 = angles_design >= (k-.5)*np.pi/Nsectors -np.pi/2
    mask2 = angles_design < (k+.5)*np.pi/Nsectors -np.pi/2
    if k == 0:
        mask2 = mask2 | (angles_design >= np.pi/2-.5*np.pi/Nsectors)
    desired_G[mask1 & mask2,sec] = 1
h_array_design = mc.fULA(angles_design,Nant,.5)

V_ls,_,_,_=np.linalg.lstsq(h_array_design[:,:,0].conj(),desired_G,rcond=None)
V_ls=V_ls/np.linalg.norm(V_ls,axis=0)
G_ls= h_array.transpose([0,2,1]).conj() @ V_ls
plt.polar(angles,np.maximum(10*np.log10(np.abs(G_ls[:,0,0])**2),mindBpolar),label='LS $k=0$')
plt.polar(angles,np.maximum(10*np.log10(np.abs(G_ls[:,0,2])**2),mindBpolar),label='LS $k=2$')

#PMI 5GNR test
i11=0
i12=0
i13=0

def vlm(l,m,n,N1,N2,O1=4,O2=4):
    um=np.exp(2j*np.pi*m*np.arange(N2)[:,None]/(N2*O2))
    vl=np.exp(2j*np.pi*l*np.arange(N1)[:,None]/(N1*O1)) 
    phin=np.exp(.5j*np.pi*n)
    return(np.kron(np.vstack([vl,phin*vl]),um))

def cb3GPPtypeImode1(PMI,N1,N2,NL):
    NAP=N1*N2*2
    if len(PMI)==4:
        i11,i12,i13,i2 = PMI
        k1,k2=(4,4)#TODO implement tableK1K2vsi13 Table 5.2.2.2.1-3 and Table 5.2.2.2.1-4
    else:
        i11,i12,i2 = PMI
        
    
    if NL == 1:
        return( vlm(i11,i12,i2,N1,N2) )
    elif NL == 2:
        return( np.hstack([
                vlm(i11,i12,i2,N1,N2),
                vlm(i11+k1,i12+k2,i2+2,N1,N2)
            ] ) )
    #TODO elif NL == 3:
    #TODO elif NL == 4:
    elif NL == 5:
        return( np.hstack([
                vlm(i11,i12,i2,N1,N2),
                vlm(i11,i12,i2+2,N1,N2),
                vlm(i11+4,i12,0,N1,N2),
                vlm(i11+4,i12,2,N1,N2),
                vlm(i11+4,i12+4,0,N1,N2)
            ] ) )
    elif NL == 6:        
        return( np.hstack([
                vlm(i11,i12,i2,N1,N2),
                vlm(i11,i12,i2+2,N1,N2),
                vlm(i11+4,i12,i2,N1,N2),
                vlm(i11+4,i12,i2+2,N1,N2),
                vlm(i11+4,i12+4,0,N1,N2),
                vlm(i11+4,i12+4,2,N1,N2)
            ] ) )
    #TODO elif NL == 7:                
    #TODO elif NL == 8:
    else:
        return( np.array([]) )

plt.legend()
plt.savefig('beamDiagCompare-%d-%d-%d.eps'%(Nant,Nrf,Nsectors))

fig_ctr+=1
fig = plt.figure(fig_ctr)

plt.polar(angles,np.maximum(10*np.log10(np.abs(G_fft[:,0,:])**2),mindBpolar))
plt.legend(['DFT $k=%d$'%k for k in range(Nant)])
plt.savefig('beamDiagDFTall-%d.eps'%(Nant))

fig_ctr+=1
fig = plt.figure(fig_ctr)

plt.polar(angles,np.maximum(10*np.log10(np.abs(G_ls[:,0,:])**2),mindBpolar))
plt.legend(['Rectangular beam $k=%d$'%k for k in range(Nant)])
plt.savefig('beamDiagSectorall-%d-%d.eps'%(Nant,Nsectors))

# fig_ctr+=1
# fig = plt.figure(fig_ctr)
# plt.figure(4)
# plt.imshow(desired_G)