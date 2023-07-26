#!/usr/bin/python

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
plt.close('all')

from CASTRO5G import multipathChannel as mc
import MIMOPilotChannel as pil

Nant = 32
Nrf = 1
mindBpolar = -30
Nplotpoint =1000
angles = np.arange(0,2*np.pi,2*np.pi/Nplotpoint)
h_array = mc.fULA(angles,Nant,.5)

pilGntr = pil.MIMOPilotChannel()

plt.figure(1)
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

plt.legend()
plt.savefig('beamDiagCompare-%d-%d-%d.eps'%(Nant,Nrf,Nsectors))

plt.figure(2)

plt.polar(angles,np.maximum(10*np.log10(np.abs(G_fft[:,0,:])**2),mindBpolar))
plt.legend(['DFT $k=%d$'%k for k in range(Nant)])
plt.savefig('beamDiagDFTall-%d.eps'%(Nant))

plt.figure(3)

plt.polar(angles,np.maximum(10*np.log10(np.abs(G_ls[:,0,:])**2),mindBpolar))
plt.legend(['Rectangular beam $k=%d$'%k for k in range(Nant)])
plt.savefig('beamDiagSectorall-%d-%d.eps'%(Nant,Nsectors))

# plt.figure(4)
# plt.imshow(desired_G)