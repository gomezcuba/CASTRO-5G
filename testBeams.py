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

tabItoK2L = {
            "N1>N2=1":[
                [0,0],
                [4,0],
                [8,0],
                [12,0],
            ],
            "N1=N2":[
                [0,0],
                [4,0],
                [0,4],
                [4,4],
            ],      
            "N1>N2>1":[
                [0,0],
                [4,0],
                [0,4],
                [8,0],
            ],      
    }
tabItoK4L = {
            "N2=1":[
                [4,0],
                [8,0],
                [12,0],
                [16,0],
            ],      
            "N2>1":[
                [4,0],
                [0,4],
                [4,4],
                [8,0],
            ],      
    }

def vlm(l,m,n,N1,N2,O1=4,O2=4):
    um=np.exp(2j*np.pi*m*np.arange(N2)[:,None]/(N2*O2))
    vl=np.exp(2j*np.pi*l*np.arange(N1)[:,None]/(N1*O1)) 
    phin=np.exp(.5j*np.pi*n)
    return(np.kron(np.vstack([vl,phin*vl]),um))

def Wlms(vl,vm,vn,N1,N2,O1=4,O2=4):
    l=[]
    for c in range(len(vl)):
        l.append(vlm(vl[c],vm[c],vn[c],N1,N2,O1,O2))
    return(np.hstack(l))

def cb3GPPtypeImode1(PMI,N1,N2,NL,mode=1):
    NAP=N1*N2*2
    if len(PMI)==4:
        i11,i12,i13,i2 = PMI
        k1,k2=(4,4)#TODO implement tableK1K2vsi13 Table 5.2.2.2.1-3 and Table 5.2.2.2.1-4
    else:
        i11,i12,i2 = PMI        
    #this hack remaps numbers l m and n of mode 2 to their mode 1 equivalents for NL={1,2}
    if (mode==2) and (NL < 3):
        if N2>1:            
            i11=2*i11 + (i2 & 0b0100 >>2)
            i11=2*i11 + (i2 & 0b1000 >>3)
            i2=i2 & 0b0011 #note that mode=2 with N2>1 uses the same beams as mode=1, with different binary index
        else:  
            i11=2*i11 + (i2 & 0b1100 >>2) #note that i2+8 and i11+1 are the same beam. THIS IS USELESS
            i11=2*i11 
            i2=i2 & 0b0011 
    #-------end hack-------  
    DXO= 4 if N2>1 else 8 #i11 3rd neighbor beam 5 6 layers
    D2O= 0 if N2>1 else 8 #i11 3rd neighbor beam 7 8 layers
    D3O= 4 if N2>1 else 12#i11 4th neighbor beam 7 8 layers
    DVO= 4 if N2>1 else 0 #i12 3rd and 4th neighbor beam 5-8 layers
    if NL == 1:
        vl=[i11]
        vm=[i12]
        vn=[i2] 
    elif NL == 2:
        vl=[i11,i11+k1]
        vm=[i12,i12+k2]
        vn=[i2 ,i2+2  ]        
    elif NL == 3:
        if NAP<16:            
            vl=[i11,i11+k1,i11  ]
            vm=[i12,i12+k2,i12  ]
            vn=[i2 ,i2    ,i2+2 ]        
        #TODO else:
    elif NL == 4:
        if NAP<16:            
            vl=[i11,i11+k1,i11  ,i11+k1]
            vm=[i12,i12+k2,i12  ,i12+k2]
            vn=[i2 ,i2    ,i2+2 , i2+2 ]        
        #TODO else:
    elif NL == 5:
        vl=[i11,i11  ,i11+4,i11+4,i11+DXO]
        vm=[i12,i12  ,i12  ,i12  ,i12+DVO]
        vn=[i2 ,i2+2 ,0    ,2    ,0      ]
    elif NL == 6:
        vl=[i11 ,i11 ,i11+4,i11+4,i11+DXO,i11+DXO]
        vm=[i12 ,i12 ,i12  ,i12  ,i12+DVO,i12+DVO]
        vn=[i2  ,i2+2,i2   ,i2+2 ,0      ,2      ]
    elif NL == 7:
        vl=[i11 ,i11 ,i11+4,i11+D2O,i11+D2O,i11+D3O,i11+D3O]
        vm=[i12 ,i12 ,i12  ,i12+DVO,i12+DVO,i12+DVO,i12+DVO]
        vn=[i2  ,i2+2,i2   ,0      ,2      ,0      ,2      ]
    elif NL == 8:
        vl=[i11 ,i11 ,i11+4,i11+4,i11+D2O,i11+D2O,i11+D3O,i11+D3O]
        vm=[i12 ,i12 ,i12  ,i12  ,i12+DVO,i12+DVO,i12+DVO,i12+DVO]
        vn=[i2  ,i2+2,i2   ,i2+2 ,0      ,2      ,0      ,2      ]
        
    return( Wlms(vl,vm,vn,N1,N2)/np.sqrt(NL*NAP) )

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