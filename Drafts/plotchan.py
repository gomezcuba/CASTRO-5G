#!/usr/bin/python


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import animation, rc
from IPython.display import HTML, Image
# equivalent to rcParams['animation.html'] = 'html5'
rc('animation', html='html5')

import sys
sys.path.append('../')
from CASTRO5G import threeGPPMultipathGenerator as pg

model = pg.ThreeGPPMultipathChannelModel(bLargeBandwidthOption=False)
plinfo,macro,clusters,subpaths = model.create_channel((0,0,10),(40,0,1.5))
tau,powC,AOA,AOD,ZOA,ZOD = clusters.T.to_numpy()
nClusters=tau.size
los, PLfree, SF = plinfo
tau_sp,pow_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp,XPR_sp,phase00,phase01,phase10,phase11 =  subpaths.T.to_numpy()

fig = plt.figure(1)
ax = Axes3D(fig)
t=np.linspace(0,2*np.pi,100)
for dbref in range(4):
    ax.plot3D(40*(1-dbref/4)*np.cos(t),40*(1-dbref/4)*np.sin(t),np.zeros_like(t),color='k')
    ax.text3D(0,-40*(1-dbref/4),0,'%s dB'%(-10*dbref),color='k')
    
t=np.linspace(0,42,100)
for labeltheta in np.arange(0,2*np.pi,2*np.pi/8):
    ax.plot3D(t*np.cos(labeltheta),t*np.sin(labeltheta),-np.ones_like(t),color='k')
    ax.text3D(42*np.cos(labeltheta),42*np.sin(labeltheta),-1,'%.2f pi'%(labeltheta/np.pi),color='k')

ax.text3D(42*np.cos(np.pi/16),42*np.sin(np.pi/16),-1,'AoA',color='k')
maxdel=np.max(tau_sp)
ax.plot3D([0,0],[0,0],[0,np.ceil(maxdel/100)*100],color='k')
ax.text3D(0,0,np.ceil(maxdel/100)*100,"delay [ns]",color='k')

allAoA=AOA_sp.reshape(-1)
allDel=tau_sp.reshape(-1)*1e9
allGain=pow_sp.reshape(-1)

inds=np.argpartition(-allGain,4,axis=0)[0:50]
selectedAoA=allAoA[inds]
selectedDel=allDel[inds]
selectedGain=allGain[inds]

for pind in range(0,len( selectedAoA )):
    AoA=selectedAoA[pind]
    delay=selectedDel[pind]
    gain=selectedGain[pind]
    x=np.maximum(10*np.log10(gain)+40,0)*np.cos(AoA)
    y=np.maximum(10*np.log10(gain)+40,0)*np.sin(AoA)
    ax.plot3D([0,x],[0,y],[delay,delay])
    ax.scatter3D(x,y,delay,marker='o')
    
plt.savefig('animation_frame0.png')
ax.view_init(azim=-61)
plt.savefig('animation_frame1.png')

import os
os.system('convert -delay 10 -loop 0 animation_frame0.png animation_frame1.png animated.gif')

from matplotlib import animation, rc
rc('animation', html='html5')
# animation function. This is called sequentially
def animate(i):
    if ax.azim==-60:
        ax.view_init(azim=-61)
    else:
        ax.view_init(azim=-60)
    return (ax,)

anim = animation.FuncAnimation(fig, animate, frames=2, interval=100)

anim.save('./testanim.gif', writer='imagemagick', fps=10, progress_callback =  lambda i, n: print(f'Saving frame {i} of {n}'))

#import imageio
#frames = ['animation_frame0.png','animation_frame1.png']
#imageio.mimsave('animation.gif', images)ç
#def animate(i):
#    if ax.azim==-60:
#        ax.view_init(azim=-61)
#    else:
#        ax.view_init(azim=-60)
#    ax.figure.canvas.draw()
#    return(ax,)
#    
#anim = animation.FuncAnimation(fig, animate, frames=2, interval=100, blit=True)
#anim.save('./animation.gif', writer='imagemagick', fps=10, progress_callback =  lambda i, n: print(f'Saving frame {i} of {n}'))
#anim.save('./animation.mp4', writer='ffmpeg', fps=30, progress_callback =  lambda i, n: print(f'Saving frame {i} of {n}'))
#fig2 = plt.figure(2)
#Nt=128
#mpchan=next(iter(model.dChansGenerated.values()))
#Ds=np.max([x.excessDelay for x in mpchan.channelPaths])
#Ts=Ds/Nt
#Nd=64
#Na=64
#h=mpchan.getDEC(Na,Nd,Nt,Ts)
#h=np.fft.fft(h,axis=0)
#h=np.fft.fft(h,axis=1)
#
#sigma2=1e-4
#z=( np.random.normal(size=(Na,1,Nt)) + 1j*np.random.normal(size=(Na,1,Nt)) ) * np.sqrt( sigma2 / 2.0 )
#y=h+z
#plt.stem(np.arange(0,Ts*Nt,Ts),np.abs(h[0,0,:]),markerfmt='ob')
#plt.stem(np.arange(0,Ts*Nt,Ts),np.abs(y[0,0,:]),markerfmt='xr')

#for nf in range(0,(Nt//16)):
#    fig3 = plt.figure(3+nf)
#    ax = Axes3D(fig3)
#    X = np.arange(0, Na, 1)
#    Y = np.arange(0, Nd, 1)
#    X, Y = np.meshgrid(X, Y)
#    Vaux=10*np.log10(np.abs(h[:,:,:])**2)
#    Vaux=Vaux-np.min(Vaux.flatten())
#    Vaux=Vaux/np.max(Vaux.flatten())
#    Call=cm.coolwarm(Vaux)
#    for dind in range(0, 16):
#        C=Call[:,:,dind+16*nf]
#        # C = cm.cool(np.sum(np.abs(h[:,:,:])**2,axis=2))
#        Z = Ts*(dind-1)*np.ones((Na,Nd))
#        surf = ax.plot_surface(X, Y, Z, facecolors=C, linewidth=0, antialiased=False)

#plt.show()
#plt.close('all')
#exit()
