#!/usr/bin/python
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

import numpy as np

plt.close('all')

import sys
sys.path.append('../')
from CASTRO5G import MultipathLocationEstimator

Npath=10
x_true=np.random.rand(Npath)*40-20
y_true=np.random.rand(Npath)*40-20
x0_true=np.random.rand(1)*40-10
y0_true=np.random.rand(1)*40-10

phi0_true=np.random.rand(1)*2*np.pi
theta0_true=np.mod(np.arctan(y0_true/x0_true)+np.pi*(x0_true<0) , 2*np.pi)

theta_true=np.mod( np.arctan(y_true/x_true)+np.pi*(x_true<0), 2*np.pi)
phi_true=np.mod( np.pi-( np.arctan((y_true-y0_true)/(x0_true-x_true))+np.pi*((x0_true-x_true)<0) ), 2*np.pi)

c=3e8
tau0_true=y0_true/np.sin(theta0_true)/c

tau_true=(np.abs(y_true/np.sin(theta_true))+np.abs((y_true-y0_true)/np.sin(np.pi-phi_true)))/c

#AoD = np.mod(theta_true,2*np.pi)
AoD = np.mod(theta_true,2*np.pi)
AoA = np.mod(phi_true-phi0_true,2*np.pi)
dels = tau_true-tau0_true


plt.figure(1)
plt.plot([0,x0_true],[0,y0_true],':g')
plt.plot(x_true,y_true,'or')
scaleguide=np.max(np.abs(np.concatenate([y_true,y0_true,x_true,x0_true],0)))

plt.plot([x0_true,x0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.cos(phi0_true)],[y0_true,y0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.sin(phi0_true)],'k')
for p in range(np.shape(theta_true)[0]):
    plt.plot([0,x_true[p],x0_true],[0,y_true[p],y0_true],':k')
    t=np.linspace(0,1,21)
    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k')
    plt.plot(x0_true+scaleguide*.05*(p+1)*np.cos(AoA[p]*t+phi0_true),y0_true+scaleguide*.05*(p+1)*np.sin(AoA[p]*t+phi0_true),'k')

plt.plot(0,0,'sb')
plt.plot(x0_true,y0_true,'^g')
plt.title("All angles of a multipath channel with correct tau0, random phi0")

loc=MultipathLocationEstimator.MultipathLocationEstimator()
phi0_search=np.linspace(0,2*np.pi,1000).reshape(-1,1)

x0all=np.zeros((1000,Npath-2))
y0all=np.zeros((1000,Npath-2))
tauEall=np.zeros((1000,Npath-2))

for ct in range(phi0_search.size):
    for gr in range(Npath-2):
        (x0all[ct,gr],y0all[ct,gr],tauEall[ct,gr],vxest,vyest)= loc.computeAllPaths(AoD[gr:gr+3],AoA[gr:gr+3],dels[gr:gr+3],phi0_search[ct])

plt.figure(2)

#with open('BStower.svg', 'r') as f:
#    img_svg = f.read()
#img_png = cairosvg.svg2png(bytestring=img_svg.encode('utf-8'))
#img_tower = OffsetImage(Image.open(BytesIO(img_png)), zoom=.1)
#with open('m2mapplications_logistics.svg', 'r') as f:
#    img_svg = f.read()
#img_png = cairosvg.svg2png(bytestring=img_svg.encode('utf-8'))
#img_mobile = OffsetImage(Image.open(BytesIO(img_png)), zoom=.075)

plt.plot([0,x0_true],[0,y0_true],':g', label='_nolegend_')
plt.plot(x0all,y0all,':', label='_nolegend_')
plt.plot(0,0,'sb',markersize=10)
#img_tower.image.axes=plt.gca()
#abt = AnnotationBbox(img_tower, (0,0), frameon=False, pad=0.0,)
#plt.gca().add_artist(abt)
plt.plot(x0_true,y0_true,'^g',markersize=10)
#img_mobile.image.axes=plt.gca()
#abm = AnnotationBbox(img_mobile, (x0_true,y0_true), frameon=False, pad=0.0,)
#plt.gca().add_artist(abm)
plt.axis([-50,50,-50,50])
plt.xlabel('$d_{ox}$ (m)')
plt.ylabel('$d_{oy}$ (m)')
plt.legend(['Transmitter','Receiver'])
plt.savefig('../Figures/graphsol%d.eps'%(Npath))


from mpl_toolkits import mplot3d
X0e=np.zeros((1000,Npath))
Y0e=np.zeros((1000,Npath))
TauEe=np.zeros((1000,Npath))
for ct in range(phi0_search.size):
    for gr in range(Npath):
        (X0e[ct,gr],Y0e[ct,gr],TauEe[ct,gr],vxest,vyest)=loc.computeAllPaths(AoD[np.arange(Npath)!=gr],AoA[np.arange(Npath)!=gr],dels[np.arange(Npath)!=gr],phi0_search[ct])


(phi0_bisec,x0_bisec,y0_bisec,_,x_bisec,y_bisec,_)= loc.computeAllLocationsFromPaths(AoD,AoA,dels,AoA0_method='brute', group_method='3path')
print(np.mod(phi0_bisec,2*np.pi),phi0_true[0])
    
plt.figure(3)
plt.plot(0,0,'sb')
plt.plot(x0_true,y0_true,'^g')
plt.plot([0,x0_true],[0,y0_true],':g')
plt.plot(x0_bisec,y0_bisec,'^c')
plt.plot([0,x0_bisec],[0,y0_bisec],':c')
plt.plot(x_true,y_true,'or')
plt.plot(x_bisec,y_bisec,'oy')
scaleguide=np.max(np.abs(np.concatenate([y_true,y0_true,x_true,x0_true],0)))
plt.plot([x0_true,x0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.cos(phi0_true)],[y0_true,y0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.sin(phi0_true)],'k')
plt.plot([x0_bisec,x0_bisec+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.cos(phi0_bisec)],[y0_bisec,y0_bisec+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.sin(phi0_bisec)],'m')

# from mpl_toolkits import mplot3d
for p in range(np.shape(theta_true)[0]):
    plt.plot([0,x_true[p],x0_true],[0,y_true[p],y0_true],':k')
    t=np.linspace(0,1,21)
    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k')
    plt.plot(x0_true+scaleguide*.05*(p+1)*np.cos(AoA[p]*t+phi0_true),y0_true+scaleguide*.05*(p+1)*np.sin(AoA[p]*t+phi0_true),'k')
    plt.plot([0,x_bisec[p],x0_bisec],[0,y_bisec[p],y0_bisec],':m')
#    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'m')
    plt.plot(x0_bisec+scaleguide*.05*(p+1)*np.cos(AoA[p]*t+phi0_bisec),y0_bisec+scaleguide*.05*(p+1)*np.sin(AoA[p]*t+phi0_bisec),'m')

plt.title("All estimations of position for the full set of multipaths, after phi0 is estimated with bisec")

loc.RootMethod='lm'
(phi0_root,x0_root,y0_root,_,x_root,y_root,_)= loc.computeAllLocationsFromPaths(AoD,AoA,dels,AoA0_method='fsolve', group_method='drop1')
print(np.mod(phi0_root,np.pi*2),phi0_true)
    
    
plt.figure(4)
plt.plot(0,0,'sb')
plt.plot(x0_true,y0_true,'^g')
plt.plot([0,x0_true],[0,y0_true],':g')
plt.plot(x0_root,y0_root,'^c')
plt.plot([0,x0_root],[0,y0_root],':c')
plt.plot(x_true,y_true,'or')
plt.plot(x_root,y_root,'oy')
scaleguide=np.max(np.abs(np.concatenate([y_true,y0_true,x_true,x0_true],0)))
plt.plot([x0_true,x0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.cos(phi0_true)],[y0_true,y0_true+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.sin(phi0_true)],'k')
plt.plot([x0_root,x0_root+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.cos(phi0_root)],[y0_root,y0_root+1.2*scaleguide*.05*np.shape(theta_true)[0]*np.sin(phi0_root)],'m')
for p in range(np.shape(theta_true)[0]):
    plt.plot([0,x_true[p],x0_true],[0,y_true[p],y0_true],':k')
    t=np.linspace(0,1,21)
    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'k')
    plt.plot(x0_true+scaleguide*.05*(p+1)*np.cos(AoA[p]*t+phi0_true),y0_true+scaleguide*.05*(p+1)*np.sin(AoA[p]*t+phi0_true),'k')
    plt.plot([0,x_root[p],x0_root],[0,y_root[p],y0_root],':m')
#    plt.plot(0+scaleguide*.05*(p+1)*np.cos(AoD[p]*t),0+scaleguide*.05*(p+1)*np.sin(AoD[p]*t),'m')
    plt.plot(x0_root+scaleguide*.05*(p+1)*np.cos(AoA[p]*t+phi0_root),y0_root+scaleguide*.05*(p+1)*np.sin(AoA[p]*t+phi0_root),'m')

plt.title("All estimations of position for the full set of multipaths, after phi0 is estimated with root method")
plt.savefig('../Figures/locfromAoAAoD.png')

error_bisec=np.sqrt(np.abs(x0_true-x0_bisec)**2+np.abs(y0_true-y0_bisec))
error_root=np.sqrt(np.abs(x0_true-x0_root)**2+np.abs(y0_true-y0_root))

print(error_bisec,error_root)


plt.figure(5)
plt.plot(X0e,Y0e,':', label='_nolegend_')
plt.plot(0,0,'sb',markersize=10)
plt.plot(x0_true,y0_true,'^g',markersize=10)
plt.axis([-50,50,-50,50])
plt.xlabel('$d_{ox}$ (m)')
plt.ylabel('$d_{oy}$ (m)')
plt.legend(['Transmitter','Receiver'])
plt.savefig('../Figures/graphsoldrop1%d.eps'%(Npath))

plt.figure(6)
ax = plt.axes(projection='3d')
for gr in range(Npath):
    ax.plot3D(X0e[:,gr],Y0e[:,gr],tau0_true*c+c*TauEe[:,gr], ':', label='_nolegend_')
ax.plot3D([0],[0],[0],'sb',markersize=10)
ax.plot3D(x0_true,y0_true,tau0_true*c*np.ones_like(y0_true),'^g',markersize=10)
ax.set_xlim(-50,50)
ax.set_ylim(-50,50)
ax.set_zlim(0,np.max(tau0_true*c+c*TauEe[TauEe>0]))
ax.set_xlabel('$d_{ox}$ (m)')
ax.set_ylabel('$d_{oy}$ (m)')
ax.set_zlabel('$\\ell_e$ (m)')
plt.legend(['Transmitter','Receiver'])
plt.savefig('../Figures/graph3Dsoldrop1%d.eps'%(Npath))



plt.figure(7)
ax = plt.axes(projection='3d')
for gr in range(Npath-2):
    ax.plot3D(x0all[:,gr],y0all[:,gr],tau0_true*c+c*tauEall[:,gr], ':', label='_nolegend_')
ax.plot3D([0],[0],[0],'sb',markersize=10)
ax.plot3D(x0_true,y0_true,tau0_true*c*np.ones_like(y0_true),'^g',markersize=10)
ax.set_xlim(-50,50)
ax.set_ylim(-50,50)
ax.set_zlim(0,np.max(tau0_true*c+3e8*TauEe[TauEe>0]))
ax.set_xlabel('$d_{ox}$ (m)')
ax.set_ylabel('$d_{oy}$ (m)')
ax.set_zlabel('$\\ell_e$ (m)')
plt.legend(['Transmitter','Receiver'])
plt.savefig('../Figures/graph3Dsol%d.eps'%(Npath))


#experimental code to find the correct phi_true as a function of dels , theta_true, x0 and y0; used in mmwave sim to modify channel model
#theta_dif=theta_true-theta0_true
#l0_true=np.sqrt(x0_true**2+y0_true**2)
#l_true=l0_true+(tau_true-tau0_true)*c
#C=l_true/np.tan(theta_dif)-l0_true/np.sin(theta_dif)
#L=l_true**2
#Z=l0_true**2
#S=(C*l0_true+np.sqrt( Z*(C**2) - (C**2+L)*(Z-L) ))/(C**2+L)
#S2=(C*l0_true-np.sqrt( Z*(C**2) - (C**2+L)*(Z-L) ))/(C**2+L)
#phi_dif=np.zeros((4,theta_true.size))
#phi_dif[0,:]=np.arcsin(S)
#phi_dif[1,:]=np.arcsin(S2)
#phi_dif[2,:]=np.pi-np.arcsin(S)
#phi_dif[3,:]=np.pi-np.arcsin(S2)
#
#x=(y0_true+x0_true*np.tan(phi_dif-theta0_true))/(np.tan(theta_true)+np.tan(phi_dif-theta0_true))
#y=x*np.tan(theta_true)
#
#dist=np.sqrt(x**2+y**2)+np.sqrt((x-x0_true)**2+(y-y0_true)**2)
#
#phi_dif_final=phi_dif[np.argmin(np.abs(dist-l_true),0),range(l_true.size)]