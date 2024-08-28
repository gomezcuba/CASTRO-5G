#!/usr/bin/python

import numpy as np
import scipy as sp
import scipy.linalg 

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

plt.close("all")

N=4
M=4

#FIR channel with time-varying phase in the taps
h0=np.zeros((N,1),dtype=np.complex64)
h0[2]=1j*np.sqrt(.4)
h0[0]=np.sqrt(.6)
dop=np.zeros((N,1))
dop[2]=3/M/N# nu/T

def hDec(h0,dop,t):
    h=h0*np.exp(-2j*np.pi*dop*t)
    return(h)

plt.figure(1)
for m in range(M):
    plt.plot(np.abs(np.fft.fft(hDec(h0,dop,4*m),4,0))**2)
plt.show()

x=np.round(np.random.rand(N*M,1))*2-1 + 1j*(np.round(np.random.rand(N*M,1))*2-1)
xcp=np.concatenate((x[-N+1:],x))
yt=np.zeros((N*M+2*(N-1),1),dtype=np.complex64)
ytinv=np.zeros((N*M+2*(N-1),1),dtype=np.complex64)
for ind in range(N*M+N-1):
    ytinv[ind:ind+N]=ytinv[ind:ind+N]+xcp[ind]*hDec(h0,dop,0)
    yt[ind:ind+N]=yt[ind:ind+N]+xcp[ind]*hDec(h0,dop,np.arange(ind-N+1,ind+1).reshape((4,1)))
yinv=ytinv[N-1:-N+1]
y=yt[N-1:-N+1]
print(f'Time invariant filters are equal: {np.all(np.isclose(yinv,np.fft.ifft(np.fft.fft(x,N*M,0)*np.fft.fft(h0,N*M,0),N*M,0)))}')

h2D=np.zeros((N,M),dtype=np.complex64)
for d in range(N):
    h2D[d,int(dop[d]*N*M)]=h0[d]

hD=np.fft.fft(h2D,M*N,1)
xD=sp.linalg.circulant(x).T[0:N,:]

yD=np.sum(xD*hD,0)

print("Time-variant time-time operations are equal: %s"%np.all(np.isclose(y.T,yD)))

hF=np.fft.fft(hD,N,0)
yF=(np.fft.fft(np.fft.ifft(xD,M,0)/N*hF,M,0)*M)[0,:]#columns of xD are time-shifted and reflected vs x, for this reason in this line we convert fft(x) into ifft(xD[]) and ifft(y) into fft(yD)
print("Time-variant time-frequency filters are equal: %s"%np.all(np.isclose(yF,yD)))
plt.plot(np.abs(hF)**2,'o')

#x2D=np.fft.fft(x.reshape((4,4)).T,M,1)/np.sqrt(M)
x2D=np.fft.ifft(x.reshape(4,4).T,M,1)
#x2D=np.fft.ifft(np.fft.fft(x,M*N,0).reshape(M,N).T,N,0)
def conv2Dcustom(X,H):
    (R,C)=np.shape(H)
    val=np.zeros(np.shape(X),dtype=np.complex64)
    for row in range(R):
        for col in range(C):
            
#            shift=np.exp(-2j*np.pi*(row*col)/R/C)
            shift2=np.exp(-2j*np.pi*np.tile((np.arange(R).reshape(R,1))/R,(1,C))*col/C)
#            shift3=np.exp(-2j*np.pi*np.tile((np.arange(C).reshape(1,C))/C,(R,1))*row/R)
#            if (H[row,col]!=0):
#                print(np.angle(shift)/2/np.pi*16)
#                print(np.angle(shift2)/2/np.pi*16)
            quasiperiod=np.ones((4,4),dtype=np.complex64)
            quasiperiod[0:row,:]=np.exp(2j*np.pi*(np.arange(C))/C)*quasiperiod[0:row,:]
            val+= shift2*H[row,col]*np.roll(quasiperiod*np.roll(X,row,0),col,1)
    return(val)
    
def zak(x,K):
    N = len(x)
    if N <= K: return x
    even = zak(x[0::2],K)
    odd =  zak(x[1::2],K)
    T= [np.exp(-2j*np.pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + \
           [even[k] - T[k] for k in range(N//2)]
def izak(x,K):
    N = len(x)
    if N <= K: return x
    even = zak(x[0::2],K)
    odd =  zak(x[1::2],K)
    T= [np.exp(2j*np.pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + \
           [even[k] - T[k] for k in range(N//2)]

y2D=conv2Dcustom(x2D,h2D)
y2DtoT=np.fft.fft(y2D,M,1).T.reshape(1,16)
print("Time-variant delay-doppler filters are equal: %s"%np.all(np.isclose(y2DtoT,yD)))

# for i in range(16):
#     aux=np.zeros((16,1))
#     aux[i,0]=1
#     aux2D=np.fft.ifft(aux.reshape(4,4).T,M,1)
#     yaux_expected=np.sum(sp.linalg.circulant(aux.T).T[0:N,:]*hD,0)
#     yaux=np.fft.fft(conv2Dcustom(aux2D,h2D),M,1).T.reshape(1,16)
#     print("Aux probes %d are equal: %s"%(i,np.all(np.isclose(yaux,yaux_expected))))
    
# aux=np.zeros((16,1))
# aux[6]=1
# #aux[6]=1
# aux2D=np.fft.ifft(aux.reshape(4,4).T,M,1)
# #aux2D=np.concatenate(zak(aux,M)).reshape(4,4).T
# #aux2D=np.fft.ifft(np.fft.ifft(aux,M*N,0).reshape(M,N).T,M,1)
# #aux2Db=np.fft.ifft(np.fft.fft(aux,16,0).reshape(4,4),M,0)
# print(aux2D)
# #expected=np.fft.fft(np.sum(sp.linalg.circulant(aux.T).T[0:N,:]*hD,0).reshape(4,4).T,M,1)
# #print(expected)
# #print(conv2Dcustom(aux2D,h2D))
# yaux_expected=np.sum(sp.linalg.circulant(aux.T).T[0:N,:]*hD,0)
# yaux=np.fft.fft(conv2Dcustom(aux2D,h2D),M,1).T.reshape(1,16)
# print(yaux_expected)
# print(yaux)
# print("Aux probes are equal: %s"%np.all(np.isclose(yaux,yaux_expected)))

# yauxZ_expected=np.fft.ifft(np.sum(sp.linalg.circulant(aux.T).T[0:N,:]*hD,0).reshape(4,4).T,M,1)
# yauxZ=conv2Dcustom(aux2D,h2D)
# print(yauxZ_expected)
# print(yauxZ)
# #print(np.fft.ifft(np.fft.fft(conv2Dcustom(aux2D,h2D),M,1).T.reshape(16,1),16,0))
# #print(np.fft.ifft(np.fft.fft(conv2Dcustom(aux2Db,h2D),M,0).reshape(1,16),16,1))