#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import time
from tqdm import tqdm

import sys
sys.path.append('../')
from CASTRO5G import compressedSensingTools as cs
from CASTRO5G import multipathChannel as mc

plt.close('all')

Nchan=10
Nd=4
Na=4
Ncp=16
Nsym=5
Nrft=1
Nrfr=2
K=64
Ts=2.5
Ds=Ts*Ncp
SNRs=10**(np.arange(-1,2.01,1.0))

omprunner = cs.CSDictionaryRunner()
dicMult=cs.CSMultiDictionary()

# Lista de pilotos a comparar
pilots_config = [
    ("IDUV", "IDUV"),
    ("BPSK", "BPSK")  # Usamos QPSK que es similar a BPSK pero con fase fija
]

Npilots = len(pilots_config)
Nsnr = len(SNRs)
MSE = np.zeros((Nchan, Nsnr, Npilots))
prepYTime = np.zeros((Nchan, Npilots))
prepHTime = np.zeros((Npilots))
sizeYDic = np.zeros((Npilots))
sizeHDic = np.zeros((Npilots))
runTime = np.zeros((Nchan, Nsnr, Npilots))
Npaths = np.zeros((Nchan, Nsnr, Npilots))
pathResults = {}

# Configuración fija del algoritmo (usamos dicMult con parámetros OMPx1m)
Xt, Xa, Xd, Xr = 1.0, 1.0, 1.0, 1.0
algName = "OMPx1m"

#-------------------------------------------------------------------------------
# Preconfiguración del diccionario H (independiente de pilotos)
t0 = time.time()
Lt, La, Ld = (int(Ncp*Xt), int(Na*Xa), int(Nd*Xd))
dicMult.setHDic((K, Ncp, Na, Nd), (Lt, La, Ld))
if isinstance(dicMult.currHDic.mPhiH, np.ndarray):
    sizeHDic[:] = dicMult.currHDic.mPhiH.size
elif isinstance(dicMult.currHDic.mPhiH, tuple):
    sizeHDic[:] = np.sum([x.size for x in dicMult.currHDic.mPhiH])
prepHTime[:] = time.time() - t0

#-------------------------------------------------------------------------------
# Preparamos los canales (sin pilotos específicos aún)
listPreparedChannels = []
for ichan in range(Nchan):    
    model = mc.DiscreteMultipathChannelModel(dims=(Ncp, Na, Nd), fftaxes=())
    hsparse = model.getDEC(10)
    itdoa, iaoa, iaod = np.where(hsparse != 0)
    pathsparse = pd.DataFrame({        
        "coefs": hsparse[itdoa, iaoa, iaod],
        "TDoA": itdoa,
        "AoA": np.arcsin(2*iaoa/Na - 2*(iaoa >= Na/2)),
        "AoD": np.arcsin(2*iaod/Nd - 2*(iaod >= Nd/2))
    })
    hk = np.fft.fft(hsparse, K, axis=0)
    hk = np.fft.fft(hk, Na, axis=1, norm="ortho")
    hk = np.fft.fft(hk, Nd, axis=2, norm="ortho")
    zp = mc.AWGN((Nsym, K, Na, 1))
    listPreparedChannels.append((pathsparse, hsparse, hk, zp))

#-------------------------------------------------------------------------------
for ipilot, (pilot_name, pilot_alg) in enumerate(tqdm(pilots_config, desc="Pilot Types: ")):
    pilgen = mc.MIMOPilotChannel(pilot_alg)
    
    for ichan in tqdm(range(Nchan), desc=f"Channels for {pilot_name}: ", leave=False):
        (pathsparse, hsparse, hk, zp) = listPreparedChannels[ichan]
        
        # Generamos pilotos específicos para este tipo
        (wp, vp) = pilgen.generatePilots(Nsym*K*Nrft, Na, Nd, Npr=Nsym*K*Nrfr, 
                                        rShape=(Nsym, K, Nrfr, Na), tShape=(Nsym, K, Nd, Nrft))
        
        # Preconfiguración del diccionario Y (dependiente de pilotos)
        t0 = time.time()
        dicMult.setHDic((K, Ncp, Na, Nd), (Lt, La, Ld))  # Aseguramos cache
        dicMult.setYDic(ichan, (wp, vp))
        if isinstance(dicMult.currYDic.mPhiY, np.ndarray):
            sizeYDic[ipilot] = dicMult.currYDic.mPhiY.size
        elif isinstance(dicMult.currYDic.mPhiY, tuple):
            sizeYDic[ipilot] = np.sum([x.size for x in dicMult.currYDic.mPhiY])
        prepYTime[ichan, ipilot] = time.time() - t0
        
        zp_bb = np.matmul(wp, zp)
        yp_noiseless = pilgen.applyPilotChannel(hk, wp, vp, None)
        zh = mc.AWGN((Ncp, Na, Nd))
        
        for isnr in range(Nsnr):
            sigma2 = 1.0 / SNRs[isnr]
            yp = yp_noiseless + zp_bb * np.sqrt(sigma2)
            hnoised = hsparse * np.sqrt(Ncp) + zh * np.sqrt(sigma2)
            
            t0 = time.time()
            omprunner.setDictionary(dicMult)
            hest, paths, _, _ = omprunner.OMP(yp, sigma2*K*Nsym*Nrfr, ichan, vp, wp, Xt, Xa, Xd, Xr, Ncp)
            runTime[ichan, isnr, ipilot] = time.time() - t0
            
            pathResults[(ichan, isnr, ipilot)] = (hest, paths)
            Npaths[ichan, isnr, ipilot] = len(paths.TDoA) if paths is not None else 0
            MSE[ichan, isnr, ipilot] = np.mean(np.abs(hk - hest)**2) / np.mean(np.abs(hk)**2)
        
        # Liberar memoria del diccionario Y para este canal
        dicMult.freeCacheOfPilot(ichan, (Ncp, Na, Nd), (Xt*Ncp, Xa*Na, Xd*Nd))

# Gráficos (adaptados para comparar pilotos en lugar de algoritmos)
outputFileTag = f'{Nsym}-{K}-{Ncp}-{Nrfr}-{Na}-{Nd}-{Nrfr}'
bytesPerFloat = np.array([0], dtype=np.complex128).itemsize
pilotLegendList = [x[0] for x in pilots_config]

plt.figure()
plt.yscale("log")
plt.semilogy(10*np.log10(SNRs), np.mean(MSE[:, :, 0], axis=0), '-o', label=pilotLegendList[0])
plt.semilogy(10*np.log10(SNRs), np.mean(MSE[:, :, 1], axis=0), '-s', label=pilotLegendList[1])
plt.legend()
plt.xlabel('SNR(dB)')
plt.ylabel('MSE')
plt.title(f'MSE vs SNR for algorithm {algName}')
plt.grid(True)

plt.figure()
plt.yscale("log")
barwidth = 0.9/Nsnr * (np.mean(np.diff(10*np.log10(SNRs))) if len(SNRs) > 1 else 1)
for ipilot in range(Npilots):
    offset = (ipilot - (Npilots-1)/2) * barwidth
    plt.bar(10*np.log10(SNRs) + offset, np.mean(runTime[:, :, ipilot], axis=0), 
            width=barwidth, label=pilotLegendList[ipilot])
plt.xlabel('SNR(dB)')
plt.ylabel('runtime')
plt.legend()
plt.title(f'Runtime vs SNR for algorithm {algName}')
plt.grid(True)

plt.figure()
barwidth = 0.9/Nsnr * np.mean(np.diff(10*np.log10(SNRs)))
for ipilot in range(Npilots):
    offset = (ipilot - (Npilots-1)/2) * barwidth
    plt.bar(10*np.log10(SNRs) + offset, np.mean(Npaths[:, :, ipilot], axis=0), 
            width=barwidth, label=pilotLegendList[ipilot])
plt.xlabel('SNR(dB)')
plt.ylabel('N paths')
plt.legend()
plt.title(f'Number of paths vs SNR for algorithm {algName}')
plt.grid(True)

plt.show()