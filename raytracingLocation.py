#
import matplotlib 
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
import pandas as pd
import collections as col
import threeGPPMultipathGenerator as mpg

""" Nesta API listaranse as funcións de consistencia necesarias para a correción de cada un dos escenarios nos que traballamos:

    1. Datos AoD e delay correctos -> Xeramos novos AoA consistentes a partir destos - genAoAConsistency
     
    2. Consistencia total escenario anterior -> desbotamos os AoD que entran polo backlobe - removeBacklobeAoD

        2.1. Implementar as dúas funcións anteriores nunha soa

    3. Datos AoD incorrectos -> Xeramos novos AoD a partir de AoA e delay - genAoDConsistency

    4. Datos delay incorrectos -> Xeramos novos tau (delay) a partir de AoA e AoD - genDelayConsistency

    ------- Funcións extra --------

    5. Adaptar según 1,3 ou 4 aleatoriamente para cada path - genRandomConsistency

    6. Usar array invertido l.142-155 simraygeommwave.py 

    
"""

class raytracingLocation:

    def __init__(self,Npath) -> None:
        self.c = 3e8


            
    #Genera canal random a partir de datos arbitrarios
    def genGeoChannel(self, Npath,Nsims,Xmax,Xmin,Ymax,Ymin):

        #generate locations and compute multipath 
        x=np.random.rand(Npath,Nsims)*(Xmax-Xmin)+Xmin
        y=np.random.rand(Npath,Nsims)*(Ymax-Ymin)+Ymin
        x0=np.random.rand(1,Nsims)*(Xmax-Xmin)+Xmin
        y0=np.random.rand(1,Nsims)*(Ymax-Ymin)+Ymin
        #angles from locations
        theta0=np.mod( np.arctan(y0/x0)+np.pi*(x0<0) , 2*np.pi)
        theta=np.mod( np.arctan(y/x)+np.pi*(x<0) , 2*np.pi)
        phi0=np.random.rand(1,Nsims)*2*np.pi #receiver angular measurement offset
        phi=np.mod(np.pi - (np.arctan((y-y0)/(x0-x))+np.pi*((x0-x)<0)) , 2*np.pi)
        #delays based on distance
        tau=(np.abs(y/np.sin(theta))+np.abs((y-y0)/np.sin(phi)))/self.c
        tau0=y0/np.sin(theta0)/self.c
        tauE=tau0+np.random.randn(1,Nsims)*40e-9

        geochannel = np.array([x,y,theta,phi,tau])

        return geochannel
    def channelConsistencyRetAOA(Nsims,xmax,ymax,xmin,ymin):

        chan = mpg.ThreeGPPMultipathChannelModel()
        small, macro = mpg.create_channel()


        return adaptedChannel

    def genChanRayTracing(x0,y0,x,y):

        model = mpg.ThreeGPPMultipathChannelModel()
        model.bLargeBandwidthOption=False
        macro,small = model.create_channel((x0,y0),(40,0,1.5))
        clusters,subpaths = small
        nClusters,tau,powC,AOA,AOD,ZOA,ZOD = clusters
        tau_sp,powC_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp = subpaths
        """ Comprobar consistencia de los retardos del canal
        
        1. Data extraction - Conocemos os AoA e os retardos 
        
        tau_vector = iago.data()
        AoA_vector = iago.data()

        2. Inventámonos unha phi consistente para cada un dos pares (desechamos a generada no modelo de iago)

        phi_vector = np.arctan(algo)

        3. A partir do LOS e tau_0, imos desechando aquelas mostras que non manteñen consistencia, e 
        generamos pares y_gorrito, x_gorrito de localizacións posibles para esos datos 
        
        """

        

        # En construcción
        threeGPPchannel = True


        return threeGPPchannel