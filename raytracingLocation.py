#
import matplotlib 
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
import pandas as pd
import collections as col

""" Clase relacionada con funcións de canle """

class raytracingLocation:

    def __init__(self,Npath) -> None:
        self.c = 3e8


            
    #Genera canal con resultados consistentes para raytracing
    #non sei si xa existe, o mrayloc e btt caotico
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

    def genChanRayTracing(self, Npath,Nsims,xmax,xmin,ymax,ymin):

        x = np.random.rand(Npath,Nsims)*(xmax-xmin)+xmin
        y = np.random.rand(Npath,Nsims)*(ymax-ymin)+ymin

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