import numpy as np
import collections as col
import multipathChannel as ch
import pandas as pd

class ThreeGPPMultipathChannelModel:
    ThreeGPPMacroParams = col.namedtuple( "ThreeGPPMacroParams",[
        'los',
        'PLdB',
        'ds',
        'asa',
        'asd',
        'zsa',
        'zsd',
        'K',
        'sf',
        #'zsd_MU',
        #'zsd_sg',
        'zod_offset_mu'
    ])
    ThreeGPPScenarioParams = col.namedtuple( "ThreeGPPScenarioParams",[
        'ds_mu',
        'ds_sg',
        'asd_mu',
        'asd_sg',
        'asa_mu',
        'asa_sg',
        'zsa_mu',
        'zsa_sg',
        #'zsd_mu',
        #'zsd_sg',
        'sf_sg',
        'K_mu',
        'K_sg',
        'rt',
        'N',
        'M',
        'cds',
        'casd',
        'casa',
        'czsa',
        'xi',
        'Cc', #Correlations Matrix Azimut [sSF, sK, sDS, sASD, sASA, sZSD, sZSA]
        'pLossFun',
        'zsdFun',
        'zodFun',
        'xpr_mu',
        'xpr_sg'
    ])
    #Crear diccionarios
    CphiNLOS = {4 : 0.779, 5 : 0.860 , 8 : 1.018, 10 : 1.090, 
                11 : 1.123, 12 : 1.146, 14 : 1.190, 15 : 1.211, 16 : 1.226, 
                19 : 1.273, 20 : 1.289, 25 : 1.358}
    CtetaNLOS = {8 : 0.889, 10 : 0.957, 11 : 1.031, 12 : 1.104, 
                 15 : 1.1088, 19 : 1.184, 20 : 1.178, 25 : 1.282}
    alpham = {0 : 0.0447, 1 : 0.0447, 2 : 0.1413, 3 : 0.1413, 4 : 0.2492,
              5 : 0.2492, 6 :  0.3715, 7 :  0.3715, 8 : 0.5129, 9 : 0.5129,
              10 : 0.6797, 11 : 0.6797, 12 :  0.8844, 13 :  0.8844, 14 : 1.1481,
              15 : 1.1481, 16 : 1.5195, 17 : 1.5195, 18 : 2.1551, 19 : 2.1551}
    
    #RMa hasta 7GHz y el resto hasta 100GHz
    def __init__(self, fc = 28, sce = "UMi", corrDistance = 15.0):
        self.frecRefGHz = fc*1e9
        self.scenario = sce
        self.corrDistance = corrDistance
        self.dMacrosGenerated = {}
        self.dChansGenerated = {}
        self.bLargeBandwidthOption = False 
        self.clight=3e8
        self.wavelength = 3e8/fc
        if sce.find("UMi")>=0:
            #LOS Probability (distance is in meters)
            self.scenarioLosProb = lambda d2D : 1 if d2D<18 else 18.0/d2D + np.exp(-d2D/36.0)*(1-18.0/d2D)
            self.scenarioParamsLOS = self.ThreeGPPScenarioParams(
                -0.24*np.log10(1+fc)-7.14,
                0.38,
                -0.05*np.log10(1+fc)+1.21,
                0.41,
                -0.08*np.log10(1+fc)+1.73,
                0.014*np.log10(1+fc)+0.28,
                -0.1*np.log10(1+fc)+0.73,
                -0.04*np.log10(1+fc)+0.34,
                4,
                9,
                5,
                3,
                12,
                20,
                5,
                3,
                17,
                7,
                3,
                np.array([[1,0.5,-0.4,-0.5,-0.4,0,0],
                   [0.5,1,-0.7,-0.2,-0.3,0,0],
                   [-0.4,-0.7,1,0.5,0.8,0,0.2],
                   [-0.5,-0.2,0.5,1,0.4,0.5,0.3],
                   [-0.4,-0.3,0.8,0.4,1,0,0],
                   [0,0,0,0.5,0,1,0],
                   [0,0,0.2,0.3,0,0,1]]),
                self.scenarioPlossUMiLOS,
                self.ZSDUMiLOS,
                0,
                9,
                3,
            )
            self.scenarioParamsNLOS = self.ThreeGPPScenarioParams(
                -0.24*np.log10(1+fc)-6.83,
                0.16*np.log10(1+fc)+0.28,
                -0.23*np.log10(1+fc)+1.53,
                0.11*np.log10(1+fc)+0.33,
                -0.08*np.log10(1+fc)+1.81,
                0.05*np.log10(1+fc)+0.3,
                -0.04*np.log10(1+fc)+0.92,
                -0.07*np.log10(1+fc)+0.41,
                7.82,
                0,
                0,
                2.1,
                19,
                20,
                11,
                10,
                22,
                7,
                3,
                np.array([[1,0,-0.7,0,-0.4,0,0],
                   [0,1,0,0,0,0,0],
                   [-0.7,0,1,0,0.4,0,-0.5],
                   [0,0,0,1,0,0.5,0.5],
                   [-0.4,0,0.4,0,1,0,0.2],
                   [0,0,0,0.5,0,1,0],
                   [0,0,-0.5,0.5,0.2,0,1]]),
                self.scenarioPlossUMiNLOS,
                self.ZSDUMiNLOS,
                self.ZODUMiNLOS,
                8,
                3,
            )
        elif sce.find("UMa")>=0:
            self.scenarioLosProb = lambda d2D,hut : 1 if d2D<18 else (18.0/d2D + np.exp(-d2D/63.0)*(1-18.0/d2D))*(1 + (0 if hut<=23 else ((hut-13.0)/10.0)**1.5)*1.25*np.exp(d2D/100.0)*(-d2D/150.0)**3.0) 
            self.scenarioParamsLOS = self.ThreeGPPScenarioParams(
                -6.955 - 0.0963*np.log10(fc),
                0.66,
                1.06 + 0.1114*np.log10(fc),
                0.28, 
                1.81,
                0.20,
                0.95,
                0.16, 
                4, 
                9, 
                3.5, 
                2.5,
                12,
                20,
                np.maximum(0.25,6.5622 - 3.4084*np.log10(fc)),
                5,
                11,
                7,
                3,
                np.array([[1,0,-0.4,-0.5,-0.5,0,-0.8],
                   [0,1,-0.4,0,-0.2,0,0],
                   [-0.4,-0.4,1,0.4,0.8,-0.2,0],
                   [-0.5,0,0.4,1,0,0.5,0],
                   [-0.5,-0.2,0.8,0,1,-0.3,0.4],
                   [0,0,-0.2,0.5,-0.3,1,0],
                   [-0.8,0,0,0,0.4,0,1]]),
                self.scenarioPlossUMaLOS,
                self.ZSDUMaLOS,
                0,
                8,
                4,
            )
            self.scenarioParamsNLOS = self.ThreeGPPScenarioParams(
                -6.28 - 0.204*np.log10(fc),
                0.39,
                1.5 - 0.1144*np.log10(fc),
                0.28,
                2.08 - 0.27*np.log10(fc),
                0.11,
                -0.3236*np.log10(fc) + 1.512,
                0.16,
                6,
                0,
                0,
                2.3,
                20,
                20,
                np.maximum(0.25, 6.5622 - 3.4084*np.log10(fc)),
                2,
                15,
                7,
                3,
                np.array([[1,0,-0.4,-0.6,0,0,-0.4],
                   [0,1,0,0,0,0,0],
                   [-0.4,0,1,0.4,0.6,-0.5,0],
                   [-0.6,0,0.4,1,0.4,0.5,-0.1],
                   [0,0,0.6,0.4,1,0,0],
                   [0,0,-0.5,0.5,0,1,0],
                   [-0.4,0,0,-0.1,0,0,1]]),
                self.scenarioPlossUMaNLOS,
                self.ZSDUMaNLOS,
                self.ZODUMaNLOS,
                7,
                3,
            )
        elif sce.find("RMa")>=0:
            self.scenarioLosProb = lambda d2D : 1 if d2D<10 else np.exp(-(d2D-10.0)/1000.0)
            self.scenarioParamsLOS = self.ThreeGPPScenarioParams(
                -7.49,
                0.55,
                0.90,
                0.38,
                1.52,
                0.24,
                0.47,
                0.40,
                4, 
                7,
                4,
                3.8,
                11,
                20,
                0,
                2,
                3,
                3,
                3,
                np.array([[1,0,-0.5,0,0,0.1,-0.17],
                   [0,1,0,0,0,0,-0.02],
                   [-0.5,0,1,0,0,-0.05,0.27],
                   [0,0,0,1,0,0.73,-0.14],
                   [0,0,0,0,1,-0.20,0.24],
                   [0.01,0,-0.05,0.73,-0.20,1,-0.07],
                   [-0.17,-0.02,0.27,-0.14,0.24,-0.07,1]]),
                self.scenarioPlossRMaNLOS,
                self.ZSDRMaLOS,
                0,
                12,
                4,
            )
            self.scenarioParamsNLOS = self.ThreeGPPScenarioParams(
                -7.43,
                0.48,
                0.95,
                0.45,
                1.52,
                0.13,
                0.58,
                0.37,
                8,
                0,
                0,
                1.7,
                10,
                20,
                0,
                2,
                3,
                3,
                3,
                np.array([[1,0,-0.5,0.6,0,-0.04,-0.25],
                   [0,1,0,0,0,0,0],
                   [-0.5,0,1,-0.4,0,-0.1,-0.4],
                   [0.6,0,-0.4,1,0,0.42,-0.27],
                   [0,0,0,0,1,-0.18,0.26],
                   [-0.04,0,-0.1,0.42,-0.18,1,-0.27],
                   [-0.25,0,-0.4,-0.27,0.26,-0.27,1]]),
                self.scenarioPlossRMaNLOS,
                self.ZSDRMaNLOS,
                self.ZODRMaNLOS,
                7,
                3,
            )
        elif sce.find("InH-Office-Mixed")>=0:
            self.scenarioLosProb = lambda d2D : 1 if d2D<1.2 else (np.exp(-(d2D-1.2)/4.7) if 1.2<d2D<6.5 else (np.exp(-(d2D-6.5)/32.6))*0.32)
            self.scenarioParamsLOS = self.ThreeGPPScenarioParams(
                -0.01*np.log10(1+fc) - 7.692,
                0.18,
                1.60,
                0.18,
                -0.19*np.log10(1+fc) + 1.781,
                0.12*np.log10(1+fc) + 0.119,
                -0.26*np.log10(1+fc) + 1.44,
                -0.04*np.log10(1+fc) + 0.264,
                3,
                7,
                4,
                3.6,
                15,
                20,
                0,
                5,
                8,
                9,
                6,
                np.array([[1,0.5,-0.8,-0.4,-0.5,0.2,0.3],
                   [0.5,1,-0.5,0,0,0,0.1],
                   [-0.8,-0.5,1,0.6,0.8,0.1,0.2],
                   [-0.4,0,0.6,1,0.4,0.5,0],
                   [-0.5,0,0.8,0.4,1,0,0.5],
                   [0.2,0,0.1,0.5,0,1,0],
                   [0.3,0.1,0.2,0,0.5,0,1]]),
               self.scenarioPlossInLOS,
               self.ZSDInLOS,
               0,
               11,
               4,
            )
            self.scenarioParamsNLOS = self.ThreeGPPScenarioParams(
                -0.28*np.log10(1+fc) - 7.173,
                0.10*np.log10(1+fc) + 0.055,
                1.62,
                0.25,
                -0.11*np.log10(1+fc) + 1.863,
                0.12*np.log10(1+fc) + 0.059,
                -0.15*np.log10(1+fc) + 1.387,
                -0.09*np.log10(1+fc) + 0.746,
                8.03,
                0,
                0,
                3,
                19,
                20,
                0,
                5,
                11,
                9,
                3,
                np.array([[1,0,-0.5,0,-0.4,0,0],
                   [0,1,0,0,0,0,0],
                   [-0.5,0,1,0.4,0,-0.27,-0.06],
                   [0,0,0.4,1,0,0.35,0.23],
                   [-0.4,0,0,0,1,-0.08,0.43],
                   [0,0,-0.27,0.35,-0.08,1,0.42],
                   [0,0,-0.06,0.23,0.43,0.42,1]]),
                self.scenarioPlossInNLOS,
                self.ZSDInNLOS,
                0,
                10,
                4,
            )
        elif sce.find("InH-Office-Open")>=0:
            self.scenarioLosProb = lambda d2D : 1 if d2D<=5 else (np.exp(-(d2D-5.0)/70.8) if 5<d2D<49 else (np.exp(-(d2D-49.0)/211.7))*0.54)
            self.scenarioParamsLOS = self.ThreeGPPScenarioParams(
                -0.01*np.log10(1+fc) - 7.692,
                0.18,
                1.60,
                0.18,
                -0.19*np.log10(1+fc) + 1.781,
                0.12*np.log10(1+fc) + 0.119,
                -0.26*np.log10(1+fc) + 1.44,
                -0.04*np.log10(1+fc) + 0.264,
                3,
                7,
                4,
                3.6,
                15,
                20,
                0,
                5,
                8,
                9,
                6,
                np.array([[1,0.5,-0.8,-0.4,-0.5,0.2,0.3],
                   [0.5,1,-0.5,0,0,0,0.1],
                   [-0.8,-0.5,1,0.6,0.8,0.1,0.2],
                   [-0.4,0,0.6,1,0.4,0.5,0],
                   [-0.5,0,0.8,0.4,1,0,0.5],
                   [0.2,0,0.1,0.5,0,1,0],
                   [0.3,0.1,0.2,0,0.5,0,1]]),
               self.scenarioPlossInLOS,
               self.ZSDInLOS,
               0,
               11,
               4,
            )
            self.scenarioParamsNLOS = self.ThreeGPPScenarioParams(
                -0.28*np.log10(1+fc) - 7.173,
                0.10*np.log10(1+fc) + 0.055,
                1.62,
                0.25,
                -0.11*np.log10(1+fc) + 1.863,
                0.12*np.log10(1+fc) + 0.059,
                -0.15*np.log10(1+fc) + 1.387,
                -0.09*np.log10(1+fc) + 0.746,
                8.03,
                0,
                0,
                3,
                19,
                20,
                0,
                5,
                11,
                9,
                3,
                np.array([[1,0,-0.5,0,-0.4,0,0],
                   [0,1,0,0,0,0,0],
                   [-0.5,0,1,0.4,0,-0.27,-0.06],
                   [0,0,0.4,1,0,0.35,0.23],
                   [-0.4,0,0,0,1,-0.08,0.43],
                   [0,0,-0.27,0.35,-0.08,1,0.42],
                   [0,0,-0.06,0.23,0.43,0.42,1]]),
                self.scenarioPlossInNLOS,
                self.ZSDInNLOS,
                0,
                10,
                4,
            )
            

    #UMi Path Loss Functions
    def scenarioPlossUMiLOS(self,d3D,d2D,hbs=10,hut=1.5,W=20,h=5):     
        """        if not indoor:
            n=1
        else:
            Nf = np.random.uniform(4,8)
            n=np.random.uniform(1,Nf)
        hut_aux = 3*(n-1) + hut"""
        prima_dBP = (4*(hbs-1)*(hut-1)*self.frecRefGHz) / self.clight
        if(d2D<prima_dBP):
            ploss = 32.4 + 21.0*np.log10(d3D)+20.0*np.log10(self.frecRefGHz)
        else:
            ploss = 32.4 + 40*np.log10(d3D)+20.0*np.log10(self.frecRefGHz)-9.5*np.log10(np.power(prima_dBP,2)+np.power(hbs-hut,2))
        return(ploss)
    def scenarioPlossUMiNLOS(self,d3D,d2D,hbs=10,hut=1.5,W=20,h=5):
        """if not indoor:
            n = 1
        else:
            Nf = np.random.uniform(4,8)
            n = np.random.uniform(1,Nf)
        hut_aux = 3*(n-1) + hut"""
        PL1 = 35.3*np.log10(d3D) + 22.4 + 21.3*np.log10(self.frecRefGHz)-0.3*(hut - 1.5)
        PL2 = self.scenarioPlossUMiLOS(d3D,d2D,hut) #PLUMi-LOS = Pathloss of UMi-Street Canyon LOS outdoor scenario
        ploss = np.maximum(PL1,PL2)
        return(ploss)
    
    
    #UMa Path Loss Functions
    def scenarioPlossUMaLOS(self,d3D,d2D,hbs=25,hut=1.5,W=20,h=5):
        """if not indoor:
            n = 1
        else:
            Nf = np.random.uniform(4,8)
            n = np.random.uniform(1,Nf)
        hut_aux = 3*(n-1) + hut"""
        prima_dBP = (4*(hbs-1)*(hut-1)*self.frecRefGHz) / self.clight
        if(d2D<prima_dBP):
            ploss = 28.0 + 22.0*np.log10(d3D)+20.0*np.log10(self.frecRefGHz)
        else:
            ploss = 28.0 + 40.0*np.log10(d3D)+20.0*np.log10(self.frecRefGHz)-9.0*np.log10(np.power(prima_dBP,2)+np.power(hbs-hut,2))
        return(ploss)
    def scenarioPlossUMaNLOS(self,d3D,d2D,hbs=25,hut=1.5,W=20,h=5):
        """if not indoor:
            n = 1
        else:
            Nf = np.random.uniform(4,8)
            n = np.random.uniform(1,Nf)
        hut_aux = 3*(n-1) + 1.5
        """
        PL1 = 13.54 + 39.08*np.log10(d3D) + 20.0*np.log10(self.frecRefGHz) - 0.6*(hut - 1.5)
        PL2 = self.scenarioPlossUMaLOS(d3D,d2D,hbs,hut)
        ploss = np.maximum(PL1,PL2)
        return(ploss)
    
    
    #RMa Path Loss Functions
    def scenarioPlossRMaLOS(self,d3D,d2D=5000,hbs=35,hut=1.5,W=20,h=5):
        dBp = (2*np.pi*hbs*hut)*(self.frecRefGHz*1e9/self.clight) #Break point distance
        if(d2D<dBp):
            ploss = 20*np.log10(40.0*np.pi*d3D*self.frecRefGHz/3.0)+np.minimum(0.03*np.power(h,1.72),10)*np.log10(d3D)-np.minimum(0.044*np.power(h,1.72),14.77)+0.002*np.log10(h)*d3D
        else:
            ploss = 20*np.log10(40.0*np.pi*dBp*self.frecRefGHz/3.0)+np.minimum(0.03*np.power(h,1.72),10)*np.log10(dBp)-np.minimum(0.044*np.power(h,1.72),14.77)+0.002*np.log10(h)*dBp + 40.0*np.log10(d3D/dBp)
        return(ploss)
    def scenarioPlossRMaNLOS(self,d3D,d2D=5000,hbs=25,hut=1.5,W=20,h=5):
        PL1= 161.04 - (7.1*np.log10(W)) + 7.5*(np.log10(h)) - (24.37 - 3.7*(np.power((h/hbs),2)))*np.log10(hbs) + (43.42 - 3.1*(np.log10(hbs)))*(np.log10(d3D)-3) + 20*np.log10(3.55) - (3.2*np.power(np.log10(11.75*hut),2)) - 4.97
        PL2= self.scenarioPlossRMaLOS(d3D,d2D)
        ploss = np.maximum(PL1,PL2)
        return(ploss)
    
    
    #Inh Path Loss Functions
    def scenarioPlossInLOS(self,d3D,d2D,hbs,hut,W=20,h=5):
        ploss= 32.4 + 17.3*np.log10(d3D) + 20.0*np.log10(self.frecRefGHz)
        return(ploss)
    def scenarioPlossInNLOS(self,d3D,d2D,hbs,hut,W=20,h=5):
        PL1= 38.3*np.log10(d3D) + 17.30 + 24.9*np.log10(self.frecRefGHz)
        PL2= self.scenarioPlossInLOS(d3D,d2D,hbs,hut)
        ploss= np.maximum(PL1,PL2)
        return(ploss)  
    
    
    def ZSDUMiLOS(self,d2D,hut=1.5):  
        zsd_mu = np.maximum(-0.21, -14.8*(d2D/1000.0) + 0.01*np.abs(hut-10.0) +0.83)
        zsd_sg = 0.35
        zsd = min( np.power(10.0,zsd_mu + zsd_sg * np.random.randn(1)), 52.0)
        return(zsd)
    def ZSDUMiNLOS(self,d2D,hut=1.5):     
        zsd_mu = np.maximum(-0.5, -3.1*(d2D/1000.0) + 0.01*np.maximum(hut-10.0,0) +0.2)
        zsd_sg = 0.35
        zsd = min( np.power(10.0,zsd_mu + zsd_sg * np.random.randn(1)), 52.0)
        return(zsd)
    def ZODUMiNLOS(self,d2D,hut=1.5):
        zod_offset_mu = -np.power(10.0,-1.5*np.log10(np.maximum(10,d2D)) + 3.3)
        return(zod_offset_mu)
    
    def ZSDUMaLOS(self,d2D,hut=1.5):
        zsd_mu = np.maximum(-0.5, -2.1*(d2D/1000.0) - 0.01*(hut - 1.5) + 0.75)
        zsd_sg = 0.40
        print(10.0**( zsd_mu + zsd_sg * np.random.randn(1)))
        zsd = min( np.power(10.0,zsd_mu + zsd_sg * np.random.randn(1)), 52.0)
        return(zsd) 
    def ZSDUMaNLOS(self,d2D,hut=1.5):     
        zsd_mu = np.maximum(-0.5, -2.1*(d2D/1000.0) - 0.01*(hut - 1.5) + 0.9)
        zsd_sg = 0.49
        zsd = min( np.power(10.0,zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
        return(zsd)
    def ZODUMaNLOS(self,d2D,hut=1.5):
        efc = 7.66*np.log10(self.frecRefGHz) - 5.96
        afc = 0.208*np.log10(self.frecRefGHz) - 0.782
        aux = afc*np.log10(np.maximum(25.0,d2D))
        cfc = - 0.13*np.log10(self.frecRefGHz) + 2.03
        zod_offset_mu = efc - np.power(10,aux)+ cfc - 0.07*(hut - 1.5)
        return(zod_offset_mu)
    
    def ZSDRMaLOS(self,d2D,hut=1.5):
        zsd_mu = np.maximum(-1, -0.17*(d2D/1000.0) - 0.01*(hut - 1.5) + 0.22)
        zsd_sg = 0.34
        zsd = min( np.power(10.0, zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
        return(zsd)
    def ZSDRMaNLOS(self,d2D,hut=1.5):     
        zsd_mu = np.maximum(-1, -0.19*(d2D/1000) - 0.01*(hut - 1.5) + 0.28)
        zsd_sg = 0.30
        zsd = min( np.power(10.0, zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
        return(zsd)
    def ZODRMaNLOS(self,d2D,hut=1.5):
        zod_offset_mu=  np.arctan((35.0 - 3.5)/d2D) - np.arctan((35.0 - 1.5)/d2D)
        return(zod_offset_mu)
    
    def ZSDInLOS(self,d2D,hut=1.5):
        zsd_mu = -1.43*np.log10(1 + self.frecRefGHz) + 2.228
        zsd_sg = 0.13*np.log10(1 + self.frecRefGHz) + 0.30
        zsd = min( np.power(10.0, zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
        return(zsd)
    def ZSDInNLOS(self,d2D,hut=1.5):     
        zsd_mu = 1.08
        zsd_sg = 0.36
        zsd = min( np.power(10.0,zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
        return(zsd)
    
   
    #Large Scale Parametres
    def create_macro(self, txPos, rxPos):
        aPos = np.array(txPos) 
        bPos = np.array(rxPos) 
        d2D = np.linalg.norm(bPos[0:-1]-aPos[0:-1])
        d3D = np.linalg.norm(bPos-aPos)
        hbs = aPos[2]
        hut = bPos[2]
        pLos=self.scenarioLosProb(d2D)
        los = (np.random.rand(1) <= pLos)
        vIndep = np.random.randn(7,1)
        if los:
            param = self.scenarioParamsLOS
        else:
            param = self.scenarioParamsNLOS
        vDep=param.Cc@vIndep
        ds = np.power(10.0, param.ds_mu + param.ds_sg * vDep[2])
        asa = min( np.power(10.0, param.asa_mu + param.asa_sg * vDep[4] ), 104.0)
        asd = min( np.power(10.0, param.asd_mu + param.asd_sg * vDep[3] ), 104.0)
        zsa = min( np.power(10.0, param.zsa_mu + param.zsa_sg * vDep[6] ), 52.0)
        zsd = param.zsdFun(d2D,hut)
        if los:
            zod_offset_mu = 0
        else: 
            zod_offset_mu = param.zodFun(d2D,hut)
        K= param.K_mu
        sf = param.sf_sg*vDep[0]
        PLdB = param.pLossFun(d3D,d2D,hbs,hut,W=20,h=5)
        if los:
            PL = PLdB + sf
        else:
            vDep = self.scenarioParamsLOS.Cc@vIndep
            sflos = self.scenarioParamsLOS.sf_sg*vDep[0]
            PL = np.maximum(PLdB + sf,PLdB + sflos)
        TgridXIndex= txPos[0] // self.corrDistance
        TgridYIndex= txPos[1] // self.corrDistance 
        RgridXIndex= rxPos[0] // self.corrDistance 
        RgridYIndex= rxPos[1] //self.corrDistance 
        key = (TgridXIndex,TgridYIndex,RgridXIndex,RgridYIndex)
        
        macro = self.ThreeGPPMacroParams(los,PL,ds,asa,asd,zsa,zsd,K,sf,zod_offset_mu)
        self.dMacrosGenerated[key]=macro
        return(macro)
    
    
    def create_clusters(self,macro,angles):    
        los = macro.los
        DS = macro.ds
        ASA = macro.asa
        ASD = macro.asd
        ZSA = macro.zsa
        ZSD = macro.zsd
        K = macro.K
        muZOD = macro.zod_offset_mu
    
        if los:
            param = self.scenarioParamsLOS
        else:
            param = self.scenarioParamsNLOS
        N = param.N
        rt = param.rt
        
        (losphiAoD,losphiAoA,losthetaAoD,losthetaAoA) = angles
        
        #Generate cluster delays
        tau_prima = -rt*DS*np.log(np.random.uniform(0,1,size=N))
        tau_prima = tau_prima-np.amin(tau_prima)
        tau = np.array(sorted(tau_prima))
        Ctau = 0.7705 - 0.0433*K + 0.0002*K**2 + 0.000017*K**3
        if los:
            tau = tau / Ctau 
        
        #Generate cluster powers
        xi = param.xi
        powPrima = np.exp(-tau*((rt-1)/(rt*DS)))*10**(-(xi*np.random.normal(0,1,size=N)/10))
        if los:
            p1LOS = K/(K+1)
            powC = (1/K+1)*(powPrima/np.sum(powPrima))
            powC[0] = powC[0] + p1LOS
        else:
            powC = powPrima/np.sum(powPrima)
        #Remove clusters with less than -25 dB power compared to the maximum cluster power. The scaling factors need not be 
        #changed after cluster elimination 
        maxP = np.amax(powC)
        #------------------------------------------------
        tau = tau[powC > (maxP*10**(-2.5))] #natural units
        powC = powC[powC > (maxP*10**(-2.5))]
        nClusters = np.size(powC)
        
        #Generate arrival angles and departure angles for both azimuth and elevation   
        #Azimut
        if los:
            Cphi = self.CphiNLOS.get(N)*(1.1035 - 0.028*K - 0.002*(K**2) + 0.0001*(K**3))
        else:
            Cphi = self.CphiNLOS.get(N)
        phiAOAprima = 2*(ASA/1.4)*np.sqrt(-np.log(powC/maxP))/Cphi
        phiAODprima = 2*(ASD/1.4)*np.sqrt(-np.log(powC/maxP))/Cphi
        
        X = np.random.uniform(-1,1,size=powC.shape)
        Y = np.random.normal(0,(ASA/7)**2,size=powC.shape)
        AOA = X*phiAOAprima + Y + losphiAoA - (X[0]*phiAOAprima[0] + Y[0] if los==1 else 0)
        AOD = X*phiAODprima + Y + losphiAoD - (X[0]*phiAODprima[0] + Y[0] if los==1 else 0)
        
        #Zenith 
        if los:
            Cteta = self.CtetaNLOS.get(N)*(1.3086 + 0.0339*K -0.0077*(K**2) + 0.0002*(K**3))
        else:
            Cteta = Cteta = self.CtetaNLOS.get(N)
        
        tetaZOAprima = -((ZSA*np.log(powC/maxP))/Cteta)
        tetaZODprima = -((ZSD*np.log(powC/maxP))/Cteta)
        
        Y1 = np.random.normal(0,(ZSA/7)**2,size=powC.shape)
        Y2 = np.random.normal(0,(ZSD/7)**2,size=powC.shape)   
        ZOA = X*tetaZOAprima + Y1 + losthetaAoA - (X[0]*tetaZOAprima[0] + Y1[0] if (los==1) else 0)
        ZOD = X*tetaZODprima + Y2 + losthetaAoD - (X[0]*tetaZODprima[0] + Y2[0]- muZOD if (los==0) else 0)
        
        return(nClusters,tau,powC,AOA,AOD,ZOA,ZOD)
   
    
    #Revisar los parametros de entrada
    def create_subpaths_largeBW(self,macro,clusters,maxM=20,Dh=2,Dv=2,B=2e6):
        (nClusters,tau,powC,AOA,AOD,ZOA,ZOD) = clusters
        
        los = macro.los
        casd = macro.casd
        casa = macro.casa
        czsa = macro.czsa
        ZSD = macro.zsd
        
        if los:
            param = self.scenarioParamsLOS
        else:
            param = self.scenarioParamsNLOS
        M = param.M
        cds = param.cds
        
        #The offset angles alpha_m
        alpha = np.random.uniform(-2,2,size=(nClusters,M))
        
        #The relative delay of m-th ray
        tau_primaprima = np.random.uniform(0,2*cds,size=(nClusters,M))
        tau_prima = tau_primaprima-np.amin(tau_primaprima)
    
        #Ray powers
        czsd = (3/8)*10**(self.zsd_mu)
        powPrima = np.exp(-tau_prima/cds)*np.exp(-(np.sqrt(2)*abs(alpha))/casa)*np.exp(-(np.sqrt(2)*abs(alpha))/casd)*np.exp(-(np.sqrt(2)*abs(alpha))/czsa)*np.exp(-(np.sqrt(2)*abs(alpha))/czsd)
        powC_sp = powC*(powPrima/np.sum(powPrima))
        
        #The number of rays per cluster
        k = 0.5
        m_t = np.ceil(4*k*cds*B)
        m_AOD = np.ceil(4*k*casd*((np.pi*Dh)/(180*self.wavelength)))
        m_ZOD = np.ceil(4*k*czsd*((np.pi*Dv)/(180*self.wavelength)))
        M = min(np.maximum(m_t*m_AOD*m_ZOD,20),maxM)

        #Angles generation 
        AOA_sp = np.zeros((nClusters,M))
        AOD_sp = np.zeros((nClusters,M))

        for i in range(nClusters):
            for j in range(M):
                AOA_sp[i,j] = AOA[i] + casa*alpha[i,j]
                AOD_sp[i,j] = AOD[i] + casa*alpha[i,j]
        
        ZOA_sp = np.zeros((nClusters,M))
        ZOD_sp = np.zeros((nClusters,M))
        for i in range(nClusters):
            for j in range(M):
                ZOA_sp[i,j] = ZOA[i] + czsa*alpha[i,j]
                ZOD_sp[i,j] = ZOD[i] + (3/8)*(10**ZSD)*alpha[i,j]
        
        return(tau_prima,powC_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp)
        
   
    def create_subpaths_basics(self,macro,clusters):
        los = macro.los
        ZSD = macro.zsd
    
        if los:
            param = self.scenarioParamsLOS
        else:
            param = self.scenarioParamsNLOS
        M = param.M
        cds = param.cds
        
        (nClusters,tau,powC,AOA,AOD,ZOA,ZOD)=clusters
        
        #Generate subpaths delays and powers
        powC_cluster = powC/M #Power of each cluster
        
        powC_sp = np.zeros((nClusters,M)) 
        tau_sp = np.zeros((nClusters,M)) 
        for i in range(nClusters):
            for j in range(M):
                powC_sp[i,j] = powC_cluster[i]
                tau_sp[i,j] = tau[i]
                
        row1 = np.array([0,0,0,0,0,0,0,0,1.28*cds,1.28*cds,1.28*cds,1.28*cds,2.56*cds,2.56*cds,2.56*cds,2.56*cds,1.28*cds,1.28*cds,0,0])
        tau_sp[0,:] = tau_sp[0,:] + row1*1e-9#ns
        tau_sp[1,:] = tau_sp[1,:] + row1*1e-9#ns
        
        """
        #Subclusters
        R1 = (1,2,3,4,5,6,7,8,19,20)
        R2 = (9,10,11,12,17,18)
        R3 = (13,14,15,16)
        
        P1 = len(R1)/M
        P2 = len(R2)/M
        P3 = len(R3)/M
        
        tau1 = 0
        tau2 = 1.28*cds
        tau3 = 2.56*cds
        """   
        casa = param.casa
        AOA_sp = np.zeros((nClusters,M))
        AOD_sp = np.zeros((nClusters,M))

        for i in range(nClusters):
            for j in range(M):
                AOA_sp[i,j] = AOA[i] + casa*self.alpham.get(j)*np.random.choice([1, -1])
                AOD_sp[i,j] = AOD[i] + casa*self.alpham.get(j)*np.random.choice([1, -1])
        
        czsa = param.czsa
        ZOA_sp = np.zeros((nClusters,M))
        ZOD_sp = np.zeros((nClusters,M))
        for i in range(nClusters):
            for j in range(M):
                ZOA_sp[i,j] = ZOA[i] + czsa*self.alpham.get(j)*np.random.choice([1, -1])
                ZOD_sp[i,j] = ZOD[i] + (3/8)*(10**ZSD)*self.alpham.get(j)*np.random.choice([1, -1])
        
        #mask = (ZOA_sp>=180) & (ZOA_sp<=360)
        #ZOA_sp[mask] = 360 - ZOA_sp
        
        return(tau_sp,powC_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp)
    
    
    def create_small_param(self,angles,macro):
        clusters = self.create_clusters(macro,angles)
        
        if self.bLargeBandwidthOption:
            subpaths = self.create_subpaths_largeBW(macro,clusters)
            (tau_sp,powC_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp) = subpaths
        else:
            subpaths = self.create_subpaths_basics(macro,clusters)
            (tau_sp,powC_sp,AOA_sp,AOD_sp,ZOA_sp,ZOD_sp) = subpaths
        
        los = macro.los
        if los:
            param = param = self.scenarioParamsLOS
        else:
            param = self.scenarioParamsNLOS
       
        for row in AOD_sp:
            np.random.shuffle(row)
        
        for row in ZOD_sp:
            np.random.shuffle(row)
        
        indicesSubcluster1 = [0,1,2,3,4,5,6,7,18,19]
        indicesSubcluster2 = [8,9,10,11,16,17]
        indicesSubcluster3 = [12,13,14,15]
        
        for i in range(2):
            AOD_sp[i][indicesSubcluster1] = AOD_sp[i][np.random.permutation(indicesSubcluster1)]
            AOD_sp[i][indicesSubcluster2] = AOD_sp[i][np.random.permutation(indicesSubcluster2)]
            AOD_sp[i][indicesSubcluster3] = AOD_sp[i][np.random.permutation(indicesSubcluster3)]
            
            ZOD_sp[i][indicesSubcluster1] = ZOD_sp[i][np.random.permutation(indicesSubcluster1)]
            ZOD_sp[i][indicesSubcluster2] = ZOD_sp[i][np.random.permutation(indicesSubcluster2)]
            ZOD_sp[i][indicesSubcluster3] = ZOD_sp[i][np.random.permutation(indicesSubcluster3)]
        
        # Generate the cross polarization power ratios
        xpr_mu = param.xpr_mu
        xpr_sg = param.xpr_sg
        X = np.random.normal(xpr_mu,xpr_sg,size=tau_sp.shape)
        kappa =  10**(X/10)
    
        return(clusters,subpaths)
    
        
    def create_channel(self, txPos, rxPos):
        aPos = np.array(txPos)
        bPos = np.array(rxPos)
        vLOS = bPos-aPos
        
        losphiAoD=np.mod( np.arctan( vLOS[1] / vLOS[0] )+np.pi*(vLOS[0]<0), 2*np.pi )
        losphiAoA=np.mod(np.pi+losphiAoD, 2*np.pi ) # revise
        vaux = (np.linalg.norm(vLOS[0:2]), vLOS[2] )
        losthetaAoD=np.pi/2-np.arctan( vaux[1] / vaux[0] )
        losthetaAoA=np.pi-losthetaAoD # revise
        
        #3GPP model is in degrees but numpy uses radians
        losphiAoD=(180.0/np.pi)*losphiAoD #angle of departure 
        losthetaAoD=(180.0/np.pi)*losthetaAoD 
        losphiAoA=(180.0/np.pi)*losphiAoA #angle of aperture
        losthetaAoA=(180.0/np.pi)*losthetaAoA
        angles = [losphiAoD,losphiAoA,losthetaAoD,losthetaAoA]
        TgridXIndex= txPos[0] // self.corrDistance
        TgridYIndex= txPos[1] // self.corrDistance 
        RgridXIndex= (rxPos[0]-txPos[0]) // self.corrDistance 
        RgridYIndex= (rxPos[1]-txPos[1]) //self.corrDistance 
        macrokey = (TgridXIndex,TgridYIndex,RgridXIndex,RgridYIndex)
        key = (txPos[0],txPos[1],rxPos[0],rxPos[1])
        
        if not macrokey in self.dMacrosGenerated:
            self.create_macro(txPos,rxPos)
        macro = self.dMacrosGenerated[macrokey]
        
        small = self.create_small_param(angles,macro)
        
        keyChannel = (txPos,rxPos)
        self.dChansGenerated[keyChannel] = (macro,small)
        return(macro,small)
