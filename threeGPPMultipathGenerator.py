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
        'zodFun'
    ])
    #Crear diccionarios
    CphiNLOS = {4 : 0.779, 5 : 0.860 , 8 : 1.018, 10 : 1.090, 
                11 : 1.123, 12 : 1.146, 14 : 1.190, 15 : 1.211, 16 : 1.226, 
                19 : 1.273, 20 : 1.289, 25 : 1.358}
    CtetaNLOS = {8 : 0.889, 10 : 0.957, 11 : 1.031, 12 : 1.104, 
                 15 : 1.1088, 19 : 1.184, 20 : 1.178, 25 : 1.282}
    alpham = {1 : 0.0447, 2 : 0.0447, 3 : 0.1413, 4 : 0.1413, 5 : 0.2492,
              6 : 0.2492, 7 :  0.3715, 8 :  0.3715, 9 : 0.5129, 10 : 0.5129,
              11 : 0.6797, 12 : 0.6797, 13 :  0.8844, 14 :  0.8844, 15 : 1.1481,
              16 : 1.1481, 17 : 1.5195, 18 : 1.5195, 19 : 2.1551, 20 : 2.1551}
    
    #RMa hasta 7GHz y el resto hasta 100GHz
    def __init__(self, fc = 28, sce = "UMi", corrDistance = 15.0):
        self.frecRefGHz = fc
        self.scenario = sce
        self.corrDistance = corrDistance
        self.dMacrosGenerated = {}
        self.dChansGenerated = {}
        self.bLargeBandwidthOption = False 
        self.clight=3e8
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
        d2D = bPos[0]
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
        #zod_offset_mu = lambda los : 0 if los else 
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
        self.dMacrosGenerated[key]=self.ThreeGPPMacroParams(los,PL,ds,asa,asd,zsa,zsd,K,sf,zod_offset_mu)
   
    def create_small_param(self, angles, macro):
        los = macro.los
        DS = macro.ds
        ASA = macro.asa
        ASD = macro.asd
        ZSA = macro.zsa
        ZSD = macro.zsd
        K = macro.K
        ZOD = macro.zod_offset_mu
        #SF = macro.sf
        #O2I = macro.O2I
    
        if los:
            param = self.scenarioParamsLOS
        else:
            param = self.scenarioParamsNLOS
        N =param.N
        M = param.M
        rt = param.rt
        #Generate cluster delays
        aux = []
        for i in range(N):
            X=np.random.uniform(0,1)
            aux.append(-rt*DS*np.log(X))
        tau_prima = np.array(aux)
        tau_prima = tau_prima-np.amin(tau_prima)
        tau = np.array(sorted(tau_prima))
        Ctau = 0.7705 - 0.0433*K + 0.0002*K**2 + 0.000017*K**3
        if los:
            tau = tau / Ctau 
        #Generate cluster powers
        xi = self.scenarioParamsLOS.xi
        aux1 = []
        for i in range(N):
            Zn = xi*np.random.rand(N,1)
            aux1.append(np.exp(-tau[i]*((rt-1)/(rt*DS)))*10**(-(Zn/10)))
        powPrima = np.array(aux1)
        if los:
            p1LOS = K/(K+1)
            powC = (1/K+1)*(powPrima/np.sum(powPrima))
            powC[0] = powC[0] + p1LOS
        else:
            powC = powPrima/np.sum(powPrima)
        #Remove clusters with less than -25 dB power compared to the maximum cluster power. The scaling factors need not be 
        #changed after cluster elimination
        maxP=np.max(powC)
        #------------------------------------------------
        #print('mas',maxP)
        #print('1',powC)
        for i in range(len(tau)):
            if (powC[i] < (maxP-(10**-2.5))).any():
                np.delete(powC,i)
        #Hay que eliminar también en el array de retardos tau?????
        #tau = np.array([tau[x] for x in range(0,len(tau)) if powC[x]>maxP*(10**-2.5)])
        #powC = np.array([x for x in powC if x>maxP*(10**-2.5)])
        nClusters=np.size(powC)
        #----------------------------------------------
        #Generate arrival angles and departure angles for both azimuth and elevation   
        #azimut
        if los:
            Cphi = self.CphiNLOS.get(N)*(1.1035 - 0.028*K - 0.002*(K**2) + 0.0001*(K**3))
        else:
            Cphi = self.CphiNLOS.get(N)
        #Cphi = lambda los: self.CphiNLOS.get(N)*(1.1035 - 0.028*K - 0.002*(K**2) + 0.0001*(K**3)) if los else self.CphiNLOS.get(N)
        auxPhi = []
        auxPhi2 = []
        Y = []
        X = []
        for i in range(N):
            auxPhi.append(((2*(ASA/1.4)*np.sqrt(-np.log(powC[i]/maxP)))/Cphi))
            auxPhi2.append(((2*(ASD/1.4)*np.sqrt(-np.log(powC[i]/maxP)))/Cphi))
            Y.append(np.random.normal(0,(ASA/7)**2))
            X.append(np.random.uniform(-1,1))
        phiAOAprima = np.array(auxPhi)
        phiAODprima = np.array(auxPhi2)
        auxphiAOA = []
        auxphiAOD = []
        if los:
            for i in range(N):
                auxphiAOA.append((X*phiAOAprima + Y) - (X[1]*phiAOAprima[1] + Y[1] - angles[1])) 
                auxphiAOD.append((X*phiAODprima + Y) - (X[1]*phiAODprima[1] + Y[1] - angles[0])) 
        else:
            auxphiAOA.append((X*phiAOAprima + Y + angles[1]))
            auxphiAOD.append((X*phiAODprima + Y + angles[0]))
        phiAOA = np.array(auxphiAOA)
        phiAOD = np.array(auxphiAOD)
        auxmphiAOA = []
        auxmphiAOD = []
        casa = param.casa
        for i in range(N):
            for j in range(M):
                auxmphiAOA.append(phiAOA[i] + casa*self.alpham.get(j+1))
                auxmphiAOD.append(phiAOD[i] + casa*self.alpham.get(j+1))
        mphiAOA = np.array(auxmphiAOA)
        mphiAOD = np.array(auxmphiAOD)
        #-------------------------------------------------------------------
        #zenith (elevation)
        if los:
            Cteta = self.CtetaNLOS.get(N)*(1.3086 + 0.0339*K -0.0077*(K**2) + 0.0002*(K**3))
        else:
            Cteta = Cteta = self.CtetaNLOS.get(N)
        #Cteta = lambda los : self.CtetaNLOS.get(N)*(1.3086 + 0.0339*K -0.0077*(K**2) + 0.0002*(K**3)) if los else self.CtetaNLOS.get(N)
        auxtetaZOAprima = []
        auxtetaZODprima = []
        X2 = []
        Y1 = []
        Y2 = []
        for i in range(N):
            auxtetaZOAprima.append(-((ZSA*np.log(powC[i]/maxP)) / Cteta))
            auxtetaZODprima.append(-((ZSD*np.log(powC[i]/maxP)) / Cteta))
            Y1.append(np.random.normal(0,(ZSA/7)**2))
            Y2.append(np.random.normal(0,(ZSD/7)**2))
            X2.append(np.random.uniform(-1,1))
        tetaZOAprima = np.array(auxtetaZOAprima)
        tetaZODprima = np.array(auxtetaZODprima)
        auxtetaZOA = []
        auxtetaZOD = []
        if los:
            for i in range(N):
                auxtetaZOA.append((X2[i]*tetaZOAprima[i] + Y1[i]) - (X2[1]*tetaZOAprima[1] + Y1[1] - angles[3]))
                auxtetaZOD.append((X2[i]*tetaZODprima[i] + Y2[i] + ZOD) - (X2[1]*tetaZODprima[1] + Y2[1] - angles[2]))
        else:
            auxtetaZOA.append((X[i]*tetaZOAprima[i] + Y1[i] + angles[3])) 
            auxtetaZOD.append((X[i]*tetaZODprima[i] + Y2[i] + angles[2] + ZOD))
        tetaZOA = np.array(auxtetaZOA)
        tetaZOD = np.array(auxtetaZOD)
        auxmtetaZOA = []
        auxmtetaZOD = []
        czsa = param.czsa
        for i in range(N):
            for j in range(M):
                auxmtetaZOA.append(tetaZOA[i] + czsa*self.alpham.get(j+1))
                auxmtetaZOD.append(tetaZOD[i] + (3/8)*(10**ZSD)*self.alpham.get(j+1))
        mtetaZOA = np.array(auxmtetaZOA)
        mtetaZOD = np.array(auxmtetaZOD)
        #if 180 < mtetaZOA < 360:
         #   mtetaZOA = 360 - mtetaZOA 
        angles = [mphiAOA,mtetaZOA,mphiAOD,mtetaZOD]
        return(tau,powC,angles)
        
    def create_channel(self, txPos, rxPos):
        aPos = np.array(txPos)
        bPos = np.array(rxPos)
        vLOS = bPos-aPos
        d2D= bPos[0]
        d3D=np.linalg.norm(bPos-aPos)
        hut=bPos[2]
        d=np.linalg.norm(vLOS)
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
        
        lista = []
        for i in range(len(macro)):
            lista.append(macro[i])

        for i in range(len(lista)):
            if isinstance(lista[i], int):
                lista[i] = str(lista[i])
            #else:
             #   lista[i] = np.array_str(lista[i])
            ##print(type(lista[i]))
            
        #print("Macro for BS: (" + str(txPos[0]) +"," + str(txPos[1]) + ")m and UT: (" + str(rxPos[0]) + "," + str(rxPos[1]) + ")m.")
        df = pd.Series({'LOS' : str(lista[0]), 'Path Loss' : lista[1], 'DS' : lista[2], 'ASA' : lista[3],'ASD' : lista[4],'ZSA' : lista[5],'ZSD' : lista[6],'K' : lista[7],'SF' : lista[8],'ZOD' : str(lista[9])})
        #print(df)
        
        
        
        #self.dChansGenerated[key] = ch.MultipathChannel(txPos,rxPos,macro,small)
        return(macro,small)