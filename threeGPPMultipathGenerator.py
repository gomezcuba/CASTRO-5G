import numpy as np
import collections as col
import multipathChannel as ch

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
    def __init__(self, fc = 28, sce = "UMi", corrDistance = 5.0):
        self.frecRefGHz = fc
        self.scenario = sce
        self.corrDistance = corrDistance
        self.dMacrosGenerated = {}
        self.dChansGenerated = {}
        self.bLargeBandwidthOption = False 
        self.clight=3e8
        if sce.find("UMi")>=0:
            #LOS Probability (distance is in meters)
            self.senarioLosProb = lambda d2D : 1 if d2D<18 else 18.0/d2D + np.exp(-d2D/36.0)*(1-18.0/d2D)
            self.scenarioParamsLOS = self.ThreeGPPScenarioParams(
                -0.24*np.log10(1+fc)-7.14,
                0.39,
                -.05*np.log10(1+fc)+1.21,
                0.41,
                -0.08*np.log10(1+fc)+1.73,
                0.014*np.log10(1+fc)+0.28,
                -0.1*np.log10(1+fc)+0.73,
                -0.03*np.log10(1+fc)+0.34,
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
                -.24*np.log10(1+fc)-6.83,
                .16*np.log10(1+fc)+0.28,
                -.23*np.log10(1+fc)+1.53,
                .11*np.log10(1+fc)+0.33,
                -.08*np.log10(1+fc)+1.81,
                0.05*np.log10(1+fc)+0.3,
                -.04*np.log10(1+fc)+0.92,
                -.07*np.log10(1+fc)+0.41,
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
            # C = lambda hut :  0 if hut<=23 else ((hut-13.0)/10.0)**1.5
            self.senarioLosProb = lambda d2D,hut : 1 if d2D<18 else (18.0/d2D + np.exp(-d2D/63.0)*(1-18.0/d2D))*(1 + (0 if hut<=23 else ((hut-13.0)/10.0)**1.5)*1.25*np.exp(d2D/100.0)*(-d2D/150.0)**3.0) 
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
                20,
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
                   [-0.5,-0.2,0.8,0,1,-0.3,0.4]
                   [0,0,-0.2,0.5,-0.3,1,0],
                   [-0.8,0,0,0,0.4,0,1]]),
                self.scenarioPlossUMaLOS,
                self.ZSDUMaLOS,
                0,
            )
            self.scenarioParamsNLOS = self.ThreeGPPScenarioParams(
                -6.28 - 0.204*np.log10(fc),
                0.39,
                1.5 - 0.144*np.log10(fc),
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
                   [0,0,0.6,0.4,1,0,0]
                   [0,0,-0.5,0.5,0,1,0],
                   [-0.4,0,0,-0.1,0,0,1]]),
                self.scenarioPlossUMaNLOS,
                self.scenarioUMaNLOS,
                self.ZSDUMaNLOS,
                self.ZODUMaNLOS,
            )
        elif sce.find("RMa")>=0:
            self.senarioLosProb = lambda d2D : 1 if d2D<10 else np.exp(-(d2D-10.0)/1000.0)
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
            self.senarioLosProb = lambda d2D : 1 if d2D<1.2 else (np.exp(-(d2D-1.2)/4.7) if 1.2<d2D<6.5 else (np.exp(-(d2D-6.5)/32.6))*0.32)
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
                self.ZODInNLOS,
            )
        elif sce.find("InH-Office-Open")>=0:
            self.senarioLosProb = lambda d2D : 1 if d2D<=5 else (np.exp(-(d2D-5.0)/70.8) if 5<d2D<49 else (np.exp(-(d2D-49.0)/211.7))*0.54)
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
                self.ZODInNLOS,
            )
            

    def scenarioPlossUMiLOS(self,d3D,d2D,hut,O2I):        
        if not O2I:
            n=1
        else:
            Nf = np.random.uniform(4,8)
            n=np.random.uniform(1,Nf)
        hut = 3*(n-1) + 1.5
        prima_dBP = (4*10.0*hut*self.frecRefGHz) / self.clight
        if(d2D<prima_dBP):
            ploss = 32.4 + 21.0*np.log10(d3D)+20.0*np.log10(self.frecRefGHz)
        else:
            ploss = 32.4 + 40*np.log10(d3D)+20.0*np.log10(self.frecRefGHz)-9.5*np.log10(prima_dBP**2+(10.0-hut)**2)
        return(ploss)
    def scenarioPlossUMiNLOS(self,d3D,d2D,hut,O2I):
        if not O2I:
            n = 1
        else:
            Nf = np.random.uniform(4,8)
            n = np.random.uniform(1,Nf)
        hut = 3*(n-1) + 1.5
        PL1 = 35.3*np.log10(d3D) + 22.4 + 21.3*np.log10(self.frecRefGHz)-0.3*(hut - 1.5)
        PL2 = self.scenarioPlossUMiLOS(d3D,d2D,hut,O2I)
        ploss = np.maximum(PL1,PL2)
        return(ploss)
    
    def scenarioPlossUMaLOS(self,d3D,d2D,hut,O2I):
        if not O2I:
            n = 1
        else:
            Nf = np.random.uniform(4,8)
            n = np.random.uniform(1,Nf)
        hut_aux = 3*(n-1) + 1.5
        prima_dBP = (4*25.0*hut_aux*self.frecRefGHz) / self.clight
        if(d2D<prima_dBP):
            ploss = 28.0 + 22.0*np.log10(d3D)+20.0*np.log10(self.frecRefGHz)
        else:
            ploss = 28.0 + 40.0*np.log10(d3D)+20.0*np.log10(self.frecRefGHz)-9.0*np.log10(np.exp(prima_dBP)+np.exp(25.0-hut_aux))
        return(ploss)
    def scenarioPlossUMaNLOS(self,d3D,d2D,hut,O2I):
        if not O2I:
            n = 1
        else:
            Nf = np.random.uniform(4,8)
            n = np.random.uniform(1,Nf)
        hut_aux = 3*(n-1) + 1.5
        PL1 = 13.54 + 39.08*np.log10(d3D) + 20.0*np.log10(self.frecRefGHz) - 0.6*(hut_aux - 1.5)
        PL2 = self.scenarioPlossUMaLOS(d3D,d2D,hut,O2I)
        ploss = np.maximum(PL1,PL2)
        return(ploss)
    
    def scenarioPlossRMaLOS(self,d3D,d2D,hut,O2I):
        dBp = (2*np.pi*35.0*1.5*self.frecRefGHz)/self.clight
        if(d2D<dBp):
            ploss = 20*np.log10((40.0*np.pi*d3D*self.frecRefGHz)/3.0)+np.minimum(0.03*(5.0**1.72),10)*np.log10(d3D)-np.minimum(0.44*(5.0**1.72),14.77)+0.002*np.log10(5.0)*d3D
        else:
            PL1 = 20*np.log10((40.0*np.pi*d3D*self.frecRefGHz)/3.0)+np.minimum(0.03*(5.0**1.72),10)*np.log10(d3D)-np.minimum(0.44*(5.0**1.72),14.77)+0.002*np.log10(5.0)*d3D
            ploss = PL1*dBp + 40.0*np.log10(d3D/dBp)
        return(ploss)
    def scenarioPlossRMaNLOS(self,d3D,d2D,hut,O2I):
        PL1= 161.04 - 7.1*np.log10(20) - 7.5*np.log10(5) - (24.37 - 3.7*((5/35)**2))*np.log10(35) + (43.42 - 3.1*np.log10(35))*(np.log10(d3D)-3) + 20*np.log10(self.frecRefGHz) - (3.2*np.log10(11.75*1.5))**2 - 4.97 
        PL2= self.scenarioPlossRMaLOS(d3D,d2D,hut,O2I)
        ploss = np.maximum(PL1,PL2)
        return(ploss)
    
    def scenarioPlossInLOS(self,d3D,d2D,hut,O2I):
        ploss= 32.4 + 17.3*np.log10(d3D) + 20.0*np.log10(self.frecRefGHz)
        return(ploss)
    def scenarioPlossInNLOS(self,d3D,d2D,hut,O2I):
        PL1= 38.3*np.log10(d3D) + 17.30 + 24.9*np.log10(self.frecRefGHz)
        PL2= self.scenarioPlossInLOS(d3D,d2D,hut,O2I)
        ploss= np.maximum(PL1,PL2)
        return(ploss)  
    
    def ZSDUMiLOS(self,d3D,d2D,hbs,hut):  
        zsd_mu = np.maximum(-0.21, -14.8*(d2D/1000.0) + 0.01*np.abs(hut-hbs) +0.83)
        zsd_sg = 0.35
        zsd = min( 10.0**( zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
        return(zsd)
    def ZSDUMaLOS(self,d3D,d2D,hbs,hut):
        zsd_mu = np.maximum(-0.5, -2.1*(d2D/1000.0) - 0.01*(hut - 1.5) + 0.75)
        zsd_sg = 0.40
        zsd = min( 10.0**( zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
        return(zsd) 
    def ZSDRMaLOS(self,d3D,d2D,hbs,hut):
        zsd_mu = np.maximum(-1, -0.17*(d2D/1000.0) - 0.01*(hut - 1.5) + 0.22)
        zsd_sg = 0.34
        zsd = min( 10.0**( zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
        return(zsd)
    def ZSDInLOS(self,d3D,d2D,hbs,hut):
        zsd_mu = -1.43*np.log10(1 + self.frecRefGHz) + 2.228
        zsd_sg = 0.13*np.log10(1 + self.frecRefGHz) + 0.30
        zsd = min( 10.0**( zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
        return(zsd)
    
    def ZSDUMiNLOS(self,d3D,d2D,hbs,hut):     
        zsd_mu = np.maximum(-0.5, -3.1*(d2D/1000.0) + 0.01*np.maximum(hut-hbs,0.0) +0.2)
        zsd_sg = 0.35
        zsd = min( 10.0**( zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
        return(zsd)
    def ZSDUMaNLOS(self,d3D,d2D,hbs,hut):     
        zsd_mu = np.maximum(-0.5, -2.1*(d2D/1000.0) - 0.01*(hut - 1.5) + 0.9)
        zsd_sg = 0.49
        zsd = min( 10.0**( zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
        return(zsd)
    def ZSDRMaNLOS(self,d3D,d2D,hbs,hut):     
        zsd_mu = np.maximum(-1, -0.19*(d2D/1000) - 0.01*(hut - 1.5) + 0.28)
        zsd_sg = 0.30
        zsd = min( 10.0**( zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
        return(zsd)
    def ZSDInNLOS(self,d3D,d2D,hbs,hut):     
        zsd_mu = 1.08
        zsd_sg = 0.36
        zsd = min( 10.0**( zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
        return(zsd)
    
    def ZODUMiNLOS(self,d2D,hut):
        zod_offset_mu = -10.0**(-1.5*np.log10(np.maximum(10,d2D)) + 3.3)
        return(zod_offset_mu)
    def ZODUMaNLOS(self,d2D,hut):
        zod_offset_mu= 7.66*np.log10(self.frecRefGHz) - 5.96 - 10**((0.208*np.log10(self.frecRefGHz) - 0.782)*np.log10(np.maximum(25.0,d2D)) - 0.13*np.log10(self.frecRefGHz) + 2.03 - 0.07*(hut - 1.5))
        return(zod_offset_mu)
    def ZODRMaNLOS(self,d2D,hut):
        zod_offset_mu= np.arctan((35.0 - 3.5)/d2D) - np.arctan((35.0 - 1.5)/d2D)
        return(zod_offset_mu)
   
    #Large Scale Parametres
    def create_macro(self, txPos, rxPos):
        aPos = np.array(txPos) 
        bPos = np.array(rxPos) 
        d3D=np.linalg.norm(bPos-aPos) 
        d2D=np.sqrt((rxPos[0]-txPos[0])**2.0+(rxPos[1]-txPos[1])**2.0)
        hbs = aPos[2]
        hut = bPos[2]
        pLos=self.senarioLosProb(d2D)
        los = (np.random.rand(1) <= pLos)
        vIndep=np.random.randn(7,1)
        vDepLOS=self.scenarioParamsLOS.Cc@vIndep
        vDepNLOS=self.scenarioParamsNLOS.Cc@vIndep
        if los:
            ds = 10.0**( self.scenarioParamsLOS.ds_mu + self.scenarioParamsLOS.ds_sg * vDepLOS[2] )
            asa = min( 10.0**( self.scenarioParamsLOS.asa_mu + self.scenarioParamsLOS.asa_sg * vDepLOS[4] ), 104.0)
            asd = min( 10.0**( self.scenarioParamsLOS.asd_mu + self.scenarioParamsLOS.asd_sg * vDepLOS[3] ), 104.0)
            zsa = min( 10.0**( self.scenarioParamsLOS.zsa_mu + self.scenarioParamsLOS.zsa_sg * vDepLOS[6] ), 52.0)
            zsd = self.scenarioParamsLOS.zsdFun(d3D,d2D,hbs,hut)
            zod_offset_mu = 0
            K= self.scenarioParamsLOS.K_mu
            sf = self.scenarioParamsNLOS.sf_sg*vDepLOS[0]
            PLdB = self.scenarioParamsLOS.pLossFun(d3D,d2D,hut,False)
            PL=PLdB+sf
        else:
            ds = 10.0**( self.scenarioParamsNLOS.ds_mu + self.scenarioParamsNLOS.ds_sg * vDepNLOS[2] )
            asa = min( 10.0**( self.scenarioParamsNLOS.asa_mu + self.scenarioParamsNLOS.asa_sg * vDepNLOS[4] ), 104.0)
            asd = min( 10.0**( self.scenarioParamsNLOS.asd_mu + self.scenarioParamsNLOS.asd_sg * vDepNLOS[3] ), 104.0)
            zsa = min( 10.0**( self.scenarioParamsNLOS.zsa_mu + self.scenarioParamsNLOS.zsa_sg * vDepNLOS[6] ), 52.0)
            zsd = self.scenarioParamsNLOS.zsdFun(d3D,d2D,hbs,hut)
            zod_offset_mu = self.scenarioParamsNLOS.zodFun(d2D,hut)
            K= self.scenarioParamsNLOS.K_mu
            sf = self.scenarioParamsNLOS.sf_sg*vDepNLOS[0]
            sflos=self.scenarioParamsLOS.sf_sg*vDepLOS[0]
            PLdB = self.scenarioParamsNLOS.pLossFun(d3D,d2D,hut,True)
            PL=np.maximum(PLdB + sf,PLdB + sflos)
        TgridXIndex= txPos[0] // self.corrDistance
        TgridYIndex= txPos[1] // self.corrDistance 
        RgridXIndex= rxPos[0] // self.corrDistance 
        RgridYIndex= rxPos[1] //self.corrDistance 
        key = (TgridXIndex,TgridYIndex,RgridXIndex,RgridYIndex)
        self.dMacrosGenerated[key]=self.ThreeGPPMacroParams(los,PL,ds,asa,asd,zsa,zsd,K,sf,zod_offset_mu)
   
    def create_small_param(self, angles, macro, O2I):
        los = macro.los
        DS = macro.ds
        ASA = macro.asa
        ASD = macro.asd
        ZSA = macro.zsa
        ZSD = macro.zsd
        K = macro.K
        #SF = macro.sf
        #O2I = macro.O2I
        if los:
            N = self.scenarioParamsLOS.N
            M = self.scenarioParamsLOS.M
            rt = self.scenarioParamsLOS.rt
            ZOD = 0
            #Generate cluster delays
            aux = []
            for i in range(N):
                X=np.random.rand(1)
                aux.append(-rt*DS*np.log(X))
            tau_prima = np.array(aux)
            tau_prima = tau_prima-np.amin(tau_prima)
            tau = np.array(sorted(tau_prima))
            Ctau = 0.7705 - 0.0433*K + 0.0002*K**2 + 0.000017*K**3
            tau = tau / Ctau 
            #Generate cluster powers
            xi = self.scenarioParamsLOS.xi
            aux1 = []
            for i in range(N):
                Zn = xi*np.random.rand(N,1)
                aux1.append(np.exp(-tau[i]*((rt-1)/(rt*DS)))*10**(-(Zn/10)))
            powPrima = np.array(aux1)
            p1LOS = K/(K+1)
            powC = (1/K+1)*(powPrima/np.sum(powPrima))
            powC[0] = powC[0] + p1LOS
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
            Cphi = self.CphiNLOS.get(N)*(1.1035 - 0.028*K - 0.002*(K**2) + 0.0001*(K**3)) 
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
            for i in range(N):
                auxphiAOA.append((X*phiAOAprima + Y) - (X[1]*phiAOAprima[1] + Y[1] - angles[1])) 
                auxphiAOD.append((X*phiAODprima + Y) - (X[1]*phiAODprima[1] + Y[1] - angles[0])) 
            phiAOA = np.array(auxphiAOA)
            phiAOD = np.array(auxphiAOD)
            auxmphiAOA = []
            auxmphiAOD = []
            for i in range(N):
                for j in range(M):
                    auxmphiAOA.append(phiAOA[i] + self.scenarioParamsLOS.casa*self.alpham.get(j+1))
                    auxmphiAOD.append(phiAOD[i] + self.scenarioParamsLOS.casa*self.alpham.get(j+1))
            mphiAOA = np.array(auxmphiAOA)
            mphiAOD = np.array(auxmphiAOD)
            #-------------------------------------------------------------------
            #zenith (elevation)
            Cteta = self.CtetaNLOS.get(N)*(1.3086 + 0.0339*K -0.0077*(K**2) + 0.0002*(K**3))
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
            for i in range(N):
                auxtetaZOA.append((X2[i]*tetaZOAprima[i] + Y1[i]) - (X2[1]*tetaZOAprima[1] + Y1[1] - angles[3]))
                auxtetaZOD.append((X2[i]*tetaZODprima[i] + Y2[i] + ZOD) - (X2[1]*tetaZODprima[1] + Y2[1] - angles[2]))
            tetaZOA = np.array(auxtetaZOA)
            tetaZOD = np.array(auxtetaZOD)
            auxmtetaZOA = []
            auxmtetaZOD = []
            for i in range(N):
                for j in range(M):
                    auxmtetaZOA.append(tetaZOA[i] + self.scenarioParamsLOS.czsa*self.alpham.get(j+1))
                    auxmtetaZOD.append(tetaZOD[i] + (3/8)*(10**ZSD)*self.alpham.get(j+1))
            mtetaZOA = np.array(auxmtetaZOA)
            mtetaZOD = np.array(auxmtetaZOD)
        else:
            N = self.scenarioParamsNLOS.N
            M = self.scenarioParamsNLOS.M
            rt = self.scenarioParamsNLOS.rt
            ZOD = macro.zod_offset_mu
            aux = []
            for i in range(N):
                X=np.random.rand(1)
                aux.append(-rt*DS*np.log(X))
            tau_prima = np.array(aux)
            tau_prima = tau_prima-np.amin(tau_prima)
            tau = np.array(sorted(tau_prima))
            xi = self.scenarioParamsLOS.xi
            aux1 = []
            for i in range(N):
                Zn = xi*np.random.rand(N,1)
                aux1.append(np.exp(-tau[i]*((rt-1)/(rt*DS)))*10**(-(Zn/10)))
            powPrima = np.array(aux1)
            powC = powPrima/np.sum(powPrima)            
            #Remove clusters with less than -25 dB power compared to the maximum cluster power. The scaling factors need not be 
            #changed after cluster elimination
            maxP=np.max(powC)
            #------------------------------------------------
            #print('mas',maxP)
            for i in range(len(tau)):
                if (powC[i] < (maxP-(10**-2.5))).any():
                    np.delete(powC,i)
            #tau = np.array([tau[x] for x in range(0,len(tau)) if powC[x]>maxP*(10**-2.5)])
            #powC = np.array([x for x in powC if x>maxP*(10**-2.5)])
            nClusters=np.size(powC)
            #----------------------------------------------
            #Generate arrival angles and departure angles for both azimuth and elevation     
            #azimut
            Cphi = self.CphiNLOS.get(N) 
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
            for i in range(N):
                auxphiAOA.append((X*phiAOAprima + Y + angles[1]))
                auxphiAOD.append((X*phiAODprima + Y + angles[0]))
            phiAOA = np.array(auxphiAOA)
            phiAOD = np.array(auxphiAOD)
            auxmphiAOA = []
            auxmphiAOD = []
            for i in range(N):
                for j in range(M):
                    auxmphiAOA.append(phiAOA[i] + self.scenarioParamsNLOS.casa*self.alpham.get(j+1))
                    auxmphiAOD.append(phiAOD[i] + self.scenarioParamsNLOS.casa*self.alpham.get(j+1))
            mphiAOA = np.array(auxmphiAOA)
            mphiAOD = np.array(auxmphiAOD)
            #-------------------------------------------------------------------
            #zenith (elevation)
            Cteta = self.CtetaNLOS.get(N)
            auxtetaZOAprima = []
            auxtetaZODprima = []
            for i in range(N):
                auxtetaZOAprima.append(-((ZSA*np.log(powC[i]/maxP)) / Cteta))
                auxtetaZODprima.append(-((ZSD*np.log(powC[i]/maxP)) / Cteta))
            tetaZOAprima = np.array(auxtetaZOAprima)
            tetaZODprima = np.array(auxtetaZODprima)
            auxtetaZOA = []
            auxtetaZOD = []
            Y1 = []
            Y2 = []
            if O2I:
                tetaZOA_mu = 90
            else:
                tetaZOA_mu = angles[3]
            for i in range(N):
                Y1.append(np.random.normal(0,(ZSA/7)**2))
                Y2.append(np.random.normal(0,(ZSD/7)**2))
                auxtetaZOA.append((X[i]*tetaZOAprima[i] + Y1[i] + tetaZOA_mu)) 
                auxtetaZOD.append((X[i]*tetaZODprima[i] + Y2[i] + angles[2] + ZOD))
            tetaZOA = np.array(auxtetaZOA)
            tetaZOD = np.array(auxtetaZOD)
            auxmtetaZOA = []
            auxmtetaZOD = []
            for i in range(N):
                for j in range(M):
                    auxmtetaZOA.append(tetaZOA[i] + self.scenarioParamsNLOS.czsa*self.alpham.get(j+1))
                    auxmtetaZOD.append(tetaZOD[i] + (3/8)*(10**ZSD)*self.alpham.get(j+1))
            mtetaZOA = np.array(auxmtetaZOA)
            mtetaZOD = np.array(auxmtetaZOD)
        #if 180 < mtetaZOA < 360:
         #   mtetaZOA = 360 - mtetaZOA
        
    def create_channel(self, txPos, rxPos, O2I):
        aPos = np.array(txPos)
        bPos = np.array(rxPos)
        vLOS = bPos-aPos
        d3D=np.linalg.norm(bPos-aPos) 
        d2D=np.sqrt((rxPos[0]-txPos[0])**2.0+(rxPos[1]-txPos[1])**2.0)
        hut=txPos[2]
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
        #create_small_params necesita O2I, por lo que convertimos macro a lista y añadimos O2I
        #para poder enviarlo ya desde macro
        #lst = list(macro)
        #lst.append(O2I)
        #macro = tuple(lst)
        self.create_small_param(angles,macro,O2I)
        return(True)
