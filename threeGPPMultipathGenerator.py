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
            

    def scenarioPlossUMiLOS(self,d3D,d2D,hut):
        outdoor = True
        if outdoor:
            n=1
        else:
            Nf=np.ramdon.rand(4,8)
            n=np.ramdon.rand(1,Nf)
        hut= 3*(n-1) + 1.5
        prima_dBP = (4*10.0*hut*self.frecRefGHz) / self.clight
        if(d2D<prima_dBP):
            ploss = 32.4 + 21.0*np.log10(d3D)+20.0*np.log10(self.frecRefGHz)
        else:
            ploss = 32.4 + 40*np.log10(d3D)+20.0*np.log10(self.frecRefGHz)-9.5*np.log10(prima_dBP**2+(10.0-hut)**2)
        return(ploss)
    def scenarioPlossUMiNLOS(self,d3D,d2D,hut):
        outdoor = True
        if outdoor:
            n=1
        else:
            Nf=np.ramdon.rand(4,8)
            n=np.ramdon.rand(1,Nf)
        hut= 3*(n-1) + 1.5
        PL1 = 35.3*np.log10(d3D) + 22.4 + 21.3*np.log10(self.frecRefGHz)-0.3*(hut - 1.5)
        PL2 = self.scenarioPlossUMiLOS(d3D,d2D,hut)
        ploss = np.maximum(PL1,PL2)
        return(ploss)
    
    def scenarioPlossUMaLOS(self,d3D,d2D,hut):
        #Hay que diferenciar entre outdoor e indoor
        outdoor = True
        if outdoor:
            n=1
        else:
            Nf=np.ramdon.rand(4,8)
            n=np.ramdon.rand(1,Nf)
        hut_aux= 3*(n-1) + 1.5
        prima_dBP = (4*25.0*hut_aux*self.frecRefGHz) / self.clight
        if(d2D<prima_dBP):
            ploss = 28.0 + 22.0*np.log10(d3D)+20.0*np.log10(self.frecRefGHz)
        else:
            ploss = 28.0 + 40.0*np.log10(d3D)+20.0*np.log10(self.frecRefGHz)-9.0*np.log10(np.exp(prima_dBP)+np.exp(25.0-hut_aux))
        return(ploss)
    def scenarioPlossUMaNLOS(self,d3D,d2D,hut):
        outdoor = True
        if outdoor:
            n=1
        else:
            Nf=np.ramdon.rand(4,8)
            n=np.ramdon.rand(1,Nf)
        hut_aux= 3*(n-1) + 1.5
        PL1 = 13.54 + 39.08*np.log10(d3D) + 20.0*np.log10(self.frecRefGHz) - 0.6*(hut_aux - 1.5)
        PL2 = self.scenarioPlossUMaLOS(d3D,d2D,hut)
        ploss = np.maximum(PL1,PL2)
        return(ploss)
    
    def scenarioPlossRMaLOS(self,d3D,d2D,hut):
        dBp = (2*np.pi*35.0*1.5*self.frecRefGHz)/self.clight
        if(d2D<dBp):
            ploss = 20*np.log10((40.0*np.pi*d3D*self.frecRefGHz)/3.0)+np.minimum(0.03*(5.0**1.72),10)*np.log10(d3D)-np.minimum(0.44*(5.0**1.72),14.77)+0.002*np.log10(5.0)*d3D
        else:
            PL1 = 20*np.log10((40.0*np.pi*d3D*self.frecRefGHz)/3.0)+np.minimum(0.03*(5.0**1.72),10)*np.log10(d3D)-np.minimum(0.44*(5.0**1.72),14.77)+0.002*np.log10(5.0)*d3D
            ploss = PL1*dBp + 40.0*np.log10(d3D/dBp)
        return(ploss)
    def scenarioPlossRMaNLOS(self,d3D,d2D,hut):
        PL1= 161.04 - 7.1*np.log10(20) - 7.5*np.log10(5) - (24.37 - 3.7*((5/35)**2))*np.log10(35) + (43.42 - 3.1*np.log10(35))*(np.log10(d3D)-3) + 20*np.log10(self.frecRefGHz) - (3.2*np.log10(11.75*1.5))**2 - 4.97 
        PL2= self.scenarioPlossRMaLOS(d3D,d2D,hut)
        ploss = np.maximum(PL1,PL2)
        return(ploss)
    
    def scenarioPlossInLOS(self,d3D,d2D,hut):
        ploss= 32.4 + 17.3*np.log10(d3D) + 20.0*np.log10(self.frecRefGHz)
        return(ploss)
    def scenarioPlossInNLOS(self,d3D,d2D,hut):
        PL1= 38.3*np.log10(d3D) + 17.30 + 24.9*np.log10(self.frecRefGHz)
        PL2= self.scenarioPlossInLOS(d3D,d2D,hut)
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
    def create_macro(self,txPos = (0,0,0), rxPos = (0,0,0)):
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
            PLdB = self.scenarioParamsLOS.pLossFun(d3D,d2D,hut)
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
            PLdB = self.scenarioParamsNLOS.pLossFun(d3D,d2D,hut)
            PL=np.maximum(PLdB + sf,PLdB + sflos)
        TgridXIndex= txPos[0] // self.corrDistance
        TgridYIndex= txPos[1] // self.corrDistance 
        RgridXIndex= rxPos[0] // self.corrDistance 
        RgridYIndex= rxPos[1] //self.corrDistance 
        key = (TgridXIndex,TgridYIndex,RgridXIndex,RgridYIndex)
        self.dMacrosGenerated[key]=self.ThreeGPPMacroParams(los,PL,ds,asa,asd,zsa,zsd,K,sf,zod_offset_mu)
   
    def create_small_param(self, los, txPos = (0,0,0), rxPos = (0,0,0)):
        aPos = np.array(txPos) 
        bPos = np.array(rxPos) 
        d3D = np.linalg.norm(bPos-aPos) 
        d2D = np.sqrt((rxPos[0]-txPos[0])**2.0+(rxPos[1]-txPos[1])**2.0)
        hbs = aPos[2]
        hut = bPos[2]
        if los:
            N=self.scenarioParamsLOS.N
            M=self.scenarioParamsLOS.M
            rt=self.scenarioParamsLOS.rt
            DS=self.scenarioParamsLOS.ds_mu
            K=self.scenarioParamsLOS.K_mu
            ASA=self.scenarioParamsLOS.asa_mu
            ZSA=self.scenarioParamsLOS.zsa_mu
            ASD=self.scenarioParamsLOS.asd_mu
            ZOD=0
            ZSD=self.scenarioParamsLOS.zsdFun(d3D,d2D,hbs,hut)
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
            powC=(1/K+1)*(powPrima/np.sum(powPrima))
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
            Cphi = self.CphiNLOS.get(N)*(1.1035 - 0.028*K - 0.002*(K**2) + 0.0001*(K**3)) 
            auxPhi = []
            auxPhi2 = []
            Y = []
            X = []
            for i in range(N):
                auxPhi.append(((2*(ASA/1.4)*np.sqrt(-np.log(powC[i]/maxP)))/Cphi))
                auxPhi2.append(((2*(ASD/1.4)*np.sqrt(-np.log(powC[i]/maxP)))/Cphi))
                Y.append(np.random.normal(0,(ASA/7)**2))
                X.append(np.random.rand())#tiene que ser entre -1 y 1
            phiAOAprima = np.array(auxPhi)
            phiAODprima = np.array(auxPhi2)
            auxphiAOA = []
            auxphiAOD = []
            for i in range(N):
                auxphiAOA.append((X*phiAOAprima + Y) - (X[1]*phiAOAprima[1] + Y[1])) #- phiLOSAOA
                auxphiAOD.append((X*phiAODprima + Y) - (X[1]*phiAODprima[1] + Y[1])) #-phiLOSAOD
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
                X2.append(np.random.rand())
            tetaZOAprima = np.array(auxtetaZOAprima)
            tetaZODprima = np.array(auxtetaZODprima)
            auxtetaZOA = []
            auxtetaZOD = []
            for i in range(N):
                auxtetaZOA.append((X2[i]*tetaZOAprima[i] + Y1[i]) - (X2[1]*tetaZOAprima[1] + Y1[1])) #-tetaLOSZOA
                auxtetaZOD.append((X2[i]*tetaZODprima[i] + Y2[i] + ZOD) - (X2[1]*tetaZODprima[1] + Y2[1]))
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
            N=self.scenarioParamsNLOS.N
            M=self.scenarioParamsNLOS.M
            rt=self.scenarioParamsNLOS.rt
            DS=self.scenarioParamsNLOS.ds_mu
            K=self.scenarioParamsNLOS.K_mu
            ASA=self.scenarioParamsNLOS.asa_mu
            ZSA=self.scenarioParamsNLOS.zsa_mu
            ASD=self.scenarioParamsNLOS.asd_mu
            ZOD=self.scenarioParamsNLOS.zodFun(d2D,hut)
            ZSD=self.scenarioParamsNLOS.zsdFun(d3D,d2D,hbs,hut)
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
            Cphi = self.CphiNLOS.get(N) 
            auxPhi = []
            auxPhi2 = []
            Y = []
            X = []
            for i in range(N):
                auxPhi.append(((2*(ASA/1.4)*np.sqrt(-np.log(powC[i]/maxP)))/Cphi))
                auxPhi2.append(((2*(ASD/1.4)*np.sqrt(-np.log(powC[i]/maxP)))/Cphi))
                Y.append(np.random.normal(0,(ASA/7)**2))
                X.append(np.random.rand())#tiene que ser entre -1 y 1
            phiAOAprima = np.array(auxPhi)
            phiAODprima = np.array(auxPhi2)
            auxphiAOA = []
            auxphiAOD = []
            for i in range(N):
                auxphiAOA.append((X*phiAOAprima + Y))
                auxphiAOD.append((X*phiAODprima + Y))
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
        for i in range(N):
            Y1.append(np.random.normal(0,(ZSA/7)**2))
            Y2.append(np.random.normal(0,(ZSD/7)**2))
            auxtetaZOA.append((X[i]*tetaZOAprima[i] + Y1[i]))
            auxtetaZOD.append((X[i]*tetaZODprima[i] + Y2[i] + ZOD))
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
            
        
    def create_channel(self,txPos = (0,0,0), rxPos = (0,0,0)):
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
        TgridXIndex= txPos[0] // self.corrDistance
        TgridYIndex= txPos[1] // self.corrDistance 
        RgridXIndex= (rxPos[0]-txPos[0]) // self.corrDistance 
        RgridYIndex= (rxPos[1]-txPos[1]) //self.corrDistance 
        macrokey = (TgridXIndex,TgridYIndex,RgridXIndex,RgridYIndex)
        key = (txPos[0],txPos[1],rxPos[0],rxPos[1])

#        if not self.dMacrosGenerated.has_key(key):
        if not macrokey in self.dMacrosGenerated:
            self.create_macro(txPos,rxPos)
        macro = self.dMacrosGenerated[macrokey]

        bLOS = macro.los
        if bLOS:
            param = self.scenarioParamsLOS
        else:
            param = self.scenarioParamsNLOS
            
        ds = macro.ds
        rt = param.rt
        M=param.M
        N=param.N
        nClusters = param.N
        CphiNLOS = param.CphiNLOS.get(N)
        CthetaNLOS = param.CthetaNLOS.get(N)
        #placeholders
       # pldB= -120
        
        pldB = macro.pLossFun(d3D,d2D,hut)
        
        tauCs= np.random.exponential(scale=ds*rt, size=(nClusters,1))
        tauCs= np.sort(tauCs - np.min(tauCs),axis=0)
        Zn = param.xi*np.random.randn(nClusters,1)
        powCs=np.exp(-tauCs*(rt-1)/(rt*ds))*(10**( (-Zn) /10 ))
        powCs=powCs/np.sum(powCs)
        if (bLOS):
            K=macro.K
            Ctau = 0.7705 - 0.0433*K + 0.0002*(K**2) + 0.000017*(K**3) 
            powCs=powCs/(K+1)
            powCs[0]=powCs[0]+K/(K+1)
            tauCs=tauCs / Ctau
            Cphi = CphiNLOS*( 1.1035 - 0.028*K + 0.002*(K**2) + 0.0001*(K**3) ) #7.5-10
            Ctheta = CthetaNLOS*(  1.3086 - 0.0339*K + 0.0077*(K**2) +  0.0002*(K**3) ) #7.5-15
        else:
            Ctau = 1
            Cphi = CphiNLOS
            Ctheta = CthetaNLOS
        maxP=np.max(powCs)
        tauCs = np.array([tauCs[x] for x in range(0,len(tauCs)) if powCs[x]>maxP*(10**-2.5)])
        powCs = np.array([x for x in powCs if x>maxP*(10**-2.5)])
        nClusters=np.size(powCs)

        #azimut
        mphiCsA = 2* macro.asa/1.4 * np.sqrt(-np.log(powCs/maxP)) / Cphi
        sphiCsA = 2.0*np.random.randint(0,2,(nClusters,1))-1.0
        vphiCsA = macro.asa/7 * np.random.randn(nClusters,1)
        if (bLOS):
            phiCsA = losphiAoA + sphiCsA*mphiCsA + vphiCsA - sphiCsA[0]*mphiCsA[0] - vphiCsA[0]
        else:
            phiCsA = losphiAoA + sphiCsA*mphiCsA + vphiCsA
        mphiCsD = 2* macro.asd/1.4 * np.sqrt(-np.log(powCs/maxP)) / Cphi
        sphiCsD = 2.0*np.random.randint(0,2,(nClusters,1))-1.0
        vphiCsD = macro.asd/7 * np.random.randn(nClusters,1)
        if (bLOS):
            phiCsD = losphiAoD + sphiCsD*mphiCsD + vphiCsD - sphiCsD[0]*mphiCsD[0] - vphiCsD[0]
        else:
            phiCsD = losphiAoD + sphiCsD*mphiCsD + vphiCsD
        #zenith (elevation)
        mthetaCsA = -macro.zsa *np.log(powCs/maxP) / Ctheta
        sthetaCsA = 2.0*np.random.randint(0,2,(nClusters,1))-1.0
        vthetaCsA = macro.zsa/7 * np.random.randn(nClusters,1)
        if (bLOS):
            thetaCsA = losthetaAoA + sthetaCsA*mthetaCsA + vthetaCsA - sthetaCsA[0]*mthetaCsA[0] - vthetaCsA[0]
        else:
            thetaCsA = losthetaAoA + sthetaCsA*mthetaCsA + vthetaCsA + macro.zod_offset_mu
        mthetaCsD = -macro.zsd *np.log(powCs/maxP) / Ctheta
        sthetaCsD = 2.0*np.random.randint(0,2,(nClusters,1))-1.0
        vthetaCsD = macro.zsd/7 * np.random.randn(nClusters,1)
        if (bLOS):
            thetaCsD = losthetaAoD + sthetaCsD*mthetaCsD + vthetaCsD - sthetaCsD[0]*mthetaCsD[0] - vthetaCsD[0]
        else:
            thetaCsD = losthetaAoD + sthetaCsD*mthetaCsD + vthetaCsD
        lp=[]
        if self.bLargeBandwidthOption: #7.6-2
            phiOffsetD=4*np.random.rand(nClusters,M)-2
            phiOffsetA=4*np.random.rand(nClusters,M)-2
            thetaOffsetD=4*np.random.rand(nClusters,M)-2
            thetaOffsetA=4*np.random.rand(nClusters,M)-2
            delayOffset=2*param.cds*np.random.rand(nClusters,M)
            delayOffset=delayOffset-np.min(delayOffset,axis=1).reshape(nClusters,1)
            czsd_aux=(3.0/8.0)*(10.0**macro.zsd_mu)
            powOffset=np.exp(-(delayOffset/param.cds
                                + np.abs(phiOffsetA)*np.sqrt(2)/param.casa
                                + np.abs(phiOffsetD)*np.sqrt(2)/param.casd
                                + np.abs(thetaOffsetA)*np.sqrt(2)/param.czsa
                                + np.abs(thetaOffsetD)*np.sqrt(2)/czsd_aux
                                ))
            powOffset = powOffset / np.sum(powOffset,axis=1).reshape(nClusters,1)
            for nc in range(0,nClusters):
                for indp in range(0,M):
                    phase= 2*np.pi*np.random.rand(1)
                    amp = np.sqrt(powCs[nc]*powOffset[nc,indp])*np.exp(1j*phase)
                    tau=tauCs[nc]*1e9+delayOffset[nc,indp]
                    phiD=phiCsD[nc]+param.casd*phiOffsetD[nc,indp]
                    phiA=phiCsA[nc]+param.casa*phiOffsetA[nc,indp]
                    thetaA=thetaCsA[nc]+param.czsa*thetaOffsetA[nc,indp]
                    if bLOS:
                        thetaD=thetaCsD[nc]+param.czsa*thetaOffsetD[nc,indp]
                    else:
                        czsd_aux=(3.0/8.0)*(10.0**macro.zsd_mu)
                        thetaD=thetaCsD[nc]+czsd_aux*thetaOffsetD[nc,indp]
                    pathInfo = ch.ParametricPath(amp,tau,phiD*np.pi/180.0,phiA*np.pi/180.0,thetaD*np.pi/180.0,thetaA*np.pi/180.0,0)
                    lp.append(pathInfo)
        else:
            alpham=[
                0.0447,
                0.1413,
                0.2492,
                0.3715,
                0.5129,
                0.6797,
                0.8844,
                1.1481,
                1.5195,
                2.1551
            ]
            for nc in range(0,nClusters):
                for indp in range(0,M):
                    phase= 2*np.pi*np.random.rand(1)
                    amp = np.sqrt(powCs[nc]/M)*np.exp(1j*phase)
                    secondMaxP=np.max([x for x in powCs if x < maxP])
                    if powCs[nc]>=secondMaxP:
                        if indp in [9,10,11,12,17,18]:
                            tau=tauCs[nc]*1e9+param.cds*1.28
                        elif indp in [13,14,15,16]:
                            tau=tauCs[nc]*1e9+param.cds*2.56
                        else:
                            tau=tauCs[nc]*1e9
                    else:
                        tau=tauCs[nc]*1e9
                    phiD=phiCsD[nc]+param.casd*alpham[indp//2]*(-1 if (indp%2)==1 else 1)
                    phiA=phiCsA[nc]+param.casa*alpham[indp//2]*(-1 if (indp%2)==1 else 1)
                    thetaA=thetaCsA[nc]+param.czsa*alpham[indp//2]*(-1 if (indp%2)==1 else 1)
                    if bLOS:
                        thetaD=thetaCsD[nc]+param.czsa*alpham[indp//2]*(-1 if (indp%2)==1 else 1)
                    else:
                        czsd_aux=(3.0/8.0)*(10.0**macro.zsd_mu)
                        thetaD=thetaCsD[nc]+czsd_aux*alpham[indp//2]*(-1 if (indp%2)==1 else 1)
                    pathInfo = ch.ParametricPath(amp,tau,phiD*np.pi/180.0,phiA*np.pi/180.0,thetaD*np.pi/180.0,thetaA*np.pi/180.0,0)
                    lp.append(pathInfo)
        self.dChansGenerated[key] = ch.MultipathChannel(txPos,rxPos,lp)
        return(self.dChansGenerated[key])
