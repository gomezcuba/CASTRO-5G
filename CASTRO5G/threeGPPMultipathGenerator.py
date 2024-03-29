import numpy as np
import pandas as pd

class ThreeGPPMultipathChannelModel:
    
    tableFunLOSprob = {
        #functions admit np.array inputs |if   ,then, else    
        "RMa": lambda d2D,hut : np.where(d2D<10, 1  , np.exp(-(d2D-10.0)/1000.0) ),
        "UMi": lambda d2D,hut : np.where(d2D<18, 1  , 18.0/np.where(d2D>0,d2D,1) + np.exp(-d2D/36.0)*(1-18.0/np.where(d2D>0,d2D,1)) ),
        "UMa": lambda d2D,hut : np.where(d2D<18, 1  , (18.0/np.where(d2D>0,d2D,1) + np.exp(-d2D/63.0)*(1-18.0/np.where(d2D>0,d2D,1)))*(1 + np.where(hut<=23, 0, ((hut-13.0)/10.0)**1.5)*1.25*((d2D/100.0)**3.0)*np.exp(-d2D/150.0)) ),
        "InH-Office-Mixed": lambda d2D,hut : np.where( d2D<1.2, 1  , np.where( d2D<6.5, np.exp(-(d2D-1.2)/4.7) , (np.exp(-(d2D-6.5)/32.6))*0.32) ),
        "InH-Office-Open": lambda d2D,hut : np.where(  d2D<=5, 1, np.where( d2D<49, np.exp(-(d2D-5.0)/70.8)  , (np.exp(-(d2D-49.0)/211.7))*0.54) )
    }
    
    def dfTS38900Table756(self,fc):
        df = pd.DataFrame(
            index=[
                    'ds_mu','ds_sg',
                    'asd_mu','asd_sg',
                    'asa_mu','asa_sg',
                    'zsa_mu','zsa_sg',
                    'funZSD_mu','zsd_sg',#ZSD lognormal mu depends on hut and d2D
                    'sf_sg',
                    'K_mu','K_sg',
                    'rt',
                    'N',
                    'M',
                    'cds',
                    'casd',
                    'casa',
                    'czsa',
                    'xi',
                    'Cc', #Correlations Matrix Azimut [sSF, sK, sDS, sASD, sASA, sZSD, sZSA]
                    'funPathLoss',
                    'funZODoffset',
                    'xpr_mu','xpr_sg',
                    'corrO21','corrLOS','corrStatistics'
                ],
            data={
                ('UMi','LOS'): [
                    -0.24*np.log10(1+fc)-7.14, 0.38, #logDS mu sigma
                    -0.05*np.log10(1+fc)+1.21, 0.41, #logASD mu sigma
                    -0.08*np.log10(1+fc)+1.73, 0.014*np.log10(1+fc)+0.28,#logASA mu sigma
                    -0.1*np.log10(1+fc)+0.73, -0.04*np.log10(1+fc)+0.34,#logZSA mu sigma
                    lambda d2D,hut: np.maximum(-0.21, -14.8*(d2D/1000.0) + 0.01*np.abs(hut-10.0) +0.83),
                    0.35,
                    4,
                    9, 5,
                    3,
                    12, 20,
                    5, 3, 17, 7, # cds casd casa czsa
                    3,
                    np.array([[1,0.5,-0.4,-0.5,-0.4,0,0],
                       [0.5,1,-0.7,-0.2,-0.3,0,0],
                       [-0.4,-0.7,1,0.5,0.8,0,0.2],
                       [-0.5,-0.2,0.5,1,0.4,0.5,0.3],
                       [-0.4,-0.3,0.8,0.4,1,0,0],
                       [0,0,0,0.5,0,1,0],
                       [0,0,0.2,0.3,0,0,1]]),
                    self.scenarioPlossUMiLOS,
                    lambda d2D,hut: 0,
                    9, 3, # cross polarization mu sigma
                    50,50,12
                ],
                ('UMi','NLOS'): [
                    -0.24*np.log10(1+fc)-6.83,
                    0.16*np.log10(1+fc)+0.28,
                    -0.23*np.log10(1+fc)+1.53,
                    0.11*np.log10(1+fc)+0.33,
                    -0.08*np.log10(1+fc)+1.81,
                    0.05*np.log10(1+fc)+0.3,
                    -0.04*np.log10(1+fc)+0.92,
                    -0.07*np.log10(1+fc)+0.41,
                    lambda d2D,hut: np.maximum(-0.21, -14.8*(d2D/1000.0) + 0.01*np.abs(hut-10.0) +0.83),
                    0.35,                    
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
                    lambda d2D,hut: -np.power(10.0,-1.5*np.log10(np.maximum(10,d2D)) + 3.3),
                    8,
                    3,
                    50,50,15
                ],
                ('UMa','LOS'): [
                    -6.955 - 0.0963*np.log10(fc),
                    0.66,
                    1.06 + 0.1114*np.log10(fc),
                    0.28, 
                    1.81,
                    0.20,
                    0.95,
                    0.16, 
                    lambda d2D,hut: np.maximum(-0.5, -2.1*(d2D/1000.0) - 0.01*(hut - 1.5) + 0.75),
                    0.40,
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
                    lambda d2D,hut: 0,
                    8,
                    4,
                    50,50,40
                ],
                ('UMa','NLOS'): [
                    -6.28 - 0.204*np.log10(fc),
                    0.39,
                    1.5 - 0.1144*np.log10(fc),
                    0.28,
                    2.08 - 0.27*np.log10(fc),
                    0.11,
                    -0.3236*np.log10(fc) + 1.512,
                    0.16,
                    lambda d2D,hut: np.maximum(-0.5, -2.1*(d2D/1000.0) - 0.01*(hut - 1.5) + 0.9),
                    0.49,
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
                    lambda d2D,hut: 7.66*np.log10(self.frecRefGHz) - 5.96 - np.power(10, (0.208*np.log10(self.frecRefGHz) - 0.782)*np.log10(np.maximum(25.0,d2D)) - 0.13*np.log10(self.frecRefGHz) + 2.03 - 0.07*(hut - 1.5)  ),
                    7,
                    3,
                    50,50,50
                ],
                ('RMa','LOS'): [
                    -7.49,
                    0.55,
                    0.90,
                    0.38,
                    1.52,
                    0.24,
                    0.47,
                    0.40,
                    lambda d2D,hut:  np.maximum(-1, -0.17*(d2D/1000.0) - 0.01*(hut - 1.5) + 0.22),
                    0.34,
                    4, 
                    7,
                    4,
                    3.8,
                    11,
                    20,
                    3.91,
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
                    lambda d2D,hut: 0,
                    12,
                    4,
                    50,60,60
                ],
                ('RMa','NLOS'): [
                    -7.43,
                    0.48,
                    0.95,
                    0.45,
                    1.52,
                    0.13,
                    0.58,
                    0.37,
                    lambda d2D,hut: np.maximum(-1, -0.19*(d2D/1000) - 0.01*(hut - 1.5) + 0.28),
                    0.30,
                    8,
                    0,
                    0,
                    1.7,
                    10,
                    20,
                    3.91,
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
                    lambda d2D,hut: np.arctan((35.0 - 3.5)/d2D) - np.arctan((35.0 - 1.5)/d2D),
                    7,
                    3,
                    50,60,60
                ],                
                ('InH-Office-Mixed','LOS'): [
                    -0.01*np.log10(1+fc) - 7.692,
                    0.18,
                    1.60,
                    0.18,
                    -0.19*np.log10(1+fc) + 1.781,
                    0.12*np.log10(1+fc) + 0.119,
                    -0.26*np.log10(1+fc) + 1.44,
                    -0.04*np.log10(1+fc) + 0.264,
                    lambda d2D,hut: -1.43*np.log10(1 + fc) + 2.228,
                    0.13*np.log10(1 + fc) + 0.30,
                    3,
                    7,
                    4,
                    3.6,
                    15,
                    20,
                    3.91,
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
                   lambda d2D,hut: 0,
                   11,
                   4,
                   0,10,10
                ],                
                ('InH-Office-Mixed','NLOS'): [
                    -0.28*np.log10(1+fc) - 7.173,
                    0.10*np.log10(1+fc) + 0.055,
                    1.62,
                    0.25,
                    -0.11*np.log10(1+fc) + 1.863,
                    0.12*np.log10(1+fc) + 0.059,
                    -0.15*np.log10(1+fc) + 1.387,
                    -0.09*np.log10(1+fc) + 0.746,
                    lambda d2D,hut: 1.08,
                    0.36,
                    8.03,
                    0,
                    0,
                    3,
                    19,
                    20,
                    3.91,
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
                    lambda d2D,hut: 0,
                    10,
                    4,
                    0,10,10
                ],                
                ('InH-Office-Open','LOS'): [
                    -0.01*np.log10(1+fc) - 7.692,
                    0.18,
                    1.60,
                    0.18,
                    -0.19*np.log10(1+fc) + 1.781,
                    0.12*np.log10(1+fc) + 0.119,
                    -0.26*np.log10(1+fc) + 1.44,
                    -0.04*np.log10(1+fc) + 0.264,
                    lambda d2D,hut: -1.43*np.log10(1 + fc) + 2.228,
                    0.13*np.log10(1 + fc) + 0.30,
                    3,
                    7,
                    4,
                    3.6,
                    15,
                    20,
                    3.91,
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
                   lambda d2D,hut: 0,
                   11,
                   4,
                   0,10,10
                ],                
                ('InH-Office-Open','NLOS'): [
                    -0.28*np.log10(1+fc) - 7.173,
                    0.10*np.log10(1+fc) + 0.055,
                    1.62,
                    0.25,
                    -0.11*np.log10(1+fc) + 1.863,
                    0.12*np.log10(1+fc) + 0.059,
                    -0.15*np.log10(1+fc) + 1.387,
                    -0.09*np.log10(1+fc) + 0.746,
                    lambda d2D,hut: 1.08,
                    0.36,
                    8.03,
                    0,
                    0,
                    3,
                    19,
                    20,
                    3.91,
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
                    lambda d2D,hut: 0,
                    10,
                    4,
                    0,10,10
                ]
           })
        return(df)
    
    #Crear tablas
    CphiNLOStable = {4 : 0.779, 5 : 0.860 , 8 : 1.018, 10 : 1.090, 
                11 : 1.123, 12 : 1.146, 14 : 1.190, 15 : 1.211, 16 : 1.226, 
                19 : 1.273, 20 : 1.289, 25 : 1.358}
    CtetaNLOStable = {8 : 0.889, 10 : 0.957, 11 : 1.031, 12 : 1.104, 
                 15 : 1.1088, 19 : 1.184, 20 : 1.178, 25 : 1.282}
    alphamTable = [0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492,
                   0.3715, -0.3715, 0.5129, -0.5129, 0.6797, -0.6797, 
                   0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 
                   2.1551, -2.1551]
    tableSubclusterIndices = [
            [0,1,2,3,4,5,6,7,18,19],
            [8,9,10,11,16,17],
            [12,13,14,15],
        ]
    
    #RMa hasta 7GHz y el resto hasta 100GHz
    def __init__(self, fc = 28, scenario = "UMi", bLargeBandwidthOption=False, avgStreetWidth=20, avgBuildingHeight=5, bandwidth=20e6, arrayWidth=1,arrayHeight=1, maxM=40, funPostprocess = None, smallCorrDist = None):
        self.frecRefGHz = fc
        self.scenario = scenario
        #self.corrDistance = corrDistance
        self.W=avgStreetWidth
        self.h=avgBuildingHeight
        self.bLargeBandwidthOption = bLargeBandwidthOption
        self.B=bandwidth
        self.Dh=arrayWidth
        self.Dv=arrayHeight
        self.maxM=maxM
        self.clight=3e8
        self.wavelength = 3e8/(fc*1e9)
        self.allParamTable = self.dfTS38900Table756(fc)
        
        self.funPostprocess = funPostprocess

        self.scenarioLosProb= self.tableFunLOSprob[self.scenario]
        self.scenarioParams = self.allParamTable[self.scenario]

        self.smallCorrDist = smallCorrDist
        
        self.dMacrosGenerated = pd.DataFrame(columns=[
            'TGridx','TGridy','RGridx','RGridy','LOS',
            'sfdB','ds','asa','asd','zsa','zsd_lslog','K'
         ]).set_index(['TGridx','TGridy','RGridx','RGridy','LOS'])
        self.dChansGenerated = {}
        #TODO modify channels generated dictionary to native pandas dataframe
        # self.dClustersGenerated = pd.DataFrame(columns=[
        #     'Xt','Yt','Zt','Xr','Yr','Zr','n',
        #       'TDOA','P','AOA','AOD','ZOA','ZOD'
        # ]).set_index(['Xt','Yt','Zt','Xr','Yr','Zr','n'])
        # self.dSubpathsGenerated = pd.DataFrame(columns=[
        #     'Xt','Yt','Zt','Xr','Yr','Zr','n','m',
        #       'TDOA','P','AOA','AOD','ZOA','ZOD','XPR','phase00','phase01','phase10','phase11'
        # ]).set_index(['Xt','Yt','Zt','Xr','Yr','Zr','n','m'])
        self.dLOSGenerated = {}
        self.miscasillas = {}

    # TODO introduce code for multi-floor hut in UMi & UMa
    #         if not indoor:
    #     n=1
    # else:
    #     Nf = np.random.uniform(4,8)
    #     n=np.random.uniform(1,Nf)
    # hut_aux = 3*(n-1) + hut
    
    #UMi Path Loss Functions    
    def scenarioPlossUMiLOS(self,d3D,d2D,hbs=10,hut=1.5):     
        prima_dBP = (4*(hbs-1)*(hut-1)*self.frecRefGHz) / self.clight
        ploss = np.where( d2D<prima_dBP,#if
        #then
            32.4 + 21.0*np.log10(d3D)+20.0*np.log10(self.frecRefGHz),
        #else
            32.4 + 40*np.log10(d3D)+20.0*np.log10(self.frecRefGHz)-9.5*np.log10(np.power(prima_dBP,2)+np.power(hbs-hut,2))
            )
        return( ploss )
    def scenarioPlossUMiNLOS(self,d3D,d2D,hbs=10,hut=1.5):
        PL1 = 35.3*np.log10(d3D) + 22.4 + 21.3*np.log10(self.frecRefGHz)-0.3*(hut - 1.5)
        PL2 = self.scenarioPlossUMiLOS(d3D,d2D,hbs,hut) #PLUMi-LOS = Pathloss of UMi-Street Canyon LOS outdoor scenario
        ploss = np.maximum(PL1,PL2)
        return( ploss )
    #UMa Path Loss Functions
    def scenarioPlossUMaLOS(self,d3D,d2D,hbs=25,hut=1.5):
        prima_dBP = (4*(hbs-1)*(hut-1)*self.frecRefGHz) / self.clight
        ploss = np.where( d2D<prima_dBP,#if
        #then
            28.0 + 22.0*np.log10(d3D)+20.0*np.log10(self.frecRefGHz),
        #else
            28.0 + 40.0*np.log10(d3D)+20.0*np.log10(self.frecRefGHz)-9.0*np.log10(np.power(prima_dBP,2)+np.power(hbs-hut,2))
            )
        return(ploss)
    def scenarioPlossUMaNLOS(self,d3D,d2D,hbs=25,hut=1.5):
        PL1 = 13.54 + 39.08*np.log10(d3D) + 20.0*np.log10(self.frecRefGHz) - 0.6*(hut - 1.5)
        PL2 = self.scenarioPlossUMaLOS(d3D,d2D,hbs,hut)
        ploss = np.maximum(PL1,PL2)
        return(ploss) 
    #RMa Path Loss Functions
    def scenarioPlossRMaLOS(self,d3D,d2D,hbs=35,hut=1.5):
        dBp = (2*np.pi*hbs*hut)*(self.frecRefGHz*1e9/self.clight) #Break point distance
        ploss = np.where( d2D<dBp,#if
        #then
             20*np.log10(40.0*np.pi*d3D*self.frecRefGHz/3.0)+np.minimum(0.03*np.power(self.h,1.72),10)*np.log10(d3D)-np.minimum(0.044*np.power(self.h,1.72),14.77)+0.002*np.log10(self.h)*d3D,
        #else
            20*np.log10(40.0*np.pi*dBp*self.frecRefGHz/3.0)+np.minimum(0.03*np.power(self.h,1.72),10)*np.log10(dBp)-np.minimum(0.044*np.power(self.h,1.72),14.77)+0.002*np.log10(self.h)*dBp + 40.0*np.log10(d3D/dBp)
            )
        return(ploss)
    def scenarioPlossRMaNLOS(self,d3D,d2D,hbs=25,hut=1.5):
        PL1= 161.04 - (7.1*np.log10(self.W)) + 7.5*(np.log10(self.h)) - (24.37 - 3.7*(np.power((self.h/hbs),2)))*np.log10(hbs) + (43.42 - 3.1*(np.log10(hbs)))*(np.log10(d3D)-3) + 20*np.log10(3.55) - (3.2*np.power(np.log10(11.75*hut),2)) - 4.97
        PL2= self.scenarioPlossRMaLOS(d3D,d2D,hbs,hut)
        ploss = np.maximum(PL1,PL2)
        return(ploss)
    #Inh Path Loss Functions
    def scenarioPlossInLOS(self,d3D,d2D,hbs=None,hut=None):
        ploss= 32.4 + 17.3*np.log10(d3D) + 20.0*np.log10(self.frecRefGHz)
        return(ploss)
    def scenarioPlossInNLOS(self,d3D,d2D,hbs=None,hut=None):
        PL1= 38.3*np.log10(d3D) + 17.30 + 24.9*np.log10(self.frecRefGHz)
        PL2= self.scenarioPlossInLOS(d3D,d2D,hbs,hut)
        ploss= np.maximum(PL1,PL2)
        return(ploss)  
       
    #macro => Large Scale Correlated parameters
    def calculateGridCoeffs(self,txPos, rxPos,Dcorr):
        XgridtIndex= (txPos[0] + Dcorr/2) // Dcorr
        YgridtIndex= (txPos[1] + Dcorr/2) // Dcorr
        XgridrIndex= (rxPos[0]-txPos[0] + Dcorr/2) // Dcorr
        YgridrIndex= (rxPos[1]-txPos[1] + Dcorr/2) // Dcorr
        return(XgridtIndex,YgridtIndex,XgridrIndex,YgridrIndex)
        
    #hidden uniform variable to compare with pLOS(distabce)
    def get_LOSUnif_from_location(self,txPos, rxPos):
        dCorr = self.scenarioParams.LOS.corrLOS
        key =  self.calculateGridCoeffs(txPos,rxPos,dCorr)
        if not key in self.dLOSGenerated:
           self.dLOSGenerated[key] = np.random.rand(1)[0]
        return(self.dLOSGenerated[key])
        
    #macro => Large Scale Correlated parameters
    def get_macro_from_location(self,txPos, rxPos,los):
        if los:
            dCorr = self.scenarioParams.LOS.corrStatistics
        else:
            dCorr = self.scenarioParams.NLOS.corrStatistics
        TgridXIndex,TgridYIndex,RgridXIndex,RgridYIndex= self.calculateGridCoeffs(txPos,rxPos, dCorr) #ray corr distance
        macrokey = (TgridXIndex,TgridYIndex,RgridXIndex,RgridYIndex,los)
        if not macrokey in self.dMacrosGenerated.index:
            return(self.create_macro(macrokey))#saves result to memory
        else:
            return(self.dMacrosGenerated.loc[macrokey,:])
        
    def create_macro(self, macrokey):
        los=macrokey[4]
        vIndep = np.random.randn(7,1)
        if los:
            param = self.scenarioParams.LOS
        else:
            param = self.scenarioParams.NLOS
        L=np.linalg.cholesky(param.Cc)
        vDep=(L@vIndep).reshape(-1)
        sfdB = param.sf_sg*vDep[0] #due to cholesky decomp this is independent
        K= np.power(10.0,  (param.K_mu + param.K_sg * vDep[1])/10)
        ds = np.power(10.0, param.ds_mu + param.ds_sg * vDep[2])
        asd = min( np.power(10.0, param.asd_mu + param.asd_sg * vDep[3] ), 104.0)
        asa = min( np.power(10.0, param.asa_mu + param.asa_sg * vDep[4] ), 104.0)
        zsd_lslog = param.zsa_sg * vDep[6]
        zsa = min( np.power(10.0, param.zsa_mu + param.zsa_sg * vDep[6] ), 52.0)
        self.dMacrosGenerated.loc[macrokey,:]=(sfdB,ds,asa,asd,zsa,zsd_lslog,K)
        return(self.dMacrosGenerated.loc[macrokey,:])
    
    #clusters => small scale groups of pahts
    def create_clusters(self,smallStatistics,LOSangles):
        los,DS,ASA,ASD,ZSA,ZSD,K,czsd,muZOD = smallStatistics
        
        if los:
            param = self.scenarioParams.LOS
        else:
            param = self.scenarioParams.NLOS
        N = param.N
        rt = param.rt
        
        (losAoD,losAoA,losZoD,losZoA) = LOSangles
        
        #Generate cluster delays
        tau_prima = -rt*DS*np.log(np.random.rand(N))
        tau = np.sort( tau_prima-np.min(tau_prima) )
        
        #Generate cluster powers
        powPrima = np.exp( -tau * (rt-1) / (rt*DS) ) * np.power( 10, -np.random.normal(0,param.xi,size=N)/10 )
        #The scaled delays are NOT to be used in cluster power generation.
        Ctau = 0.7705 - 0.0433*K + 0.0002*K**2 + 0.000017*K**3
        if los:
            tau = tau / Ctau 
        if los:
            powC = ( 1/(K+1) ) * ( powPrima/np.sum(powPrima) )
            powC[0] = powC[0] + K / (K+1)
        else:
            powC = powPrima/np.sum(powPrima)
        #Remove clusters with less than -25 dB power compared to the maximum cluster power. The scaling factors need not be 
        #changed after cluster elimination
        #TODO confirm whether htis should go before or after LOS conversion
        maxP=np.max(powC)
        tau = tau[powC>(maxP*(10**(-2.5)))]
        powC = powC[powC>(maxP*(10**(-2.5)))]        
        nClusters=powC.size
        #Generate arrival angles and departure angles for both azimuth and elevation   
        #Azimut
        if los:
            Cphi = self.CphiNLOStable[N]*(1.1035 - 0.028*K - 0.002*(K**2) + 0.0001*(K**3))
        else:
            Cphi = self.CphiNLOStable[N]
        phiAOAprima = 2*(ASA/1.4)*np.sqrt(-np.log(powC/maxP))/Cphi
        phiAODprima = 2*(ASD/1.4)*np.sqrt(-np.log(powC/maxP))/Cphi
        
        Xaoa = np.random.choice((-1,1),size=powC.shape)
        Xaod = np.random.choice((-1,1),size=powC.shape)
        Yaoa = np.random.normal(0,ASA/7,size=powC.shape)
        Yaod = np.random.normal(0,ASD/7,size=powC.shape)
        AOA = Xaoa*phiAOAprima + Yaoa + losAoA - (Xaoa[0]*phiAOAprima[0] + Yaoa[0] if los==1 else 0)
        AOD = Xaod*phiAODprima + Yaod + losAoD - (Xaod[0]*phiAODprima[0] + Yaod[0] if los==1 else 0)
        
        #Zenith 
        if los:
            Cteta = self.CtetaNLOStable.get(N)*(1.3086 + 0.0339*K -0.0077*(K**2) + 0.0002*(K**3))
        else:
            Cteta = Cteta = self.CtetaNLOStable.get(N)
        
        tetaZOAprima = -((ZSA*np.log(powC/maxP))/Cteta)
        tetaZODprima = -((ZSD*np.log(powC/maxP))/Cteta)
        
        Xzoa = np.random.choice((-1,1),size=powC.shape)
        Xzod = np.random.choice((-1,1),size=powC.shape)
        Yzoa = np.random.normal(0, ZSA/7 ,size=powC.shape)
        Yzod = np.random.normal(0, ZSD/7 ,size=powC.shape)
        ZOA = Xzoa*tetaZOAprima + Yzoa + losZoA - (Xzoa[0]*tetaZOAprima[0] + Yzoa[0] if (los==1) else 0)
        ZOD = Xzod*tetaZODprima + Yzod + losZoD + muZOD - (Xzod[0]*tetaZODprima[0] + Yzod[0] + muZOD if (los==1) else 0)
          
        return( pd.DataFrame(columns=['TDOA','P','AOA','AOD','ZOA','ZOD'],
                             data=np.array([tau,powC,AOA,AOD,ZOA,ZOD]).T,
                             index=pd.Index(np.arange(nClusters),name='n')) )
       
    def create_subpaths_basics(self,smallStatistics,clusters,LOSangles):
        los,DS,ASA,ASD,ZSA,ZSD,K,czsd,muZOD = smallStatistics
        (losAoD,losAoA,losZoD,losZoA) = LOSangles
        (tau,powC,AOA,AOD,ZOA,ZOD)=clusters.T.to_numpy()
        nClusters=tau.size   
        if los:
            param = self.scenarioParams.LOS
            powC_nlos= powC*(1+K)
            powC_nlos[0]=powC_nlos[0]-K
        else:
            param = self.scenarioParams.NLOS
            powC_nlos= powC 
        M=param.M
        
        indStrongestClusters = np.argsort(-powC)[0:2]
        
        #Generate subpaths delays and powers
        pow_sp=np.tile(powC_nlos[:,None],(1,M))/M
        tau_sp=np.tile(tau[:,None],(1,M))
        #the two first clusters are divided in 3 subclusters
        for n in indStrongestClusters:
            tau_sp[n,self.tableSubclusterIndices[1]] += 1.28*param.cds*1e-9
            tau_sp[n,self.tableSubclusterIndices[2]] += 2.56*param.cds*1e-9
        
        AOA_sp = np.tile(AOA[:,None],(1,M)) + param.casa*np.tile(self.alphamTable,(nClusters,1)) #* np.random.choice([1, -1],size=(nClusters,M))
        AOD_sp = np.tile(AOD[:,None],(1,M)) + param.casd*np.tile(self.alphamTable,(nClusters,1)) #* np.random.choice([1, -1],size=(nClusters,M))
       
        ZOA_sp = np.tile(ZOA[:,None],(1,M)) + param.czsa*np.tile(self.alphamTable,(nClusters,1)) #* np.random.choice([1, -1],size=(nClusters,M))
        ZOD_sp = np.tile(ZOD[:,None],(1,M)) + czsd*np.tile(self.alphamTable,(nClusters,1)) #* np.random.choice([1, -1],size=(nClusters,M))
    
        for n in range(nClusters):
            if n not in indStrongestClusters:
                #couple randomly AOD/ZOD/ZOA angles to AOA angles within cluster n
                AOD_sp[n,:]=np.random.permutation(AOD_sp[n,:])
                ZOA_sp[n,:]=np.random.permutation(ZOA_sp[n,:])
                ZOD_sp[n,:]=np.random.permutation(ZOD_sp[n,:])
            else:#clusters 1 and 2
                for scl in range(3):#c
                    AOD_sp[n,self.tableSubclusterIndices[scl]]=np.random.permutation(AOD_sp[n,self.tableSubclusterIndices[scl]])
                    ZOA_sp[n,self.tableSubclusterIndices[scl]]=np.random.permutation(ZOA_sp[n,self.tableSubclusterIndices[scl]])
                    ZOD_sp[n,self.tableSubclusterIndices[scl]]=np.random.permutation(ZOD_sp[n,self.tableSubclusterIndices[scl]])
            
        #mask = (ZOA_sp>=180) & (ZOA_sp<=360)
        #ZOA_sp[mask] = 360 - ZOA_sp             
    
        # Generate the cross polarization power ratios
        xpr_mu = param.xpr_mu
        xpr_sg = param.xpr_sg
        X = np.random.normal(xpr_mu,xpr_sg,size=tau_sp.shape)
        XPR_sp =  10**(X/10)
        
        # Generate the initial phase
        phase = np.random.uniform(-np.pi,np.pi,size=(4,nClusters,M))        
        
        subpaths = pd.DataFrame(
            columns=['TDOA','P','AOA','AOD','ZOA','ZOD','XPR','phase00','phase01','phase10','phase11'],
            data=np.vstack([
                tau_sp.reshape(-1),
                pow_sp.reshape(-1),
                AOA_sp.reshape(-1),
                AOD_sp.reshape(-1),
                ZOA_sp.reshape(-1),
                ZOD_sp.reshape(-1),
                XPR_sp.reshape(-1),
                phase[0,:,:].reshape(-1),
                phase[1,:,:].reshape(-1),
                phase[2,:,:].reshape(-1),
                phase[3,:,:].reshape(-1)
                ]).T,
            index=pd.MultiIndex.from_product([np.arange(nClusters),np.arange(M)],names=['n','m'])
            )
        if los:
            subpaths.P[:]=subpaths.P[:]/(K+1)
            #the LOS ray is the M+1-th subpath of the first cluster
            subpaths.loc[(0,M),:]= (tau[0],K/(K+1),losAoA,losAoD,losZoA,losZoD,0,0,0,0,0)
        
        return(subpaths)
   
    
    def create_subpaths_largeBW(self,smallStatistics,clusters,LOSangles,d2D,hut):
        los,DS,ASA,ASD,ZSA,ZSD,K,czsd,muZOD = smallStatistics
        (losAoD,losAoA,losZoD,losZoA) = LOSangles
        (tau,powC,AOA,AOD,ZOA,ZOD)=clusters.T.to_numpy()
        nClusters=tau.size      
        if los:
            param = self.scenarioParams.LOS
            powC_nlos= powC*(1+K)
            powC_nlos[0]=powC_nlos[0]-K
        else:
            param = self.scenarioParams.NLOS
            powC_nlos= powC 
            
        cds = param.cds
        casd = param.casd
        casa = param.casa
        czsa = param.czsa
        #The number of rays per cluster, replacing param.M
        k = 0.5#sparsity coefficient
        M_t = np.ceil(4*k*cds*self.B)
        M_AOD = np.ceil(4*k*casd*((np.pi*self.Dh)/(180*self.wavelength)))
        M_ZOD = np.ceil(4*k*czsd*((np.pi*self.Dv)/(180*self.wavelength)))
        M = int(np.minimum( np.maximum(M_t*M_AOD*M_ZOD, param.M ) ,self.maxM))
        #The offset angles alpha_m
        alpha_AOA = np.random.uniform(-2,2,size=(nClusters,M))
        alpha_AOD = np.random.uniform(-2,2,size=(nClusters,M))
        alpha_ZOA = np.random.uniform(-2,2,size=(nClusters,M))
        alpha_ZOD = np.random.uniform(-2,2,size=(nClusters,M))
        
        #The relative delay of m-th ray
        tau_primaprima = np.random.uniform(0,2*cds*1e-9,size=(nClusters,M))#ns
        tau_prima = np.sort(tau_primaprima-np.min(tau_primaprima,axis=1,keepdims=True))
        tau_sp = tau_prima + tau.reshape((-1,1))
        
        #Ray powers
        pow_prima = np.exp(-tau_prima/cds)*np.exp(-(np.sqrt(2)*abs(alpha_AOA))/casa)*np.exp(-(np.sqrt(2)*abs(alpha_AOD))/casd)*np.exp(-(np.sqrt(2)*abs(alpha_ZOA))/czsa)*np.exp(-(np.sqrt(2)*abs(alpha_ZOD))/czsd)
        pow_sp = powC_nlos.reshape(-1,1)*(pow_prima/np.sum(pow_prima,axis=1,keepdims=True))
                
        AOA_sp = np.tile(AOA[:,None],(1,M)) + casa*alpha_AOA
        AOD_sp = np.tile(AOD[:,None],(1,M)) + casd*alpha_AOD
       
        ZOA_sp = np.tile(ZOA[:,None],(1,M)) + czsa*alpha_ZOA
        ZOD_sp = np.tile(ZOD[:,None],(1,M)) + czsd*alpha_ZOD
        
        # Generate the cross polarization power ratios
        xpr_mu = param.xpr_mu
        xpr_sg = param.xpr_sg
        X = np.random.normal(xpr_mu,xpr_sg,size=tau_sp.shape)
        XPR_sp =  10**(X/10)
        
        # Generate the initial phase
        phase = np.random.uniform(-np.pi,np.pi,size=(4,nClusters,M))        
        
        subpaths = pd.DataFrame(
            columns=['TDOA','P','AOA','AOD','ZOA','ZOD','XPR','phase00','phase01','phase10','phase11'],
            data=np.vstack([
                tau_sp.reshape(-1),
                pow_sp.reshape(-1),
                AOA_sp.reshape(-1),
                AOD_sp.reshape(-1),
                ZOA_sp.reshape(-1),
                ZOD_sp.reshape(-1),
                XPR_sp.reshape(-1),
                phase[0,:,:].reshape(-1),
                phase[1,:,:].reshape(-1),
                phase[2,:,:].reshape(-1),
                phase[3,:,:].reshape(-1)
                ]).T,
            index=pd.MultiIndex.from_product([np.arange(nClusters),np.arange(M)],names=['n','m'])
            )
        if los:
            subpaths.P[:]=subpaths.P[:]/(K+1)
            #the LOS ray is the M+1-th subpath of the first cluster
            subpaths.loc[(0,M),:]= (tau[0],K/(K+1),losAoA,losAoD,losZoA,losZoD,0,0,0,0,0)
        
        return(subpaths)
    
    
    def create_small_param(self,LOSangles,smallStatistics,d2D,hut):
        los,DS,ASA,ASD,ZSA,ZSD,K,cZSD,muZOD = smallStatistics
        
        clusters = self.create_clusters(smallStatistics,LOSangles)
        
        if self.bLargeBandwidthOption:
            subpaths = self.create_subpaths_largeBW(smallStatistics,clusters,LOSangles,d2D,hut)
        else:
            subpaths = self.create_subpaths_basics(smallStatistics,clusters,LOSangles)
    
        return(clusters,subpaths)
    	
    ####################################################
    #Parte de consistencia espacial

    
    def displaceMultipathChannel(self, dataframe, origTxPos, origRxPos, destTxPos, destRxPos):
        # Código que cumple con el procedimiento A del apartado de consistencia espacial 
        #Parámetros para actualizar tau
        c = 3e8
        tau = dataframe['TDOA'].T.to_numpy()
        zoa = dataframe['ZOA'].T.to_numpy()*np.pi/180
        aoa = dataframe['AOA'].T.to_numpy()*np.pi/180
        zod = dataframe['ZOD'].T.to_numpy()*np.pi/180
        aod = dataframe['AOD'].T.to_numpy()*np.pi/180
        #ensure numpy format
        origTxPos = np.array(origTxPos)
        origRxPos = np.array(origRxPos)
        origTxPos = np.array(origTxPos)
        origRxPos = np.array(origRxPos)
        
        deltaTxPos = destTxPos - origTxPos
        deltaRxPos = destRxPos - origRxPos
        deltaNRxPos = deltaRxPos - deltaTxPos
        deltaNTxPos = deltaTxPos - deltaRxPos
        d3D = np.linalg.norm(origRxPos - origTxPos)
        
        ###############################TAU#########################

        Rrx = np.array([[np.sin(zoa) * np.cos(aoa)],
                        [np.sin(zoa) * np.sin(aoa)],
                        [np.cos(zoa)]]).transpose(2,1,0)
        Rtx = np.array([[np.sin(zod) * np.cos(aod)],
                        [np.sin(zod) * np.sin(aod)],
                        [np.cos(zod)]]).transpose(2,1,0)

    
        deltaRxPos = np.reshape(deltaRxPos, (3, 1)) 
        deltaTxPos = np.reshape(deltaTxPos, (3, 1)) 
        Rrx_array = np.array(Rrx)
        Rtx_array = np.array(Rtx)

        tau_tilde_previo = tau + d3D/c

        auxtau= Rrx_array @ deltaRxPos + Rtx_array @ deltaTxPos
        auxtau2= auxtau / c
        auxtau2 = np.squeeze(auxtau2)
        tau_tilde = tau_tilde_previo - auxtau2

        tau_nueva = tau_tilde-np.min( tau_tilde) 
     

        ############################AOD#####################3
        phiAoD = np.array([[-np.sin(aod)],
                        [np.cos(aod)],
                        [np.zeros_like(aod)]]).transpose(2,0,1)
       
        auxAoD1 = deltaNRxPos.T @ phiAoD
        auxAoD2 = auxAoD1.squeeze()
        auxAoD3 = auxAoD2 /(c*tau_tilde_previo * np.sin(zod))
        aodnueva= auxAoD3 + aod

        ###############################ZOD######################3
        tetaZoD = np.array([[np.cos(zod) * np.cos(aod)],
                        [np.cos(zod) * np.sin(aod)],
                        [-np.sin(aod)]]).transpose(2,0,1)
       
        auxZoD1 = deltaNRxPos.T @ tetaZoD
        auxZoD2 = auxZoD1.squeeze()
        auxZoD3= auxZoD2 / (c*tau_tilde_previo)
        zodnueva = zod + auxZoD3

        ##########################AOA###################

        phiAoA = np.array([[-np.sin(aoa)],
                        [np.cos(aoa)],
                        [np.zeros_like(aoa)]]).transpose(2,0,1)
      
        auxAoA1 =deltaNTxPos.T @ phiAoA
        auxAoA1 = auxAoA1.squeeze()
        auxAoA2 = auxAoA1 /(c*tau_tilde_previo *np.sin(zoa))
        aoanueva = aoa + auxAoA2

        ##################################ZOA#########################
        tetaZoA = np.array([[np.cos(zoa) * np.cos(aoa)],
                        [np.cos(zoa) * np.sin(aoa)],
                        [-np.sin(aoa)]]).transpose(2,0,1)
    
        auxZoA1 = deltaNTxPos.T @ tetaZoA
        auxZoA2= auxZoA1.squeeze()
        auxZoA3= auxZoA2 / (c*tau_tilde_previo)
        zoanueva = zoa + auxZoA3
        dfsalida = dataframe.copy()
        dfsalida['TDOA']= tau_nueva
        dfsalida['AOA'] = aoanueva*180/np.pi
        dfsalida['AOD'] = aodnueva*180/np.pi
        dfsalida['ZOA'] = zoanueva*180/np.pi
        dfsalida['ZOD'] = zodnueva*180/np.pi

        return (dfsalida)
            
    
    def getLOSmeasurements(self,txPos,rxPos):
        vLOS = np.array(rxPos)-np.array(txPos)#in case tuples are given      
        d2D = np.linalg.norm(vLOS[0:2])#works in any space of dim>=2
        d3D = np.linalg.norm(vLOS)
        hbs = txPos[2]
        hut = rxPos[2]
        losAoD=np.mod( np.arctan2( vLOS[1] , vLOS[0] ), 2*np.pi )
        losAoA=np.mod( np.pi +losAoD , 2*np.pi ) # reversed angle
        losZoD=np.mod(np.pi/2-np.arctan2( vLOS[2] , d2D ), 2*np.pi )
        losZoA=np.mod(np.pi-losZoD , 2*np.pi )# reversed angle        
        #3GPP model is in degrees but numpy uses radians
        LOSangles = ((180.0/np.pi)*losAoD,(180.0/np.pi)*losAoA,(180.0/np.pi)*losZoD,(180.0/np.pi)*losZoA)
        return(vLOS,d2D,d3D,hbs,hut,LOSangles)
        
    def create_channel(self, txPos, rxPos):
        vLOS,d2D,d3D,hbs,hut,LOSangles = self.getLOSmeasurements(txPos,rxPos)
                
        pLos=self.scenarioLosProb(d2D,hut)
        los = ( self.get_LOSUnif_from_location(txPos, rxPos) <= pLos)#TODO: make this memorized
        
        if los:
            param = self.scenarioParams.LOS            
        else:
            param = self.scenarioParams.NLOS
        PLconst = param.funPathLoss(d3D,d2D,hbs,hut)        
        if isinstance(PLconst, np.ndarray): #for safety remove numpy array properties of scalar pathloss return value
            PLconst = float(PLconst)
        # PL = PLconst + sf
        
        macro = self.get_macro_from_location(txPos, rxPos,los)
        
        sfdB,ds,asa,asd,zsa,zsd_lslog,K =macro            
      
        plinfo = (los,PLconst,sfdB)

        clusters,subpaths = self.get_small_from_location(txPos,rxPos,plinfo,macro)
        
        return(plinfo,macro,clusters,subpaths)
    
    def getRefLocationParams(self,txPos,rxPos,corrDist):
        key = self.calculateGridCoeffs(txPos,rxPos,corrDist)
        txPosGrid = np.array(txPos)
        txPosGrid[0:2] = np.array(key[0:2]) * corrDist
        rxPosGrid = np.array(rxPos)
        rxPosGrid[0:2] = np.array(key[2:4]) * corrDist
        return(key,txPosGrid,rxPosGrid)
        
    def get_small_from_location(self, txPos, rxPos, plinfo, macro):

        if self.smallCorrDist:
            key,genTxPos,genRxPos = self.getRefLocationParams(txPos,rxPos,self.smallCorrDist)
        else:
            key=tuple(txPos)+tuple(rxPos)
            genTxPos = np.array(txPos)
            genRxPos = np.array(rxPos)
    
        if key in self.dChansGenerated:
            clusters, subpaths = self.dChansGenerated[key]
        else:            
            vLOS,d2D,d3D,hbs,hut,LOSangles = self.getLOSmeasurements(genTxPos,genRxPos)
            los,PLconst,sfdb = plinfo
            if los:
                param = self.scenarioParams.LOS            
            else:
                param = self.scenarioParams.NLOS
            sfdB,ds,asa,asd,zsa,zsd_lslog,K =macro            
            zsd_mu = param.funZSD_mu(d2D,hut)          
            zsd = min( np.power(10.0,zsd_mu + zsd_lslog ), 52.0)
            zod_offset_mu = param.funZODoffset(d2D,hut)        
            czsd = (3/8)*(10**zsd_mu)#intra-cluster ZSD
            smallStatistics = (los,ds,asa,asd,zsa,zsd,K,czsd,zod_offset_mu)
            
            clusters, subpaths = self.create_small_param(LOSangles, smallStatistics, d2D, hut)
            if self.funPostprocess:
                plinfo,clusters,subpaths = self.funPostprocess(txPos,rxPos,plinfo,clusters,subpaths)
                # self.dClustersGenerated = self.dClustersGenerated.append(pd.concat({ keyChannel: clusters },names=['Xt','Yt','Zt','Xr','Yr','Zr']))
    			# self.dSubpathsGenerated = self.dSubpathsGenerated.append(pd.concat({ keyChannel: subpaths },names=['Xt','Yt','Zt','Xr','Yr','Zr']))
			
            self.dChansGenerated[key] = (clusters,subpaths)
        
        if self.smallCorrDist:
            clusters = self.displaceMultipathChannel(clusters, genTxPos, genRxPos, txPos, rxPos)
            subpaths = self.displaceMultipathChannel(subpaths, genTxPos, genRxPos, txPos, rxPos)
        
        return clusters,subpaths

    
    ###########################################################################
    # premade post-processing functions
    # they always must receive (txPos,rxPos,plinfo,clusters,subpaths) as input 
    # and return the same as output, enabling their infinite pipelining
    ###########################################################################    
       
    def fullFitAOA(self,txPos,rxPos,plinfo,clusters,subpaths,mode3D=True):
        tauOffset = self.fixExcessDelayNLOS(txPos,rxPos,plinfo,clusters,mode3D)
        if mode3D:
            aoa_new,zoa_new,locs = self.fitAOA(txPos,rxPos,clusters.TDOA+ tauOffset,clusters.AOD.to_numpy()*np.pi/180,clusters.ZOD.to_numpy()*np.pi/180)            
            clusters.ZOA=zoa_new*180/np.pi
            clusters['Zs']=locs[2,:]
        else:
            aoa_new,locs = self.fitAOA(txPos,rxPos,clusters.TDOA.to_numpy()+tauOffset,clusters.AOD.to_numpy()*np.pi/180)
        clusters.AOA=aoa_new*180/np.pi
        clusters['Xs']=locs[0,:]
        clusters['Ys']=locs[1,:]
        if mode3D:
            aoa_new,zoa_new,locs = self.fitAOA(txPos,rxPos,subpaths.TDOA+ tauOffset,subpaths.AOD.to_numpy()*np.pi/180,subpaths.ZOD.to_numpy()*np.pi/180)            
            subpaths.ZOA=zoa_new*180/np.pi
            subpaths['Zs']=locs[2,:]
        else:
            aoa_new,locs = self.fitAOA(txPos,rxPos,subpaths.TDOA.to_numpy()+tauOffset,subpaths.AOD.to_numpy()*np.pi/180)
        subpaths.AOA=aoa_new*180/np.pi
        subpaths['Xs']=locs[0,:]
        subpaths['Ys']=locs[1,:]
        return (txPos,rxPos,plinfo,clusters,subpaths)
    
    def fullFitAOD(self,txPos,rxPos,plinfo,clusters,subpaths,mode3D=True):
        tauOffset = self.fixExcessDelayNLOS(txPos,rxPos,plinfo,clusters,mode3D)
        if mode3D:
            aod_new,zod_new,locs = self.fitAOD(txPos,rxPos,clusters.TDOA+ tauOffset,clusters.AOA.to_numpy()*np.pi/180,clusters.ZOA.to_numpy()*np.pi/180)            
            clusters.ZOD=zod_new*180/np.pi
            clusters['Zs']=locs[2,:]
        else:
            aod_new,locs = self.fitAOD(txPos,rxPos,clusters.TDOA.to_numpy()+tauOffset,clusters.AOA.to_numpy()*np.pi/180)
        clusters.AOD=aod_new*180/np.pi
        clusters['Xs']=locs[0,:]
        clusters['Ys']=locs[1,:]
        if mode3D:
            aod_new,zod_new,locs = self.fitAOD(txPos,rxPos,subpaths.TDOA+ tauOffset,subpaths.AOA.to_numpy()*np.pi/180,subpaths.ZOA.to_numpy()*np.pi/180)            
            subpaths.ZOD=zod_new*180/np.pi
            subpaths['Zs']=locs[2,:]
        else:
            aod_new,locs = self.fitAOD(txPos,rxPos,subpaths.TDOA.to_numpy()+tauOffset,subpaths.AOA.to_numpy()*np.pi/180)
        subpaths.AOD=aod_new*180/np.pi
        subpaths['Xs']=locs[0,:]
        subpaths['Ys']=locs[1,:]
        return (txPos,rxPos,plinfo,clusters,subpaths)
    
    def attemptFullFitTDOA(self,txPos,rxPos,plinfo,clusters,subpaths,mode3D=True,fallbackFun=None):
        confRelax3D = ( fallbackFun == "relax3D" )
        if mode3D:
            tau_new,locs,valid = self.fitTDOA(txPos,rxPos,clusters.AOA.to_numpy()*np.pi/180,clusters.AOD.to_numpy()*np.pi/180,clusters.ZOA.to_numpy()*np.pi/180,clusters.ZOD.to_numpy()*np.pi/180,relax3D=confRelax3D)
            clusters['Zs']=locs[2,:]
        else:
            tau_new,locs,valid = self.fitTDOA(txPos,rxPos,clusters.AOA.to_numpy()*np.pi/180,clusters.AOD.to_numpy()*np.pi/180,relax3D=False)
        clusters.TDOA[valid]=tau_new[valid]
        clusters['Xs']=locs[0,:]#non valids are inf
        clusters['Ys']=locs[1,:]#non valids are inf        
        if mode3D:
            tau_new,locs,valid = self.fitTDOA(txPos,rxPos,subpaths.AOA.to_numpy()*np.pi/180,subpaths.AOD.to_numpy()*np.pi/180,subpaths.ZOA.to_numpy()*np.pi/180,subpaths.ZOD.to_numpy()*np.pi/180,relax3D=confRelax3D)
            subpaths['Zs']=locs[2,:]
        else:
            tau_new,locs,valid = self.fitTDOA(txPos,rxPos,subpaths.AOA.to_numpy()*np.pi/180,subpaths.AOD.to_numpy()*np.pi/180,relax3D=False)
        subpaths.TDOA[valid]=tau_new[valid]
        subpaths['Xs']=locs[0,:]#non valids are inf
        subpaths['Ys']=locs[1,:]#non valids are inf
        #TODO if fallbackFun:
            
        return (txPos,rxPos,plinfo,clusters,subpaths)
    
    def randomFitAllSubpaths(self, txPos, rxPos,plinfo,clusters,subpaths,P=np.full(4,.25),mode3D=True):
        #P=[Pnot,Paoa,Paod,Ptoa]  
        P=np.array(P)#safety to permit tuple, list and dataframe as input
        transformIndexClusters=self.chooseRandomTransform(txPos, rxPos, clusters, P,mode3D)
        (indexNot,indexAoA,indexAoD,indexToA)= self.getIndicesSubgroups(clusters,transformIndexClusters,maxG=4)
        tauOffset = self.fixExcessDelayNLOS(txPos,rxPos,plinfo,clusters)
        clusters=self.applyMultiTransform(txPos,rxPos,clusters, tauOffset, indexAoA, indexAoD, indexToA,mode3D)
        
        transformIndexClusters=self.chooseRandomTransform(txPos, rxPos, subpaths, P,mode3D)        
        (indexNot,indexAoA,indexAoD,indexToA)= self.getIndicesSubgroups(subpaths,transformIndexClusters,maxG=4)
        subpaths=self.applyMultiTransform(txPos,rxPos,subpaths, tauOffset, indexAoA, indexAoD, indexToA,mode3D)
        
        return(txPos,rxPos,plinfo,clusters,subpaths)    
            
    def randomFitEpctClusters(self, txPos, rxPos,plinfo,clusters,subpaths,Ec=.75,Es=.75,P=np.full(4,.25),mode3D=True,skipLOS=True):
        #P=[Pnot,Paoa,Paod,Ptoa]        
        P=np.array(P)#safety to permit tuple, list and dataframe as input
        transformRandom=self.chooseRandomTransform(txPos, rxPos, clusters, P,mode3D)
        if skipLOS and plinfo[0]:
            Plos = subpaths.P.loc[0,self.maxM]
            clusters.P.loc[0]=clusters.P.loc[0] - Plos
        indexTopEnergyClusters=self.chooseEnergyPctileGlobal(clusters, Ec)
        if skipLOS and plinfo[0]:
            clusters.P.loc[0]=clusters.P.loc[0] + Plos
        transformIndexClusters=indexTopEnergyClusters*transformRandom
        (indexNot,indexAoA,indexAoD,indexToA)= self.getIndicesSubgroups(clusters,transformIndexClusters,maxG=4)
        tauOffset = self.fixExcessDelayNLOS(txPos,rxPos,plinfo,clusters,mode3D)
        clusters_bk=clusters.copy()
        clusters=self.applyMultiTransform(txPos,rxPos,clusters, tauOffset, indexAoA, indexAoD, indexToA,mode3D)
        delta_clusters = clusters-clusters_bk#save this for non adapted subpaths consistency
        
        if skipLOS and plinfo[0]:
            subpaths.P.loc[0,self.maxM] = 0
        indexTopEnergySubpaths=self.chooseEnergyPctileGroups(subpaths, Es)        
        if skipLOS and plinfo[0]:
            subpaths.P.loc[0,self.maxM] = Plos
            indexTopEnergySubpaths.loc[0,self.maxM] = True
        transformCBroadcastSubpaths=transformIndexClusters.reindex(index=indexTopEnergySubpaths.index,level=0)
        transformIndexSubpaths=indexTopEnergySubpaths*transformCBroadcastSubpaths
        (indexNot,indexAoA,indexAoD,indexToA)= self.getIndicesSubgroups(subpaths,transformIndexSubpaths,maxG=4)
        subpaths=self.applyMultiTransform(txPos,rxPos,subpaths, tauOffset, indexAoA, indexAoD, indexToA,mode3D)
        
        #TODO: non fitted subpaths that belong in a transformed cluster should get ther cluster mean values transformed
        transformShiftSubpaths=np.logical_not(indexTopEnergySubpaths)*transformCBroadcastSubpaths
        delta_clusters_broadcast = delta_clusters.reindex(indexTopEnergySubpaths.index,level=0)
        (indexNot,indexAoA,indexAoD,indexToA)= self.getIndicesSubgroups(subpaths,transformShiftSubpaths,maxG=4)
        subpaths.AOA.loc[indexAoA] = np.mod(subpaths.loc[indexAoA].AOA + delta_clusters_broadcast.loc[indexAoA].AOA,360)
        subpaths.AOD.loc[indexAoD] = np.mod(subpaths.loc[indexAoD].AOD + delta_clusters_broadcast.loc[indexAoD].AOD,360)
        subpaths.TDOA.loc[indexToA] += delta_clusters_broadcast.loc[indexToA].TDOA
        if mode3D:
            subpaths.ZOA.loc[indexAoA] = np.mod(subpaths.loc[indexAoA].ZOA + delta_clusters_broadcast.loc[indexAoA].ZOA,360)
            subpaths.ZOD.loc[indexAoD] = np.mod(subpaths.loc[indexAoD].ZOD + delta_clusters_broadcast.loc[indexAoD].ZOD,360)
                
        return(txPos,rxPos,plinfo,clusters,subpaths)
        
    
    def fullDeleteBacklobes(self,txPos,rxPos,plinfo,clusters,subpaths,tAOD=0,rAOA=180):
        tAODmin=np.mod(tAOD-90,360.0)
        tAODmax=np.mod(tAOD+90,360.0)
        rAOAmin=np.mod(rAOA-90,360.0)
        rAOAmax=np.mod(rAOA+90,360.0)        
        
        ## DEPRECATED cluster validity independently of subpaths
        # cAOD = np.mod(clusters.AOD,360.0)
        # cAOA = np.mod(clusters.AOA,360.0)
        # # cAODValid=(np.mod(cAOD-tAODmin,360)<np.mod(tAODmax-tAODmin,360))
        # if tAODmin<tAODmax:
        #     cAODValid = (cAOD>tAODmin)&(cAOD<tAODmax)
        # else:
        #     cAODValid = (cAOD<tAODmax)|(cAOD>tAODmin)
        # if rAOAmin<rAOAmax:
        #     cAOAValid = (cAOA>rAOAmin)&(cAOA<rAOAmax)
        # else:
        #     cAOAValid = (cAOA<rAOAmax)|(cAOA>rAOAmin)
        # cValid = cAODValid & cAOAValid
        # clusters = clusters[cValid]       
        
        sAOD = np.mod(subpaths.AOD,360.0)
        sAOA = np.mod(subpaths.AOA,360.0)
        if tAODmin<tAODmax:
            sAODValid = (sAOD>tAODmin)&(sAOD<tAODmax)
        else:
            sAODValid = (sAOD<tAODmax)|(sAOD>tAODmin)
        if rAOAmin<rAOAmax:
            sAOAValid = (sAOA>rAOAmin)&(sAOA<rAOAmax)
        else:
            sAOAValid = (sAOA<rAOAmax)|(sAOA>rAOAmin)
        sValid = sAODValid & sAOAValid
        subpaths = subpaths[sValid]
        #a cluster remains valid if any of its subpaths is valid
        cValid = [x in subpaths.index.get_level_values(0) for x in clusters.index]
        clusters = clusters[cValid]       
        return (txPos,rxPos,plinfo,clusters,subpaths)
        
    ###########################################################################
    # auxiliary functions. Contain actual post-processing logic and algorithms
    ###########################################################################
    
    def getIndicesSubgroups(self,data,vGroupIndex,maxG=None):
        # Forms Ng non-overlapping groups with dataframe Data,
        # returns only indices of each group so storage is fast (in place no copy)
        # vGroupIndex is a vector of integers from 0 to Ng-1
        # example: vector [0 1 2 0 2 2 1 0] returns
        # Indices of data elements 0, 2 and 7 for group 0
        # Indices of data elements 1, and 6 for group 1
        # Indices of data elements 2, 4 and 5 for group 
        # if maxG is None, implied group sets. otherwise range(maxG)
        lResult = []
        if maxG:
            groups=range(maxG)
        else:
            groups = np.unique(vGroupIndex)
        for g in groups:
            lResult.append(data.index[vGroupIndex==g])
        return(tuple(lResult))   
    
    def checkValidIntersection2D(self, txPos, rxPos, aoa, aod):
        vLOS = np.array(rxPos) - np.array(txPos)
        aod0 = np.mod(np.arctan(vLOS[1]/vLOS[0])+np.pi*(vLOS[0]<0),2*np.pi)
        dAOD = np.mod(aod - aod0,2*np.pi)
        dAOA = np.mod(aoa - aod0,2*np.pi)
        
        validIntersectionU = (dAOD<=np.pi)&(dAOA<=np.pi)&(dAOA>dAOD)
        validIntersectionL = (dAOD>np.pi)&(dAOA>np.pi)&(dAOA<dAOD)        
        return( validIntersectionU | validIntersectionL )
    
    def chooseRandomTransform(self,txPos,rxPos,data,P,mode3D):
        # Forms a group index vector with random probability P such that
        # P(group n) = P[n]
        # The last group, with P[-1], only admits paths with a valid intersection
        # for delay adaptation. Bayes is applied to guarantee global probability
        if mode3D:
            _,_,validIntersection = self.fitTDOA(txPos, rxPos,data.AOA*np.pi/180,data.AOD*np.pi/180,data.ZOA*np.pi/180,data.ZOD*np.pi/180, relax3D = False)
        else:
            validIntersection = self.checkValidIntersection2D(txPos,rxPos,data.AOA*np.pi/180,data.AOD*np.pi/180).to_numpy()
        PV = np.sum(validIntersection)/data.shape[0]#prob adapt tau is valid
        PtoaCV=np.minimum(P[3]/PV,1)# P (adapt toa | valid)
        PothersCV=(1-PtoaCV) * P[0:3] /(np.sum(P[0:3]))# P ( others | valid)
        PCV=np.concatenate([PothersCV,[PtoaCV]])
        PtoaCN=0# P (adapt tau | not valid) = 0
        PothersCN=P[0:3] /(np.sum(P[0:3]))# P ( others | not valid)
        PCN=np.concatenate([PothersCN,[PtoaCN]])
        r=np.random.rand(data.shape[0])
        transformIndexClusters=np.where(
        #if
            validIntersection,
        #then
            np.digitize(r,np.cumsum(PCV)),
        #else
            np.digitize(r,np.cumsum(PCN)),
            )
        return(pd.Series(transformIndexClusters,index=data.index))     
        
    def chooseEnergyPctileGlobal(self,data,E):
        # Forms a group index vector such that the strongest paths with sum
        # power sum_n(P[n])>E are in group 1 and the rest in group 0.
        # Thus enabling a boolean mask on other vectors.
        srtdP = data.P.sort_values(ascending=True)
        transformIndexClusters = (srtdP.cumsum()/srtdP.sum())>(1-E)
        return(transformIndexClusters.sort_index())
    def chooseEnergyPctileGroups(self,data,E):
        # Forms a group index vector such that the strongest paths with sum
        # power sum_m(P[n,m])>E for each n are in group 1 and the rest in 0
        # Applies separately to each value of level 0 of the data index,
        # thus enabling a boolean mask on other vectors.
        srtdP = data.P.sort_values(ascending=True)
        grpdP = srtdP.groupby(level=0)
        transformIndexClusters = (grpdP.cumsum()/grpdP.sum())>(1-E)
        return(transformIndexClusters.sort_index())
    
    def applyMultiTransform(self,txPos,rxPos,data,tauOffset,indexAoA,indexAoD,indexToA,mode3D=True):
        data['Xs']=np.full_like(data.TDOA,fill_value=np.inf)
        data['Ys']=np.full_like(data.TDOA,fill_value=np.inf)
        if mode3D:
            data['Zs']=np.full_like(data.TDOA,fill_value=np.inf)
            
        if not indexAoA.empty:
            if mode3D:
                aoa_new,zoa_new,locs_a = self.fitAOA(txPos,rxPos,data.loc[indexAoA].TDOA+ tauOffset,data.loc[indexAoA].AOD*np.pi/180,data.loc[indexAoA].ZOD*np.pi/180)
                data.Zs.loc[indexAoA]=locs_a[2,:]
            else:
                aoa_new,locs_a = self.fitAOA(txPos,rxPos,data.loc[indexAoA].TDOA+ tauOffset,data.loc[indexAoA].AOD*np.pi/180)
            data.AOA.loc[indexAoA]=aoa_new*180/np.pi
            data.Xs.loc[indexAoA]=locs_a[0,:]
            data.Ys.loc[indexAoA]=locs_a[1,:]
        if not indexAoD.empty:
            if mode3D:
                aod_new,zod_new,locs_d = self.fitAOD(txPos,rxPos,data.loc[indexAoD].TDOA+ tauOffset,data.loc[indexAoD].AOA*np.pi/180,data.loc[indexAoD].ZOA*np.pi/180)
                data.Zs.loc[indexAoD]=locs_d[2,:]
            else:
                aod_new,locs_d = self.fitAOD(txPos,rxPos,data.loc[indexAoD].TDOA+ tauOffset,data.loc[indexAoD].AOA*np.pi/180)
            data.AOD.loc[indexAoD]=aod_new*180/np.pi
            data.Xs.loc[indexAoD]=locs_d[0,:]
            data.Ys.loc[indexAoD]=locs_d[1,:]
        if not indexToA.empty:
            if mode3D:
                tau_new,locs_t,valid = self.fitTDOA(txPos,rxPos,data.loc[indexToA].AOA*np.pi/180,data.loc[indexToA].AOD*np.pi/180,data.loc[indexToA].ZOA*np.pi/180,data.loc[indexToA].ZOD*np.pi/180)
                data.Zs.loc[indexToA]=locs_t[2,:]            
            else:
                tau_new,locs_t,valid = self.fitTDOA(txPos,rxPos,data.loc[indexToA].AOA*np.pi/180,data.loc[indexToA].AOD*np.pi/180)
            data.TDOA.loc[indexToA]=tau_new
            data.Xs.loc[indexToA]=locs_t[0,:]
            data.Ys.loc[indexToA]=locs_t[1,:]
            if not np.all(valid):
                print("WARNING: Multi Transform was called with DToA option at paths that are not compatible. Some delays have not been transformed.")    
        
        return(data)
    
    def fixExcessDelayNLOS(self,txPos,rxPos,plinfo,clusters,mode3D=True):
        los,PLconst,sfdB = plinfo
        if not los and (np.min(clusters.TDOA)==0):# attempt to rebuild the initial delay of first  NLOStau observation        
            #we will always fix this by clusters, subpaths get the same correction factor
            sortclt = clusters.sort_values("TDOA")
            ctr = 0
            valid=False
            #find the first cluster which can be leveraged to reconstruct clock offset
            while (( not valid ) and (ctr < sortclt.shape[0]) ):
                aod=sortclt.iloc[ctr].AOD*np.pi/180
                aoa=sortclt.iloc[ctr].AOA*np.pi/180
                if mode3D:
                    zod=sortclt.iloc[ctr].ZOD*np.pi/180
                    zoa=sortclt.iloc[ctr].ZOA*np.pi/180        
                    fitted_tdoa,loc,valid = self.fitTDOA(txPos,rxPos,aoa,aod,zoa,zod,relax3D=True)#for tauE fixing in 3D we accept an approximate LS solution
                else:
                    fitted_tdoa,loc,valid = self.fitTDOA(txPos,rxPos,aoa,aod,relax3D=False)#for tauE fixing in 3D the intersection must still be exact
                if valid:
                    break
                else:
                    ctr=ctr+1
            if (ctr < sortclt.shape[0]):                
                #consider the cluster already-reported delay in the clock offset
                tauE = np.maximum(fitted_tdoa - sortclt.iloc[ctr].TDOA,0) # we cannot make the first cluster arrive earlier than the LOS path
                return( tauE )            
            #else
        #else
        return(0)
    
    def prepLocVectors(self,txPos,rxPos,clip2D=False):
        if clip2D:
            dtx = np.array(txPos[0:2])
            drx = np.array(rxPos[0:2])
        else:
            dtx = np.array(txPos)
            drx = np.array(rxPos)
        return(dtx,drx)
    
    def getUnitaryVectors(self,azimut,zenit=None):
        if zenit is None:
            uv = np.column_stack([np.cos(azimut),np.sin(azimut)])
        else:
            uv = np.column_stack([np.cos(azimut)*np.sin(zenit),np.sin(azimut)*np.sin(zenit),np.cos(zenit)])
        return(uv)#,av,zv)
    
    def fitAOA(self, txPos, rxPos, tdoa, aod, zod=None):        
        # Datos iniciais - l0
        (dtx,drx) = self.prepLocVectors(txPos,rxPos,clip2D = (zod is None))
        d0 = drx-dtx
        l0 = np.linalg.norm(d0)
        
        uDi = self.getUnitaryVectors(aod,zod)
        li = l0 + tdoa * 3e8                
        lDi =np.where(tdoa>0,
                      .5*(li**2-l0**2)/(li-(uDi[None,:]@d0[:,None]).reshape(-1)),
                      1e-6)#div0 safeward for LOS case
        
        posLocRelative = lDi*uDi.T
        dAi = posLocRelative-d0[:,None]
        aoaFix = np.mod(np.arctan2(dAi[1,:],dAi[0,:]),2*np.pi)
        posLoc = posLocRelative + dtx[:,None]
        if zod is None:            
            return (aoaFix,posLoc)
        else:
            zoaFix = np.mod(np.pi/2-np.arctan2(dAi[2,:],np.linalg.norm(dAi[0:2,:],axis=0)),2*np.pi)
            return (aoaFix,zoaFix,posLoc)

    def fitAOD(self, txPos, rxPos, tdoa, aoa,  zoa=None):               
        # Datos iniciais - l0
        (dtx,drx) = self.prepLocVectors(txPos,rxPos,clip2D = (zoa is None))
        d0 = drx-dtx
        l0 = np.linalg.norm(d0)
        
        uAi = self.getUnitaryVectors(aoa,zoa)
        li = l0+tdoa*3e8
        lAi =np.where(tdoa>0,
                      .5*(li**2-l0**2)/(li+(uAi[None,:]@d0[:,None]).reshape(-1)),
                      1e-6)#div0 safeward for LOS case
        
        dDi = lAi*uAi.T+d0[:,None]
        aodFix = np.mod(np.arctan2(dDi[1,:],dDi[0,:]),2*np.pi)
        posLoc = dDi + dtx[:,None]        
        if zoa is None:
            return (aodFix,posLoc)
        else:            
            zodFix = np.mod(np.pi/2-np.arctan2(dDi[2,:],np.linalg.norm(dDi[0:2,:],axis=0)),2*np.pi)
            return (aodFix,zodFix,posLoc)
    
    def fitTDOA(self, txPos, rxPos, aoa, aod, zoa=None, zod=None,relax3D = False):       
        # Datos iniciais - l0
        (dtx,drx) = self.prepLocVectors(txPos,rxPos,clip2D = ((zoa is None) or (zod is None)) )
        d0 = drx-dtx
        l0 = np.linalg.norm(d0)
        
        uDi = self.getUnitaryVectors(aod,zod)
        uAi = self.getUnitaryVectors(aoa,zoa)
        l0=np.linalg.norm(d0)
        U=np.stack([uDi,-uAi],axis=2)
        vli=np.stack([np.linalg.lstsq(U[n,:,:],d0,rcond=None)[0] for n in range(U.shape[0])])
        lAi=vli[:,1]
        lDi=vli[:,0]
        dAi = lAi*uAi.T
        dDi = lDi*uDi.T
        lDif = np.linalg.norm(d0[:,None]+dAi-dDi,axis=0)    
        li=lAi+lDi+lDif
        
        validIntersection=(lAi>0) & (lDi>0) & ( relax3D | np.isclose(lDif,0) )   
        tdoaFix = np.full_like(li,fill_value=np.inf)  
        tdoaFix[validIntersection]=(li[validIntersection]-l0)/3e8  
        posLoc = np.full_like(dDi,fill_value=np.inf)
        posLoc[:,validIntersection] = dDi[:,validIntersection] + dtx[:,None] 
        return (tdoaFix,posLoc,validIntersection)
           

    
    
