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
        'zsd_mu',
        'zsd_sg',
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
        # 'zsd_mu',
        # 'zsd_sg',
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
    ])
    def __init__(self, fc = 28, sce = "UMi", corrDistance = 5.0):
        self.frecRefGHz = fc
        self.scenario = sce
        self.corrDistance = corrDistance
        self.dMacrosGenerated = {}
        self.dChansGenerated = {}
        self.bLargeBandwidthOption = False
        if sce.find("UMi")>=0:
            self.senarioLosProb = lambda d2D : 1 if d2D<18 else 18.0/d2D + np.exp(-d2D/36.0)*(1-18.0/d2D)

            self.scenarioParamsLOS = self.ThreeGPPScenarioParams(
                -.24*np.log10(1+fc)-7.14,
                .39,
                -.05*np.log10(1+fc)+1.21,
                .41,
                -.08*np.log10(1+fc)+1.73,
                0.014*np.log10(1+fc)+0.28,
                -.1*np.log10(1+fc)+0.73,
                -.03*np.log10(1+fc)+0.34,
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
            )
        else:
            print("Scenario TBW")
    def create_macro(self,txPos = (0,0), rxPos = (0,0)):
        aPos = np.array(txPos)
        bPos = np.array(rxPos)
        d=np.linalg.norm(bPos-aPos)
        d2D=np.sqrt((rxPos[0]-txPos[0])**2.0+(rxPos[1]-txPos[1])**2.0)
        pLos=self.senarioLosProb(d2D)
        los = (np.random.rand(1) <= pLos)
        if los:
            ds = 10.0**( self.scenarioParamsLOS.ds_mu + self.scenarioParamsLOS.ds_sg * np.random.randn(1) )
            asa = min( 10.0**( self.scenarioParamsLOS.asa_mu + self.scenarioParamsLOS.asa_sg * np.random.randn(1) ), 104.0)
            asd = min( 10.0**( self.scenarioParamsLOS.asd_mu + self.scenarioParamsLOS.asd_sg * np.random.randn(1) ), 104.0)
            zsa = min( 10.0**( self.scenarioParamsLOS.zsa_mu + self.scenarioParamsLOS.zsa_sg * np.random.randn(1) ), 52.0)
            zsd_mu = np.maximum(-0.21, -14.8*(d2D/1000.0) + 0.01*np.abs(rxPos[2]-txPos[2]) +0.83)
            zsd_sg = 0.35
            zsd = min( 10.0**( zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
            #TODO the avobe normals should be cross-correlated
            K= 0.5 if los else 0
            sf = 1
            fc= 28;#GHz
            clight=3e8
            dDP=4*(aPos[2]-1)*(bPos[2]-1)*(fc*1e9)/clight
            if d2D<dDP:
                PLdBLOS=28.0+22*np.log10(d)+20*np.log10(fc)
            else:
                PLdBLOS=28.0+40*np.log10(d)+20*np.log10(fc)-9.5*np.log10(dBP**2+(aPos[2]-bPos[2])**2)
            sflos=self.scenarioParamsLOS.sf_sg*np.random.randn(1)
            PL=PLdBLOS+sflos
            zod_offset_mu= 0
        else:
            ds = 10.0**( self.scenarioParamsNLOS.ds_mu + self.scenarioParamsNLOS.ds_sg * np.random.randn(1) )
            asa = min( 10.0**( self.scenarioParamsNLOS.asa_mu + self.scenarioParamsNLOS.asa_sg * np.random.randn(1) ), 104.0)
            asd = min( 10.0**( self.scenarioParamsNLOS.asd_mu + self.scenarioParamsNLOS.asd_sg * np.random.randn(1) ), 104.0)
            zsa = min( 10.0**( self.scenarioParamsNLOS.zsa_mu + self.scenarioParamsNLOS.zsa_sg * np.random.randn(1) ), 52.0)
            zsd_mu = np.maximum(-0.5, -3.1*(d2D/1000.0) + 0.01*np.maximum(rxPos[2]-txPos[2],0.0) +0.2)
            zsd_sg = 0.35
            zsd = min( 10.0**( zsd_mu + zsd_sg * np.random.randn(1) ), 52.0)
            #TODO the avobe normals should be cross-correlated
            K= 0.5 if los else 0
            sf = 1
            fc= 28;#GHz
            clight=3e8
            dDP=4*(aPos[2]-1)*(bPos[2]-1)*(fc*1e9)/clight
            if d2D<dDP:
                PLdBLOS=28.0+22*np.log10(d)+20*np.log10(fc)
            else:
                PLdBLOS=28.0+40*np.log10(d)+20*np.log10(fc)-9.5*np.log10(dBP**2+(aPos[2]-bPos[2])**2)
            PLdBNLOS=22.4+35.3*np.log10(d)+21.3*np.log10(fc)-0.3*np.log10(aPos[2]-1.5)
            sfnlos=self.scenarioParamsNLOS.sf_sg*np.random.randn(1)
            sflos=self.scenarioParamsLOS.sf_sg*np.random.randn(1)
            PL=np.maximum(PLdBNLOS+sfnlos,PLdBLOS+sflos)
            zod_offset_mu= 7.66*np.log10(fc)-5.96 - 10**( (0.208*np.log10(fc)- 0.782) * np.log10(max(25,d2D))  -0.13*np.log10(fc)+2.03 -0.07*(rxPos[2]-1.5))
        TgridXIndex= txPos[0] // self.corrDistance
        TgridYIndex= txPos[1] // self.corrDistance 
        RgridXIndex= rxPos[0] // self.corrDistance 
        RgridYIndex= rxPos[1] //self.corrDistance 
        key = (TgridXIndex,TgridYIndex,RgridXIndex,RgridYIndex)
        self.dMacrosGenerated[key]=self.ThreeGPPMacroParams(los,PL,ds,asa,asd,zsa,zsd,K,sf,zsd_mu,zsd_sg,zod_offset_mu)
    def create_channel(self,txPos = (0,0), rxPos = (0,0)):
        aPos = np.array(txPos)
        bPos = np.array(rxPos)
        vLOS = bPos-aPos
        d=np.linalg.norm(vLOS)
        losphiAoD=np.mod( np.arctan( vLOS[1] / vLOS[0] )+np.pi*(vLOS[0]<0), 2*np.pi )
        losphiAoA=np.mod(np.pi+losphiAoD, 2*np.pi ) # revise
        vaux = (np.linalg.norm(vLOS[0:2]), vLOS[2] )
        losthetaAoD=np.arctan( vaux[1] / vaux[0] )
        losthetaAoA=-losthetaAoD # revise
        #3GPP model is in degrees but numpy uses radians
        losphiAoD=(180.0/np.pi)*losphiAoD
        losthetaAoD=(180.0/np.pi)*losthetaAoD
        losphiAoA=(180.0/np.pi)*losphiAoA
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
        nClusters = param.N
        CphiNLOS = 1.273 #placeholder
        CthetaNLOS = 1.184
        #placeholders
        pldB= -120

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
            Cphi = CphiNLOS*( 1.1035 - 0.028*K + 0.002*(K**2) + 0.0001*(K**3) )
            Ctheta = CthetaNLOS*(  1.3086 - 0.0339*K + 0.0077*(K**2) +  0.0002*(K**3) )
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
        if self.bLargeBandwidthOption:
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
