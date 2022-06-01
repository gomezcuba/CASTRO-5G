import MIMOPilotChannel as pil
import threeGPPMultipathGenerator as pg

import numpy as np

class SimpleMultipathChannelModel:
    def __init__(self,Nt=1,Na=1,Nd=1):
        self.Nt=Nt
        self.Na=Na
        self.Nd=Nd
        self.Natoms=Nt*Nd*Na        
    def zAWGN(self,shape):
        return ( np.random.normal(size=shape) + 1j*np.random.normal(size=shape) ) * np.sqrt( 1 / 2.0 )
    def generateDEC(self,Npoints=1):
        indices=np.random.choice(self.Natoms,Npoints)
        h=np.zeros(self.Natoms,dtype=np.complex64)
        h[indices]=self.zAWGN(Npoints)
        h=h/np.sqrt(np.sum(np.abs(h)**2))
        h=h.reshape(self.Nt,self.Na,self.Nd)
        h=np.fft.fft(h,axis=1)
        h=np.fft.fft(h,axis=2)        
        return h*np.sqrt(1.0*self.Na*self.Nd)
    
class csProblemGenerator:
    
    def __init__(self,Nt=1, Nd=1, Na=1, Nrft=1, Nrfr=1, Nxp=1, Ts=1.0, pilotType = "Rectangular" , chanModel = "simple"):
        self.Nt = Nt
        self.Nd = Nd
        self.Na = Na
        self.Nrft = Nrft
        self.Nrfr = Nrfr
        self.Nxp = Nxp
        self.Ts = Ts
        self.pilGen = pil.MIMOPilotChannel(pilotType)
        if chanModel == "3GPP":
            self.model = pg.ThreeGPPMultipathChannelModel()
            self.model.bLargeBandwidthOption=True
        elif chanModel == "simple":
            self.model=SimpleMultipathChannelModel(Nt,Na,Nd)
        else:
            print("unknown channel model requested: '%s' "%chanModel)
        
        self.lastReturnedProblem = -1
        self.listPreparedProblems = []
        
    def pregenerate(self,N=1):
        for ichan in range(N):
            self.listPreparedProblems.append(self.generateOneAt(ichan))    
    def zAWGN(self,shape):
        return ( np.random.normal(size=shape) + 1j*np.random.normal(size=shape) ) * np.sqrt( 1 / 2.0 )
    def generateOneAt(self,n):            
#        self.model.create_channel((n,0,10),(n+40,0,1.5))
#        key='%s%s'%((n,0,10),(n+40,0,1.5))
#        h=np.sqrt(1.0*self.Nd*self.Na)*self.model.dChansGenerated[key].getDEC(self.Na,self.Nd,self.Nt,self.Ts)
#        h=np.transpose(h,(2,0,1))
        h=self.model.generateDEC(3)
        hk=np.fft.fft(h,axis=0)
        #            zh=zAWGN((Nt,Na,Nd))
        zp=self.zAWGN((self.Nt,self.Nxp,self.Na,1))
        (wp,vp)=self.pilGen.generatePilots((self.Nt,self.Nxp,self.Nrfr,self.Na,self.Nd,self.Nrft))      
        return(hk,zp,wp,vp)            
    def resetIterator(self):
        self.lastReturnedProblem=-1
    def getOrGenerateNext(self):
        if self.lastReturnedProblem+1 >= len(self.listPreparedProblems)-1:
            self.listPreparedProblems.append(self.generateOneAt(self.lastReturnedProblem))
        self.lastReturnedProblem += 1
        return(self.listPreparedProblems[self.lastReturnedProblem])