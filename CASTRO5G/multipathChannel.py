import numpy as np
import pandas as pd
from itertools import product

import sys
sys.path.append('../')
from CASTRO5G import MultipathLocationEstimator

def AWGN(shape,sigma2=1):
    return ( np.random.normal(size=shape) + 1j*np.random.normal(size=shape) ) * np.sqrt( sigma2 / 2.0 )

#TODO get rid of column vector default

def fULA(incidAngle , Nant = 4, dInterElement = .5):
    # returns an anttenna array response vector corresponding to a Uniform Linear Array for each item in incidAngle (one extra dimensions is added at the end)
    # inputs  incidAngle : numpy array containing one or more incidence angles
    # input         Nant : number of MIMO antennnas of the array
    # input InterElement : separation between antenna elements, default lambda/2
    # output arrayVector : numpy array containing one or more response vectors, with simensions (incidAngle.shape ,Nant ,1 )
    
    if isinstance(incidAngle,np.ndarray):
        incidAngle=incidAngle[...,None]
                        
    return np.exp( -2j * np.pi *  dInterElement * np.arange(Nant) * np.sin(incidAngle) ) /np.sqrt(Nant)

def fUCA(incidAngle , Nant = 5, dInterElement = .5):
    if isinstance(incidAngle,np.ndarray):
        incidAngle=incidAngle[...,None]
    R=Nant*dInterElement/(2*np.pi)
    phiAnt=2*np.pi*np.arange(0,1,1/Nant)
    a=np.exp(-2j*np.pi*R*np.cos(incidAngle-phiAnt)) /np.sqrt(Nant)
    return(a)

def pSinc(tau,N,Nmin=0,oversampling=1,trunc=None):    
    if isinstance(tau,np.ndarray):
        tau=tau[...,None]
    taxis=np.arange(Nmin,N,1/oversampling)-tau
    if trunc is None:
       mask=1
    else:
       mask=(taxis<trunc)&(taxis>-trunc)
    return mask*np.sinc(taxis)#should work for any shape of tau, adding one dimension at the end  

def pCExp(tau,K,Kmin=0,oversampling=1):
    if isinstance(tau,np.ndarray):
        tau=tau[...,None]
    return np.exp(-2j*np.pi*np.arange(Kmin,K,1/oversampling)*tau/K)#should work for any shape of tau, adding one dimension at the end  

def pDirichlet(tau,P,Nmax=None,Nmin=0,oversampling=1):
    if Nmax is None:
        Nmax=P
    if isinstance(tau,np.ndarray):
        tau=tau[...,None]
    t=np.arange(Nmin,Nmax,1/oversampling)-tau
    if isinstance(P,np.ndarray):
        P=P[...,None]
    #divide by zero 
    return( np.where(t!=0,
               np.exp(1j*np.pi*(P-1)*t/P)*np.sin(np.pi*t)/np.sin(np.pi*t/P)/P,
               1+0j) )
    
class DiscreteMultipathChannelModel:
    def __init__(self,dims=(128,4,4),fftaxes=(1,2)):
        self.dims=dims
        self.Faxes=fftaxes
        self.Natoms=np.prod(dims)
    def getDEC(self,Npoints=1):
        indices=np.random.choice(self.Natoms,Npoints)
        h=np.zeros(self.Natoms,dtype=np.complex64)
        h[indices]=AWGN(Npoints)
        #TODO: config unnormalize
        h=h/np.sqrt(np.sum(np.abs(h)**2))
        h=h.reshape(self.dims)
        #TODO: config postprocessing functions
        for a in self.Faxes:
            h=np.fft.fft(h,axis=a,norm="ortho")    
        return h

class UniformMultipathChannelModel:
    def __init__(self,Npath=20,Ds=1,mode3D=True):
        self.Npath=Npath
        self.Ds=Ds
        self.mode3D=mode3D
    #TODO cache
    def create_channel(self,Nue=1):
        P = np.random.exponential(1,(Nue,self.Npath))
        P = P/np.sum(P,axis=-1,keepdims=True)
        phase = np.random.uniform(0,2*np.pi,(Nue,self.Npath))
        TDoA = np.random.uniform(0,self.Ds,(Nue,self.Npath))
        AoD = np.random.uniform(0,2*np.pi,(Nue,self.Npath))
        AoA = np.random.uniform(0,2*np.pi,(Nue,self.Npath))
        pathsData = pd.DataFrame(index=pd.MultiIndex.from_product([np.arange(Nue),np.arange(self.Npath)],names=["ue","n"]),
                                data={
                                    "P" : P.reshape(-1),
                                    "phase00" : phase.reshape(-1),
                                    "AoD" : AoD.reshape(-1),
                                    "AoA" : AoA.reshape(-1),
                                    "TDoA" : TDoA.reshape(-1)
                                    })
        if self.mode3D:
            ZoD = np.random.uniform(0,2*np.pi,(Nue,self.Npath))
            ZoA = np.random.uniform(0,2*np.pi,(Nue,self.Npath))
            pathsData['ZoD'] = ZoD.reshape(-1)
            pathsData['ZoA'] = ZoA.reshape(-1)
        return(pathsData)
        
    
class ReflectedMultipathChannelModel:
    def __init__(self,Npath=20,bounds=np.array(((0,1),(0,1),(0,1))),mode3D=True):
        self.Npath=Npath
        self.bounds=bounds
        self.mode3D=mode3D
        self.c=3e8
    #TODO cache
    def create_channel(self, txPos, rxPos):
        d0=np.array(rxPos)-np.array(txPos)
        Nue=np.prod(d0.shape[0:-1]) if len(d0.shape)>1 else 1
        ToA0=np.linalg.norm(d0,axis=-1)/self.c       
        Ndim= 3 if self.mode3D else 2
        #random locations in a 40m square
        Dmax=self.bounds[:,1]
        Dmin=self.bounds[:,0]
        d=np.random.rand(Nue,self.Npath,Ndim)*(Dmax-Dmin)[0:Ndim]+Dmin[0:Ndim]

        #delac
        ToA=(np.linalg.norm(d,axis=2)+np.linalg.norm(d-d0[:,None,:],axis=2))/self.c
        TDoA = ToA-ToA0[:,None]

        #angles from locations
        DoD=d/np.linalg.norm(d,axis=2,keepdims=True)
        DoA=(d-d0[:,None,:])/np.linalg.norm( d-d0[:,None,:] ,axis=2,keepdims=True)
        #TODO move angVector functions and such to a standalone path geometry module
        loc=MultipathLocationEstimator.MultipathLocationEstimator(nPoint=100,orientationMethod='lm',disableTQDM= True)
        if self.mode3D:
            AoD0,ZoD0=loc.angVector(d0)
            AoD,ZoD=loc.angVector(DoD)
            RoT0=np.random.rand(3)*np.array([2,1,2])*np.pi #receiver angular measurement offset
            R0=np.array([ loc.rMatrix(*x) for x in RoT0])
            DDoA=DoA@R0 #transpose of R0.T @ DoA.transpose([0,2,1])
            AoA,ZoA=loc.angVector(DDoA)
        else:    
            # AoD0=loc.angVector(d0)
            AoD=loc.angVector(DoD)
            # AoA=loc.angVector(DoA)
            RoT0=np.random.rand()*2*np.pi #receiver angular measurement offset
            R0=np.array([ loc.rMatrix(x) for x in RoT0])
            DDoA=DoA@R0 #transpose of R0.T @ DoA.transpose([0,2,1])
            AoA=loc.angVector(DDoA)
        allPathsData = pd.DataFrame(index=pd.MultiIndex.from_product([np.arange(Nue),np.arange(self.Npath)],names=["ue","n"]),
                                    data={
                                        "AoD" : AoD.reshape(-1),
                                        "AoA" : AoA.reshape(-1),
                                        "TDoA" : TDoA.reshape(-1),
                                        "Xs": d[...,0].reshape(-1),
                                        "Ys": d[...,1].reshape(-1),
                                        })
        if self.mode3D:
            allPathsData['Zs'] = d[...,2].reshape(-1)
            allPathsData['ZoD'] = ZoD.reshape(-1)
            allPathsData['ZoA'] = ZoA.reshape(-1)
        return (allPathsData)
        

class MIMOPilotChannel:
    def __init__(self,algorithm="MPSK", algConfig = { "M":2 } ):
        self.defAlgorithm = algorithm
        for name,value in algConfig.items():
            setattr(self,name,value)
    def getCbFun(self,algorithm):
        cbFunDict = {
            "Eye": self.getCodebookEye,
            "Gaussian": self.getCodebookGaussian,
            "IDUV": self.getCodebookIDUV,
            "MPSK": self.getCodebookMPSK,
            "UPhase": self.getCodebookMPSK,
            "Rectangular": self.getCodebookRectangular,
            "DMRS": self.getCodebookDMRS
            }
        return(cbFunDict[algorithm])
    def generatePilots(self,Np,Nr,Nt,Npr=None,rShape=None,tShape=None,algorithm=None):      
        if not algorithm:
            return self.generatePilots(Np,Nr,Nt,Npr,rShape,tShape,self.defAlgorithm)
        elif algorithm == "DMRS":
            #tratar Np, Npr, rShape y tShape de otra manera
            
            #vp[Nsim,K,Nrft]
            if tShape is None: 
                raise ValueError("El modo DMRS requiere tShape")
                
            Nsym,K,Nd,Nrft = tShape
            # Nrft = no. ports de los DMRSs
            # K = no. portadoras total (PRBset se saca de aqui dividiendo por 12)
            # Generar:
            Nslot = Nsym // self.DMRSLength
            K*= self.DMRSLength
            vp = self.getCodebookDMRS(Nd,K,Nslot).reshape(tShape)
            
            if rShape is None: 
                raise ValueError("El modo DMRS requiere rShape")
                
            Nsym,K,Nrfr,Na = rShape
            wp = np.ones((Nsym,K))[:,:,None,None] * np.eye(Nr)
            return((wp,vp))
        else:
            #este fragmento genera codebooks en 1 dim y les aplica un reshape
            bCombinatorial = ( algorithm in ["Eye","Rectangular"] )
            if not Npr:
                Npr=Np
            cbFun = self.getCbFun(algorithm)
            wp=cbFun(Nr,Npr).T.reshape(Npr,1,Nr)
            vp=cbFun(Nt,Np).T.reshape(Np,Nt,1)
            if bCombinatorial:
                wp=np.tile(wp,[Np,1,1,1]).reshape((Np*Npr,1,Nr))
                vp=np.tile(vp[:,None,:,:],[1,Npr,1,1]).reshape((Np*Npr,Nt,1)) 
            if rShape:
                wp=wp.reshape(rShape)
            if tShape:
                vp=vp.reshape(tShape)
            return((wp,vp))
    def getCodebookEye(self,Nant,Ncol):
        return( np.eye(Nant,Ncol) )
    def getCodebookFFT(self,Nant,Ncol):
        return( np.fft.fft( self.getCodebookEye(Nant,Ncol) ,norm="ortho",axis=0) )
    def getCodebookGaussian(self,Nant,Ncol):
        return( AWGN( (Nant,Ncol) , sigma2=1/Nant ) )
    def getCodebookIDUV(self,Nant,Ncol):
        w=AWGN( (Nant,Ncol) , sigma2=1/Nant )
        return(w/(np.linalg.norm(w,axis=0,keepdims=True)))
    def getCodebookMPSK(self,Nant,Ncol):
        return( np.exp( (2j*np.pi/self.M)*np.random.randint(0,self.M,size=(Nant,Ncol)) ) * np.sqrt( 1 /Nant ) )
    def getCodebookUPhase(self,Nant,Ncol):
        return( np.exp( 2j*np.pi*np.random.uniform(0,4,size=(Nant,Ncol)) ) * np.sqrt( 1 /Nant ) )
    def getCodebookRectangular(self,Nant,Ncol):
        Ndesiredgains=Nant*16
        angles_design = np.arange(-np.pi/2,np.pi/2,np.pi/Ndesiredgains)
        desired_G = np.zeros((Ndesiredgains,Ncol))
        for sec in range(Ncol):
            k = np.mod(sec+Ncol/2,Ncol)
            mask1 = angles_design >= (k-.5)*np.pi/Ncol -np.pi/2
            mask2 = angles_design < (k+.5)*np.pi/Ncol -np.pi/2
            if k == 0:
                mask2 = mask2 | (angles_design >= np.pi/2-.5*np.pi/Ncol)
            desired_G[mask1 & mask2,sec] = 1    
        A_array_design = fULA(angles_design, Nant, .5)
        W_ls,_,_,_=np.linalg.lstsq(A_array_design.conj(),desired_G,rcond=None)
        return(W_ls/np.linalg.norm(W_ls,axis=0))
    def getCodebookDMRS(self, Nant, K, Nslot):

    #estas 4 las tienes que sacar de Nsim, K y tShape, y se pasan en cada llamada a getCodebookDMRS            
        self.NumLayers = Nant
        self.PRBSet = int(np.ceil(K/12))
        self.DMRSPortSet = [1000 + i for i in range(Nant)]
        self.NSlot_list = list(range(Nslot))
        self.NSizeSymbol = Nslot 
        
        self.SymbolsPerSlot = 14
        self.SubcarriersPerPRB = 12
        self.NSymbols = self.NSizeSymbol * self.SymbolsPerSlot
        self.NSubcarriers = self.PRBSet * self.SubcarriersPerPRB
    
        dmrs_pilots = np.zeros((Nslot, K, Nant), dtype=complex)
        
        grid, dmrs_symbols = self.generate_dmrs_grid(self.PilotType)
        print(self.PilotType)
        
        for slot in range(Nslot):
            for ant in range(Nant):
                # Calculate symbol indices for this slot
                slot_symbols = [s + slot * self.SymbolsPerSlot for s in dmrs_symbols if s + slot * self.SymbolsPerSlot < grid.shape[0]]
                if not slot_symbols:
                    continue
                # Extract pilots for this antenna and slot
                pilots = grid[slot_symbols, :, ant].flatten()  # Flatten across symbols
                nonzero_pilots = pilots[pilots != 0]
                if len(nonzero_pilots) == 0:
                    continue
                # Tile to fill K subcarriers
                n_repeats = int(np.ceil(K / len(nonzero_pilots)))
                dmrs_pilots[slot, :, ant] = np.tile(nonzero_pilots, n_repeats)[:K]
        return dmrs_pilots
    
    def get_pdsch_dmrs_symbol_indices(self):
        start_symbol, duration = self.SymbolAllocation
        
        if self.MappingType == 'A':
            if duration <= 7:
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition],
                        2: [self.DMRSTypeAPosition],
                        3: [self.DMRSTypeAPosition],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                                    
            if duration in (8, 9):
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 7],
                        2: [self.DMRSTypeAPosition, 7],
                        3: [self.DMRSTypeAPosition, 7],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                            
            if duration in (10, 11):
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 9],
                        2: [self.DMRSTypeAPosition, 6, 9],
                        3: [self.DMRSTypeAPosition, 6, 9],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1, 8, 9],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                                    
            if duration == 12:
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 9],
                        2: [self.DMRSTypeAPosition, 6, 9],
                        3: [self.DMRSTypeAPosition, 5, 8, 11],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1, 8, 9],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                                    
            if duration in (13, 14):
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 11],
                        2: [self.DMRSTypeAPosition, 7, 11],
                        3: [self.DMRSTypeAPosition, 5, 8, 11],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1, 10, 11],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                    
        elif self.MappingType == 'B':
            if duration <= 4:
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol],
                        2: [start_symbol],
                        3: [start_symbol],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [],
                        1: [],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
    
            if duration in (5, 6, 7):
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 4],
                        2: [start_symbol, 4],
                        3: [start_symbol, 4],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
                    
            if duration == 8:
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 6],
                        2: [start_symbol, 3, 6],
                        3: [start_symbol, 3, 6],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1, 5, 6],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
                    
            if duration == 9:
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 7],
                        2: [start_symbol, 4, 7],
                        3: [start_symbol, 4, 7],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1, 5, 6],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
                    
            if duration == 10:
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 7],
                        2: [start_symbol, 4, 7],
                        3: [start_symbol, 4, 7],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1, 7, 8],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
    
            if duration == 11:
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 8],
                        2: [start_symbol, 4, 8],
                        3: [start_symbol, 3, 6, 9],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1, 7, 8],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
    
            if duration in (12, 13):
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 9],
                        2: [start_symbol, 5, 9],
                        3: [start_symbol, 3, 6, 9],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1, 8, 9],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
                
            if duration == 14:
                if self.DMRSLength == 1:
                    table = {
                        0: [],
                        1: [],
                        2: [],
                        3: [],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [],
                        1: [],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
        else:
            raise ValueError("Unsupported Mapping type (A or B)")
    
        if self.DMRSAdditionalPosition not in table:
            raise ValueError("Unsupported additional position. Use (0,1,2,3) for DMRSLength=1, (0,1) for DMRSLength=2")

        return sorted(table[self.DMRSAdditionalPosition])

    def get_pusch_dmrs_symbol_indices(self):
        start_symbol, duration = self.SymbolAllocation
        
        if self.MappingType == 'A':
            if duration < 4:
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition],
                        2: [self.DMRSTypeAPosition],
                        3: [self.DMRSTypeAPosition],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                    
            if duration in (4, 5, 6, 7):
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition],
                        2: [self.DMRSTypeAPosition],
                        3: [self.DMRSTypeAPosition],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                                    
            if duration in (8, 9):
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 7],
                        2: [self.DMRSTypeAPosition, 7],
                        3: [self.DMRSTypeAPosition, 7],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                            
            if duration in (10, 11):
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 9],
                        2: [self.DMRSTypeAPosition, 6, 9],
                        3: [self.DMRSTypeAPosition, 6, 9],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1, 8, 9],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                                    
            if duration == 12:
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 9],
                        2: [self.DMRSTypeAPosition, 6, 9],
                        3: [self.DMRSTypeAPosition, 5, 8, 11],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1, 8, 9],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                                    
            if duration in (13, 14):
                if self.DMRSLength == 1:
                    table = {
                        0: [self.DMRSTypeAPosition],
                        1: [self.DMRSTypeAPosition, 11],
                        2: [self.DMRSTypeAPosition, 7, 11],
                        3: [self.DMRSTypeAPosition, 5, 8, 11],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1],
                        1: [self.DMRSTypeAPosition, self.DMRSTypeAPosition+1, 10, 11],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type A")
                    
        elif self.MappingType == 'B':
            if duration <= 4:
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol],
                        2: [start_symbol],
                        3: [start_symbol],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [],
                        1: [],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
    
            if duration in (5, 6, 7):
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 4],
                        2: [start_symbol, 4],
                        3: [start_symbol, 4],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
                    
            if duration in (8, 9):
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 6],
                        2: [start_symbol, 3, 6],
                        3: [start_symbol, 3, 6],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1, 5, 6],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
                    
            if duration in (10, 11):
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 8],
                        2: [start_symbol, 4, 8],
                        3: [start_symbol, 3, 6, 9],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, start_symbol+1, 7, 8],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
                    
            if duration in (12, 13, 14):
                if self.DMRSLength == 1:
                    table = {
                        0: [start_symbol],
                        1: [start_symbol, 10],
                        2: [start_symbol, 5, 10],
                        3: [start_symbol, 3, 6, 9],
                    }
                elif self.DMRSLength == 2:
                    table = {
                        0: [start_symbol, start_symbol+1],
                        1: [start_symbol, start_symbol+1, 9, 10],
                    }
                else:
                    raise ValueError("Unsupported DMRS length for mapping type B")
        else:
            raise ValueError("Unsupported Mapping type (A or B)")
    
        if self.DMRSAdditionalPosition not in table:
            raise ValueError("Unsupported additional position. Use (0,1,2,3) for DMRSLength=1, (0,1) for DMRSLength=2")
    
        return sorted(table[self.DMRSAdditionalPosition])

    def generate_dmrs_grid(self, pilot):
        grid = np.zeros((self.NSymbols, self.NSubcarriers, self.NumLayers), dtype=complex)
        
        match pilot:
            case "PDSCH":
                dmrs_symbols = self.get_pdsch_dmrs_symbol_indices()
            case "PUSCH":
                dmrs_symbols = self.get_pusch_dmrs_symbol_indices()
            case _:
                print("Invalid option. Use PDSCH or PUSCH")
                return grid
    
        symbol_start, symbol_len = self.SymbolAllocation
        symbol_end = symbol_start + symbol_len
    
        for layer, port in enumerate(self.DMRSPortSet):
            sc_offsets = self.get_dmrs_subcarrier_offsets(port)
            for _, symbol in enumerate(dmrs_symbols):
    
                if not (symbol_start <= symbol < symbol_end):
                    continue
                
                for prb, rep in product(np.arange(self.PRBSet), np.arange(self.NSizeSymbol)):
                    c_init = ((2 ** 17) * (14 * self.NSlot_list[rep] + symbol + 1) * (2 * self.NIDNSCID + 1) + 2 * self.NIDNSCID + self.NSCID) % (2 ** 31)
                    c = self.generate_gold_sequence(c_init, len(sc_offsets) * 2)
                    r = self.qpsk_modulate(c)
                    
                    for i, offset in enumerate(sc_offsets):
                        k = prb * self.SubcarriersPerPRB + offset
                        l = rep * self.SymbolsPerSlot + symbol
                        symbol_val = self.apply_cdm(r[i], port)
                        grid[l, k, layer] = symbol_val

        # for port_idx in range(self.NumLayers):
        #     print(f"\n signal[:, :, {self.DMRSPortSet[port_idx]-1000}]")
        #     print(grid[:, :, port_idx].T)
        #     print('----------------------------------------------------------------')
                            
        return grid, dmrs_symbols

    def get_dmrs_subcarrier_offsets(self, port):
        port_idx = port - 1000
        k_values = []
        if self.DMRSConfigurationType == 1:
            if (port_idx % 4) in [0, 1]:
                delta = 0
                for n, k_p in product((0, 1, 2), (0, 1)):
                    k=4*n+2*k_p+delta
                    k_values.append(k)
                return np.array(k_values)
            else:
                delta = 1
                for n, k_p in product((0, 1, 2), (0, 1)):
                    k=4*n+2*k_p+delta
                    k_values.append(k)
                return np.array(k_values)
            
        elif self.DMRSConfigurationType == 2:
            if (port_idx % 6) in [0, 1]:
                delta = 0
                for n, k_p in product((0, 1), (0, 1)):
                    k=6*n+k_p+delta
                    k_values.append(k)
                return np.array(k_values)
            elif (port_idx % 6) in [2, 3]:
                delta = 2
                for n, k_p in product((0, 1), (0, 1)):
                    k=6*n+k_p+delta
                    k_values.append(k)
                return np.array(k_values)
            else:
                return np.array([0, 1, 6, 7])
            
        raise ValueError("Unsupported DMRS configuration. Use 1 or 2")
    
    def apply_cdm(self, r, port):
        port_idx = port - 1000
        if self.DMRSConfigurationType == 1:
            if (port_idx % 2 == 1 and port_idx <= 3) or (port_idx % 2 == 0 and port_idx > 3):
                r *= (-1)
        elif self.DMRSConfigurationType == 2:
            if (port_idx % 2 == 1 and port_idx <= 5) or (port_idx % 2 == 0 and port_idx > 5):
                r *= (-1)
        return r
   
    def generate_gold_sequence(self, c_init, length):
        N_c = 1600
        x1 = np.zeros(N_c + length, dtype=np.int32)
        x2 = np.zeros(N_c + length, dtype=np.int32)
    
        x1[0] = 1
        for i in range(31):
            x2[i] = (c_init >> i) & 1
    
        for n in range(31, N_c + length):
            x1[n] = (x1[n - 3] + x1[n - 31]) % 2
            x2[n] = (x2[n - 3] + x2[n - 2] + x2[n - 1] + x2[n - 31]) % 2
    
        c = (x1[N_c:] + x2[N_c:]) % 2
        return c
    
    def qpsk_modulate(self, bits):
        return (1 / np.sqrt(2)) * ((1 - 2 * bits[0::2]) + 1j * (1 - 2 * bits[1::2]))
    
    def applyPilotChannel(self,hk,wp,vp,zp=None):          
        yp=np.matmul( wp,  np.sum( np.matmul( hk[...,:,:,:] ,vp) ,axis=-1,keepdims=True) + ( 0 if zp is None else zp))        
        return(yp)
    def approxCBHBF(self,V,Nrft):
        Nt=V.shape[-1]
        if len(V.shape)>1:
            V=V.reshape(-1,Nt)
            Nvec=V.shape[0]
        Vbb=np.zeros((Nvec,Nrft),dtype=complex)
        Vrf=np.ones((Nvec,Nt,Nrft),dtype=complex)
        for nv in range(Nvec):
            r=V[nv,:]
            for nr in range(Nrft):
                Vrf[nv,:,nr]=np.exp(1j*np.angle(r))/np.sqrt(Nt)
                Vbb_unorm=np.linalg.lstsq(Vrf[nv,:,0:nr+1], V[nv,:],rcond=None)[0]
                Vbb[nv,0:nr+1]=Vbb_unorm/np.linalg.norm(Vrf[nv,:,0:nr+1]@Vbb_unorm)
                r=V[nv,:]-Vrf[nv,:,0:nr+1]@Vbb[nv,0:nr+1,None][...,0]
        Vrf=Vrf.reshape(V.shape+(Nrft,))
        Vbb=Vbb.reshape(V.shape[:-1]+(Nrft,))
        return(Vrf,Vbb)


class MultipathDEC:
    def __init__(self, tPos = (0,0,10), rPos = (0,1,1.5), dfPaths = None, customResponse=None ):
        self.txLocation = tPos
        self.rxLocation = rPos
        self.channelPaths = dfPaths
        if customResponse is None:
            self.fResponse = {
                "TDoA" : pSinc,
                "AoA" : fULA,
                "AoD" : fULA
                }
        else:
            self.fResponse = customResponse

        
    def insertPathsFromDF(self,dfPaths):
        self.channelPaths = dfPaths
        
    def insertPathsFromListParameters (self, lcoef, lTDoA, lAoD, lAoA, lZoD, lZoA, lDopp ):
        Npaths = min ( len(lcoef), len(lTDoA), len(lAoD), len(lAoA), len(lZoD), len(lZoA), len(lDopp) )        
        self.channelPaths = pd.DataFrame(index=np.arange(Npaths),
                                    data={
                                        "P" : np.abs(lcoef)**2,
                                        "phase00" : np.angle(lcoef),
                                        "TDoA" : lTDoA,
                                        "AoD" : lAoD,
                                        "AoA" : lAoA,
                                        "ZoD": lZoD,
                                        "ZoA": lZoA,
                                        "Dopp":lDopp
                                        })
    #TODO make Ts be stored into fResponse["TDoA"] and make Na,Nd,Nt be provided as a dims tuples
    #TODO make this into a loop of arbitrary dimensions with a list of names
    def getDEC(self,Na=1,Nd=1,Nt=1,Ts=1):
        coef=( np.sqrt(self.channelPaths.P)*np.exp(1j*self.channelPaths.phase00) ).to_numpy()
        timeResponse=self.fResponse["TDoA"](self.channelPaths["TDoA"].to_numpy()/Ts,Nt)
        arrivalResponse=self.fResponse["AoA"](self.channelPaths["AoA"].to_numpy(),Na)
        departureResponse=self.fResponse["AoD"](self.channelPaths["AoD"].to_numpy(),Nd)
        h=np.sum(coef[:,None,None,None]*timeResponse[:,:,None,None]*arrivalResponse[:,None,:,None]*departureResponse[:,None,None,:],axis=0)
        return(h)
        
    def dirichlet(self,t,P):
        if isinstance(t,np.ndarray):
            return np.array([self.dirichlet(titem,P) for titem in t])
        elif isinstance(t, (list,tuple)):            
            return [self.dirichlet(titem,P) for titem in t]
        else:
            return 1 if t==0 else np.exp(1j*np.pi*(P-1)*t/P)*np.sin(np.pi*t)/np.sin(np.pi*t/P)/P
        
    def createZakIR(self,N,M):
        h2D=np.zeros((N,M),dtype=np.complex64)
        for pit in self.channelPaths:
            tau0 = pit.excessDelay
            nu0 = pit.environmentDoppler
            g0 = pit.complexAmplitude
            
            tau = np.arange(N)-tau0
            St=  self.dirichlet(tau,N).reshape((N,1))
            nu = np.arange(M)-nu0
            Sv=  self.dirichlet(nu,M).reshape((1,M))
            
            nu=nu.reshape(1,M)
            tau=tau.reshape((N,1))
            Wv=  np.exp(-2j*np.pi*nu/M*np.floor(tau0/N))
            nuNoDelay=np.arange(M).reshape((1,M))
            Wt= np.exp(2j*np.pi*np.floor(nuNoDelay/M)*tau/N)
    #        tauNoDelay=np.arange(N).reshape((N,1))
            W4= np.exp(2j*np.pi*nu/M*tau/N)
            
            h2D+= g0 * W4 * Wt * Wv * (St*Sv)
        return(h2D)