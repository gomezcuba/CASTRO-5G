import numpy as np

def AWGN(shape,sigma2=1):
    return ( np.random.normal(size=shape) + 1j*np.random.normal(size=shape) ) * np.sqrt( sigma2 / 2.0 )

def fULA(incidAngle , Nant = 4, dInterElement = .5):
    # returns an anttenna array response column vector corresponding to a Uniform Linear Array for each item in incidAngle (two extra dimensions are added at the end)
    # inputs  incidAngle : numpy array containing one or more incidence angles
    # input         Nant : number of MIMO antennnas of the array
    # input InterElement : separation between antenna elements, default lambda/2
    # output arrayVector : numpy array containing one or more response vectors, with simensions (incidAngle.shape ,Nant ,1 )
                        
    return np.exp( -2j * np.pi *  dInterElement * np.arange(Nant).reshape(Nant,1) * np.sin(incidAngle[...,None,None]) ) /np.sqrt(1.0*Nant)


def fUCA(incidAngle , Nant = 5, dInterElement = .5):
    R=Nant*dInterElement/(2*np.pi)
    phiAnt=2*np.pi*np.arange(0,1,1/Nant)
    a=np.exp(-2j*np.pi*R*np.cos(incidAngle[...,None,None]-phiAnt[:,None])) /np.sqrt(1.0*Nant)
    return(a)
    

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

class MIMOPilotChannel:
    def __init__(self, defAlgorithm="Eye"):
        self.defAlgorithm=defAlgorithm
    def getCbFun(self,algorithm):
        cbFunDict = {
            "Eye": self.getCodebookEye,
            "Gaussian": self.getCodebookGaussian,
            "IDUV": self.getCodebookIDUV,
            "QPSK": self.getCodebookQPSK,
            "UPhase": self.getCodebookQPSK,
            "Rectangular": self.getCodebookRectangular
            }
        return(cbFunDict[algorithm])
    def generatePilots(self,Np,Nr,Nt,Npr=None,rShape=None,tShape=None,algorithm=None):      
        if not algorithm:
            return self.generatePilots(Np,Nr,Nt,Npr,rShape,tShape,self.defAlgorithm)
        else:
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
    def getCodebookQPSK(self,Nant,Ncol):
        return( np.exp( .5j*np.pi*np.random.randint(0,4,size=(Nant,Ncol)) ) * np.sqrt( 1 /Nant ) )
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
        W_ls,_,_,_=np.linalg.lstsq(A_array_design[:,:,0].conj(),desired_G,rcond=None)
        return(W_ls)
    def applyPilotChannel(self,hk,wp,vp,zp=None):          
        yp=np.matmul( wp,  np.sum( np.matmul( hk[...,:,:,:] ,vp) ,axis=3,keepdims=True) + ( 0 if zp is None else zp))        
        return(yp)

#TODO: this class only purpose is to hold all data pertaining to a single path reflectoin, it should be replaced with a Pandas data frame
class ParametricPath:
    def __init__(self, g = 0, d = 0, aod = 0, aoa = 0, zod = 0, zoa = 0, nu = 0):
        self.complexAmplitude = g
        self.excessDelay = d
        self.azimutOfDeparture = aod
        self.azimutOfArrival = aoa
        self.zenithOfDeparture = zod
        self.zenithOfArrival = zoa
        self.environmentDoppler = nu
    def __str__(self):
        return "(%s,%f,%f,%f,%f,%f,%f)"%(self.complexAmplitude,self.excessDelay,self.azimutOfDeparture,self.azimutOfArrival,self.zenithOfDeparture,self.zenithOfArrival,self.environmentDoppler)

class MultipathChannel:
    def __init__(self, tPos = (0,0,10), rPos = (0,1,1.5), lPaths = [] ):
        self.txLocation = tPos
        self.rxLocation = rPos
        self.channelPaths = lPaths
                
    def __str__(self):
        return """
MultiPathChannel %s ---> %s
    %s
                """%(
                    self.txLocation,
                    self.rxLocation,
                    ''.join( [ '%s'%x for x in self.channelPaths] )
                    )                
    
#    def insertPathFromParameters (self, gain, delay, aod, aoa, zod, zoa, dopp ):
#        self.channelPaths.append( ParametricPath(gain, delay, aod, aoa, zod, zoa, dopp ) )
        
    def insertPathsFromListParameters (self, lG, lD, lAoD, lAoA, lZoD, lZoA, lDopp ):
        Npaths = min ( len(lG), len(lD), len(lAoD), len(lAoA), len(lZoD), len(lZoA), len(lDopp) )
        for pit in range(Npaths):
            self.channelPaths.append( ParametricPath(lG[pit],
                                                     lD[pit],
                                                     lAoD[pit],
                                                     lAoA[pit],
                                                     lZoD[pit],
                                                     lZoA[pit],
                                                     lDopp[pit]  ) )
    
    def getDEC(self,Na=1,Nd=1,Nt=1,Ts=1):
        h=np.zeros((Na,Nd,Nt))
        for pind in range(0,len( self.channelPaths )):
            delay=self.channelPaths[pind].excessDelay
            gain=self.channelPaths[pind].complexAmplitude
            AoD=self.channelPaths[pind].azimutOfDeparture
            AoA=self.channelPaths[pind].azimutOfArrival
            #TODO make the pulse and array vectors configurable using functional programming
            timePulse=np.sinc( (np.arange(0.0,Ts*Nt,Ts)-delay ) /Ts )
            departurePulse=np.exp( -1j* np.pi *np.arange(0.0,Nd,1.0)* np.sin(AoD) ) /np.sqrt(1.0*Nd)
            arrivalPulse=np.exp( -1j* np.pi *np.arange(0.0,Na,1.0)* np.sin(AoA) ) /np.sqrt(1.0*Na)
            timeTensor=np.reshape(timePulse,(1,1,Nt))
            departureTensor=np.reshape(departurePulse,(1,Nd,1))
            arrivalTensor=np.reshape(arrivalPulse,(Na,1,1))
            comp=gain*timeTensor*departureTensor*arrivalTensor
            h=h+comp
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