import numpy as np

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
    def __init__(self, tPos = (0,0), rPos = (0,0), lPaths = [] ):
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
