import numpy as np

def AWGN(shape,sigma=1):
    return ( np.random.normal(size=shape) + 1j*np.random.normal(size=shape) ) * np.sqrt( sigma / 2.0 )

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
            h=np.fft.fft(h,axis=a)*np.sqrt(1.0*self.dims[a])        
        return h

class MIMOPilotChannel:
    def __init__(self, defAlgorithm="eye"):
        self.defAlgorithm=defAlgorithm
    def generatePilots(self,dimensions=(1,1,1,1,1,1),algorithm=False):
        
        if not algorithm:
            return self.generatePilots(dimensions,self.defAlgorithm)
        else:
            if algorithm == "eye":
                return self.generatePilotsEye(dimensions)
            elif algorithm == "Gaussian":
                return self.generatePilotsGaussian(dimensions)
            elif algorithm == "IDUV":
                return self.generatePilotsIDUV(dimensions)
            elif algorithm == "QPSK":
                return self.generatePilotsQPSK(dimensions)
            elif algorithm == "UPhase":
                return self.generatePilotsUPhase(dimensions)
            elif algorithm == "Rectangular":
                return self.generatePilotsRectangular(dimensions)
            else:
                print("Unrecognized algoritm %s"%algorithm)
    def generatePilotsEye(self,dimensions):
        (Nt,Nxp,Nrfr,Na,Nd,Nrft)=dimensions
        vp=np.empty(shape=(Nt,Nxp,Nd,Nrft))
        wp=np.empty(shape=(Nt,Nxp,Nrfr,Na))
        for k in range(0,Nt):
            for p in range(0,Nxp):
                vp[k,p,:,:]=np.eye(Nd,Nrft,-int(np.mod(p*Nrft,Nd/Nrft)))/np.sqrt(Nrft)
                wp[k,p,:,:]=np.eye(Nrfr,Na,int(np.floor(p*Nrft/Nd))*Nrft*Nrfr)
        return((wp,vp))
    def generatePilotsGaussian(self,dimensions):
        (Nt,Nxp,Nrfr,Na,Nd,Nrft)=dimensions
        vp=( np.random.normal(size=(Nt,Nxp,Nd,Nrft)) + 1j*np.random.normal(size=(Nt,Nxp,Nd,Nrft)) ) * np.sqrt( 1 / 2.0 /Nd )
        wp=( np.random.normal(size=(Nt,Nxp,Nrfr,Na)) + 1j*np.random.normal(size=(Nt,Nxp,Nrfr,Na)) ) * np.sqrt( 1 / 2.0 /Na )
        return((wp,vp))
    def generatePilotsIDUV(self,dimensions):
        (Nt,Nxp,Nrfr,Na,Nd,Nrft)=dimensions
        (wp,vp)=self.generatePilotsGaussian(dimensions)
        for k in range(0,Nt):
            for p in range(0,Nxp):
                for r in range(0,Nrft):
                    vp[k,p,:,r]=vp[k,p,:,r]/np.sqrt(np.sum(np.abs(vp[k,p,:,r])**2))/np.sqrt(Nrft)
                for r in range(0,Nrfr):
                    wp[k,p,r,:]=wp[k,p,r,:]/np.sqrt(np.sum(np.abs(wp[k,p,r,:])**2))
        return((wp,vp))
    def generatePilotsQPSK(self,dimensions):
        (Nt,Nxp,Nrfr,Na,Nd,Nrft)=dimensions
        vp=np.exp( .5j*np.pi*np.random.randint(0,4,size=(Nt,Nxp,Nd,Nrft)) ) * np.sqrt( 1 /Nd )
        wp=np.exp( .5j*np.pi*np.random.randint(0,4,size=(Nt,Nxp,Nrfr,Na)) ) * np.sqrt( 1 /Na )
        return((wp,vp))
    def generatePilotsUPhase(self,dimensions):
        (Nt,Nxp,Nrfr,Na,Nd,Nrft)=dimensions
        vp=np.exp( 2j*np.pi*np.random.uniform(size=(Nt,Nxp,Nd,Nrft)) ) * np.sqrt( 1 /Nd )
        wp=np.exp( 2j*np.pi*np.random.uniform(size=(Nt,Nxp,Nrfr,Na)) ) * np.sqrt( 1 /Na )
        return((wp,vp))
    def aUPA(self,theta,N):
        return( np.exp( -1j*np.pi * np.arange(0.0,N,1.0).reshape((1,N)) * np.sin(theta).reshape((len(theta),1)) ) /np.sqrt(1.0*N) )
    def generatePilotsRectangular(self,dimensions):
        if len(dimensions)==6:
            (Nt,Nxp,Nrfr,Na,Nd,Nrft)=dimensions
            exK=np.log(Nd)/np.log(Na*Nd)        
            Kt=int(np.floor( (Nxp*Nrfr)**(exK) ))
            Kr=int(np.floor(Nxp*Nrfr/Kt))
        elif  len(dimensions)==7:
            (Nt,Nxp,Nrfr,Na,Nd,Nrft,Kt)=dimensions
            Kr=int(np.floor(Nxp*Nrfr/Kt))
        elif  len(dimensions)==8:
            (Nt,Nxp,Nrfr,Na,Nd,Nrft,Kt,Kr)=dimensions
        Nangles=128
        RectWidtht=int(Nangles/Kt)
        RectWidthr=int(Nangles/Kr)
        allAngles=np.arange(0,np.pi,np.pi/Nangles)-np.pi/2
        At=self.aUPA(allAngles,Nd)
        Ar=self.aUPA(allAngles,Na)
        vp=np.empty(shape=(Nt,Nxp,Nd,Nrft),dtype=np.cdouble)
        wp=np.empty(shape=(Nt,Nxp,Nrfr,Na),dtype=np.cdouble)
        for k in range(0,Nt):
            for p in range(0,Nxp):
                for r in range(0,Nrfr):
                    bt=int(np.floor((r+p*Nrfr)/Kr))
                    br=int(np.mod((r+p*Nrfr),Kr))
                    Gt=np.zeros((Nangles,1),dtype=np.cdouble)
                    Gt[bt*RectWidtht:(bt+1)*RectWidtht,0]=1.0
                    Gr=np.zeros((Nangles,1),dtype=np.cdouble)
                    Gr[br*RectWidthr:(br+1)*RectWidthr,0]=1.0
                    vdes=np.linalg.lstsq(At,Gt,rcond=None)[0]
                    wdes=np.linalg.lstsq(Ar,Gr,rcond=None)[0].reshape((Na,))
                    vp[k,p,:,:]=vdes/np.sqrt(np.sum(np.abs(vdes)**2))
                    wp[k,p,r,:]=wdes/np.sqrt(np.sum(np.abs(wdes)**2))
        return((wp,vp))
    def applyPilotChannel(self,hk,wp,vp,zp):
        Nt=np.shape(vp)[0]
#        Nxp=np.shape(vp)[1]
        Nd=np.shape(vp)[2]
#        Nrfr=np.shape(wp)[2]
        Na=np.shape(wp)[3]
#        yp=np.empty(shape=(Nt,Nxp,Nrfr),dtype=np.cdouble)
#        for k in range(0,Nt):
#            for p in range(0,Nxp):
#                yp[k,p,:]=np.matmul(wp[k,p,:,:] , np.sum(np.matmul(hk[k,:,:],vp[k,p,:,:]),axis=1,keepdims=True)+zp[k,p,:,:] ).reshape((Nrfr,))                
        yp=np.matmul( wp,  np.sum( np.matmul( hk.reshape(Nt,1,Na,Nd) ,vp) ,axis=3,keepdims=True) + zp )        
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

def fULA(incidAngle , Nant = 4, dInterElement = .5):
    # returns an anttenna array response column vector corresponding to a Uniform Linear Array for each item in incidAngle (two extra dimensions are added at the end)
    # inputs  incidAngle : numpy array containing one or more incidence angles
    # input         Nant : number of MIMO antennnas of the array
    # input InterElement : separation between antenna elements, default lambda/2
    # output arrayVector : numpy array containing one or more response vectors, with simensions (incidAngle.shape ,Nant ,1 )
                        
    return np.exp( -2j * np.pi *  dInterElement * np.arange(Nant).reshape(Nant,1) * np.sin(incidAngle[...,None,None]) ) /np.sqrt(1.0*Nant)

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