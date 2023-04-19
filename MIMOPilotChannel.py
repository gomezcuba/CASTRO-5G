import numpy as np

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
      
    (yp) r  = W^H * ( H * x +  * z )
    dimensiones r = [ simbolo OFDM, k, N_RF, 1]
    