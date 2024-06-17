import numpy as np
import collections as col
import pandas as pd

#SIMPLIFIED algorithms assume that observation matrix is the identity, v is directly the sparse vector and dictionary columns are orthonormal
def simplifiedOMP(v,xi):
    #assumes v is explicitly sparse and returns its N largest coefficients where rho(N)<xi
    (Ncp,Nd,Na)=np.shape(v)
    r=v
    et=1j*np.zeros(np.shape(r))
    #at least one iteration
    ind=np.unravel_index(np.argmax(np.abs(r)),np.shape(r))
    et[ind]=r[ind]
    r=v-et
    ctr=1
    while np.sum(np.abs(r)**2)>xi:
        ind=np.unravel_index(np.argmax(np.abs(r)),np.shape(r))
        et[ind]=r[ind]
        r=v-et
        ctr=ctr+1
#    print('OMP ctr %d'%ctr)
    e=et
    return(e)
    
def simplifiedISTA(v,xi,Niter):
    #assumes v is explicitly sparse and returns N iterations of ISTA 
    (Ncp,Nd,Na)=np.shape(v)
    r=v
    et=np.zeros(np.shape(r))
    beta=.5
    for n in range(Niter):
        c=et+beta*r        
        et=np.exp(1j*np.angle(c))*np.maximum(np.abs(c)-xi,0)
        r=v-et
    e=et
    return(e)
    
def simplifiedFISTA(v,xi,Niter):
    #assumes v is explicitly sparse and returns N iterations of FISTA 
    (Ncp,Nd,Na)=np.shape(v)
    r=v
    et=np.zeros(np.shape(r))
    old_et=et
    beta=.5
    for n in range(Niter):
        c=et+beta*r+(et-old_et)*(n-2.0)/(n+1.0)
        old_et=et
        et=np.exp(1j*np.angle(c))*np.maximum(np.abs(c)-xi,0)
        r=v-et
    e=et
    return(e)

def simplifiedAMP(v,xi,Niter):
    #assumes v is explicitly sparse and returns N iterations of AMP 
    (Ncp,Nd,Na)=np.shape(v)
    r=v
    et=np.zeros(np.shape(r))
    old_r=r
    for n in range(Niter):
        bt=np.sum(np.abs(et)>0)/(Ncp*Nd*Na)
        c=et+r
        ldt = xi*np.sqrt(np.sum(np.abs(r)**2)/(Ncp*Nd*Na))
        et=np.exp(1j*np.angle(c))*np.maximum(np.abs(c)-ldt,0)
        old_r=r
        r=v-et+bt*old_r
    e=et
    return(e)

#------------------------------------------------------------------------------
# Baseline Dictionary with default behavior
class CSCachedDictionary:
    typeHCacheItem = col.namedtuple( "typeHCacheItem",[
            "TDoAdic",
            "AoDdic",
            "AoAdic",
            "mPhiH",
            "cacheYdic"
        ])
    
    typeYCacheItem = col.namedtuple( "typeHCacheItem",[
            "dimY",
            "pilotPattern",
            "mPhiY"
        ])
    def __init__(self,dimH=None,dimPhi=None):
        #TODO add K here
        self.funTDoAh= lambda TDoA,Ncp,K: np.fft.fft(np.sinc( np.arange(0.0,Ncp,1.0)[:,None] - TDoA ),K,axis=0,norm="ortho")
        self.funAoAh= lambda aoas,Na: np.exp( -1j*np.pi * np.arange(0.0,Na,1.0)[:,None] * np.sin(aoas) ) /np.sqrt(1.0*Na)
        self.funAoDh= lambda aods,Nd: np.exp( -1j*np.pi * np.arange(0.0,Nd,1.0)[:,None] * np.sin(aods) ) /np.sqrt(1.0*Nd)        
        self.dimH = dimH if dimH else None
        self.dimPhi = dimPhi if dimPhi else None
        self.currHDic = None
        self.cacheHDic = {}
        self.currYDic = None
    ###########################################################################
    # Functions that depend only on DEC simensions  
    # generally all children will use these functions without need to redefine
    def createHCols(self, TDoA,aoas,aods,dimH=None):        
        K,Ncp,Na,Nd = dimH if dimH else self.dimH
        #TODO add support for comb pilots
        vt=self.funTDoAh(TDoA,Ncp,K)[:,None,None,:]
        va=self.funAoAh(aoas,Na)[None,:,None,:]
        vd=self.funAoDh(aods,Nd)[None,None,:,:]
        vall=(vt*va*vd).reshape((K*Na*Nd,-1))#second dimension can be arbitrary, even a single column
        vall=vall/np.sqrt(np.sum(np.abs(vall)**2,axis=0)) #if vectors are not normalized, correlation-comparison is not valid        
        return( vall )
    
    def createYFromH(self, Hcols, pilotPattern=None):
        pilotPattern = pilotPattern if pilotPattern  else self.currYDic.pilotPattern
        K,Ncp,Na,Nd = self.dimH
        Ncolumns = Hcols.shape[1]
        Haux = Hcols.T.reshape(Ncolumns,1,K,Na,Nd)
        #TODO encapsulate in an "applyPilot" function
        wp,vp=pilotPattern
        vref=np.matmul( wp,  np.sum( np.matmul( Haux,vp) ,axis=4,keepdims=True) )        
        vref=vref.reshape(Ncolumns,-1).T
        #TODO study normalization variations
        # vref=vref/np.linalg.norm(vref,axis=0)
        return( vref )
    
    def createYCols(self, TDoA,aoas,aods, pilotPattern=None, dimH=None):
        pilotPattern = pilotPattern if pilotPattern else self.currYDic.pilotPattern
        Hcols = self.createHCols(TDoA, aoas, aods, dimH)
        return ( self.createYFromH(Hcols,pilotPattern) )
    
    def evalHCols(self,TDoA,aoas,aods,coefs, dimH=None):
        Hcols = self.createHCols(TDoA, aoas, aods, dimH)
        return(Hcols @ coefs)
    
    def evalYCols(self,TDoA,aoas,aods,coefs,pilotPattern, dimH=None):
        Ycols = self.createYCols(TDoA, aoas, aods,pilotPattern, dimH)
        return(Ycols @ coefs)
    
    def param2Ind(self,TDoA,aoas,aods):
        t_ind = np.where(self.currHDic.TDoAdic == TDoA)
        a_ind = np.where(self.currHDic.AoAdic == aoas)
        d_ind = np.where(self.currHDic.AoDdic == aods)
        ind=np.ravel_multi_index((t_ind,a_ind,d_ind),self.dimPhi)
        return (ind)
    def ind2Param(self,inds):
        t_ind,a_ind,d_ind = np.unravel_index(inds,self.dimPhi)
        return( self.currHDic.TDoAdic[t_ind], self.currHDic.AoAdic[a_ind], self.currHDic.AoDdic[d_ind] )
    
    def setHDic(self,dimH,dimPhi):
        if (dimH,dimPhi) not in self.cacheHDic:
            self.currHDic = self.createHDic(dimH,dimPhi)
            self.cacheHDic[(dimH,dimPhi)]=self.currHDic
        else:
            self.currHDic=self.cacheHDic[(dimH,dimPhi)]
        self.dimH=dimH
        self.dimPhi=dimPhi
        return(self.currHDic.mPhiH)
    def setYDic(self,pilotsID,pilotPattern=None):
        if pilotsID not in self.currHDic.cacheYdic:             
            if pilotPattern:
                self.currYDic = self.createYDic(pilotPattern)
                self.currHDic.cacheYdic[pilotsID] = self.currYDic
            else:
                raise(TypeError("CSCachedDictionary setYDic() requires pilotPattern argument if pilotsID is not in the cache"))
        else:
            self.currYDic = self.currHDic.cacheYdic[pilotsID]
        return(self.currYDic.mPhiY)
    
    
    def freeCacheOfPilot(self,pilotsID,dimH,dimPhi):
        if (dimH,dimPhi) in self.cacheHDic:
            if pilotsID in self.cacheHDic[(dimH,dimPhi)].cacheYdic:
                self.cacheHDic[(dimH,dimPhi)].cacheYdic.pop(pilotsID)
    
    ###########################################################################
    # Functions that create MATRIX dictionaries explicitly, DEC simensions x DIC dimensions
    # generally children will not use these functions and do not need to redefine
    def createHDic(self,dimH=None,dimPhi=None):                
        K,Ncp,Na,Nd = dimH if dimH else self.dimH
        Lt,La,Ld =dimPhi if dimPhi else self.dimPhi
        TDoAdic=np.arange(0.0,Ncp,float(Ncp)/Lt)#in discrete samples Ds = Ts*Ncp, fractional TDoA supported
        AoDdic=np.arcsin(np.arange(-1.0,1.0,2.0/Ld))
        AoAdic=np.arcsin(np.arange(-1.0,1.0,2.0/La))
        mPhiH=self.createHCols(
            np.tile( TDoAdic.reshape(Lt,1,1) , (1,Ld,La) ).reshape((Lt*Ld*La,)),
            np.tile( AoAdic.reshape(1,La,1) , (Lt,1,Ld) ).reshape((Lt*Ld*La,)),
            np.tile( AoDdic.reshape(1,1,Ld) , (Lt,La,1) ).reshape((Lt*Ld*La,)),
            dimH
            )
        return( self.typeHCacheItem(TDoAdic,AoDdic,AoAdic,mPhiH,{}) )   
    def createYDic(self,pilotPattern):
        #TODO encapsulate in an "applyPilot" function
        wp,vp=pilotPattern
        Nsym,K,Nrfr=wp.shape[0:3]
        dimY=(Nsym,K,Nrfr)
        mPhiY=self.createYFromH( self.currHDic.mPhiH, pilotPattern)
        return( self.typeYCacheItem( dimY, pilotPattern, mPhiY ) )
    ###########################################################################
    # Functions that define the interface of the dictionary
    # generally children MUST redefine these functions
    
    def getHCols(self,inds=None):
        inds = np.arange(np.prod(self.dimPhi),dtype=int) if inds is None else inds
        return(self.currHDic.mPhiH[:,inds])
    def getYCols(self,inds=None):
        inds = np.arange(np.prod(self.dimPhi),dtype=int) if inds is None else inds
        return(self.currYDic.mPhiY[:,inds])    
    def projY(self,vSamples):
        # Delay domain projection complexity K*Xt*Ncp
        return((vSamples.T.conj() @ self.currYDic.mPhiY ).T.conj())
    def evalHDic(self,coef,inds=None):
        if inds is None:
            Nvec=coef.shape[1]
            res=np.zeros((np.prod(self.dimH),Nvec))
            for v in range(Nvec):
                inds = np.where(coef[:,v]!=0)[0]
                res[:,v]=self.getHCols(inds)@coef[inds,v]
            return(res)
        else: 
            return(self.getHCols(inds) @coef)        
    def evalYDic(self,coef,inds=None):
        if inds is None:
            Nvec=coef.shape[1]
            res=np.zeros((np.prod(self.currYDic.dimY),Nvec))
            for v in range(Nvec):
                inds = np.where(coef[:,v]!=0)[0]
                res[:,v]=self.getYCols(inds)@coef[inds,v]
            return(res)
        else: 
            return(self.getYCols(inds)@coef)
        
#------------------------------------------------------------------------------
# An FFT-accelerated improvement of the baseline Dictionary, with default behavior in AoA and AoD
class CSBasicFFTDictionary(CSCachedDictionary):
    def createHDic(self,dimH=None,dimPhi=None):                
        K,Ncp,Na,Nd = dimH if dimH else self.dimH
        Lt,La,Ld =dimPhi if dimPhi else self.dimPhi
        TDoAdic=np.arange(0.0,Ncp,float(Ncp)/Lt)#in discrete samples Ds = Ts*Ncp, fractional TDoA supported
        AoDdic=np.arcsin(np.arange(-1.0,1.0,2.0/Ld))
        AoAdic=np.arcsin(np.arange(-1.0,1.0,2.0/La))
        mPhiH=self.createHCols(
            np.zeros((Ld*La,)),
            np.tile( AoAdic.reshape(La,1) , (1,Ld) ).reshape((Ld*La,)),
            np.tile( AoDdic.reshape(1,Ld) , (La,1) ).reshape((Ld*La,)),
            dimH
            )
        return( self.typeHCacheItem(TDoAdic,AoDdic,AoAdic,mPhiH,{}) )
    def getHCols(self,inds=None):
        inds = inds if inds else np.arange(np.prod(self.dimPhi),dtype=int)
        Ncol=len(inds)
        K,Ncp,Na,Nd = self.dimH
        Lt,La,Ld = self.dimPhi
        ind_tdoa,ind_angles=np.unravel_index(inds, (Lt,La*Ld))
        TDoA_vals=self.currHDic.TDoAdic[ind_tdoa]
        #the factor 1/sqrt(K) is already contained in mPhiH, which has K subcarriers with amplitude 1/sqrt(k) each
        col_tdoa=np.exp(-2j*np.pi*np.arange(0,1,1/K).reshape(1,K,1,1)*TDoA_vals)  
        col_angles=self.currHDic.mPhiH[:,ind_angles].reshape(K,Na*Nd,Ncol)
        col_tot= (col_tdoa*col_angles).reshape(K*Na*Nd,Ncol)  
        return(col_tot)
    def getYCols(self,inds=None):
        inds = inds if inds else np.arange(np.prod(self.dimPhi),dtype=int)
        Ncol=len(inds)
        Nsym,K,Nrfr=self.currYDic.dimY
        K,Ncp,Na,Nd = self.dimH
        Lt,La,Ld = self.dimPhi
        ind_tdoa,ind_angles=np.unravel_index(inds, (Lt,La*Ld))
        TDoA_vals=self.currHDic.TDoAdic[ind_tdoa]
        #the factor 1/sqrt(K) is already contained in mPhiH, which has K subcarriers with amplitude 1/sqrt(k) each
        col_tdoa=np.exp(-2j*np.pi*np.arange(0,1,1/K).reshape(1,K,1,1)*TDoA_vals)
        col_angles=self.currYDic.mPhiY[:,ind_angles].reshape(Nsym,K,Nrfr,Ncol)
        col_tot= (col_tdoa*col_angles).reshape(Nsym*K*Nrfr,Ncol)  
        return(col_tot)
    def projY(self,vSamples):
        # Delay domain projection complexity K*Xt*(1+log(Ncp*Xt))
        K,Ncp,Na,Nd = self.dimH
        Lt,La,Ld = self.dimPhi
        Nsym,K,Nrfr=self.currYDic.dimY
        Ncomb=K//Ncp
        Ccorr=self.currYDic.mPhiY.conj()*vSamples
        Ccomb=np.sum(Ccorr.reshape(Nsym,Ncp,Ncomb,Nrfr,Ld*La),axis=(0,3))
        Cfft=np.fft.ifft(Ccomb,Lt,axis=0)*Lt#fft conj
        Cbutterfly=Cfft*np.exp(2j*np.pi*np.arange(0,Ncp,Ncp/Lt).reshape(Lt,1,1)*np.arange(0,Ncomb/K,1/K).reshape(1,Ncomb,1))
        c=np.sum(Cbutterfly,axis=1).reshape(-1,1)
        return( c )

#------------------------------------------------------------------------------
# A multi dimensional dictionary without acceleration, but significant memory saving
class CSMultiDictionary(CSCachedDictionary):
    def createHDic(self,dimH=None,dimPhi=None):                
        K,Ncp,Na,Nd = dimH if dimH else self.dimH
        Lt,La,Ld =dimPhi if dimPhi else self.dimPhi
        TDoAdic=np.arange(0.0,Ncp,float(Ncp)/Lt)#in discrete samples Ds = Ts*Ncp, fractional TDoA supported
        AoAdic=np.arcsin(np.arange(-1.0,1.0,2.0/La))   
        AoDdic=np.arcsin(np.arange(-1.0,1.0,2.0/Ld))     
        mPhiH_tdoa=self.funTDoAh(TDoAdic,Ncp,K)
        mPhiH_aoa=self.funAoAh(AoAdic,Na)
        mPhiH_aod=self.funAoDh(AoDdic,Nd)
        mPhiH=(mPhiH_tdoa,mPhiH_aoa,mPhiH_aod)
        return( self.typeHCacheItem(TDoAdic,AoDdic,AoAdic,mPhiH,{}) )
    def getHCols(self,inds=None):
        inds = inds if inds else np.arange(np.prod(self.dimPhi),dtype=int)
        Ncol=len(inds)
        K,Ncp,Na,Nd = self.dimH
        Lt,La,Ld = self.dimPhi
        ind_tdoa,ind_aoa,ind_aod=np.unravel_index(inds, (Lt,La,Ld))
        col_tdoa = self.currHDic.mPhiH[0][:,None,None,ind_tdoa]
        col_aoa = self.currHDic.mPhiH[1][None,:,None,ind_aoa]
        col_aod = self.currHDic.mPhiH[2][None,None,:,ind_aod]
        col_tot= (col_tdoa*col_aoa*col_aod).reshape(K*Na*Nd,Ncol) #values are already repeated as necessary by unravel-index
        return(col_tot)    
    def createYDic(self,pilotPattern):
        wp,vp=pilotPattern
        Nsym,K,Nrfr=wp.shape[0:3]
        dimY=(Nsym,K,Nrfr)
        mPhiY_tdoa=self.currHDic.mPhiH[0]
        mPhiY_aoa=np.matmul(wp , self.currHDic.mPhiH[1])
        mPhiY_aod=np.matmul(self.currHDic.mPhiH[2].T,vp).transpose((0,1,3,2))
        mPhiY=(mPhiY_tdoa,mPhiY_aoa,mPhiY_aod)
        return( self.typeYCacheItem( dimY, pilotPattern, mPhiY ) )    
    def getYCols(self,inds=None):
        inds = inds if inds else np.arange(np.prod(self.dimPhi),dtype=int)
        Ncol=len(inds)
        Nsym,K,Nrfr = self.currYDic.dimY
        Lt,La,Ld = self.dimPhi
        ind_tdoa,ind_aoa,ind_aod=np.unravel_index(inds, (Lt,La,Ld))
        col_tdoa = self.currYDic.mPhiY[0][None,:,None,None,ind_tdoa]
        col_aoa = self.currYDic.mPhiY[1][:,:,:,None,ind_aoa]
        col_aod = self.currYDic.mPhiY[2][:,:,None,:,ind_aod]
        col_tot= (col_tdoa*col_aoa*col_aod).reshape(Nsym*K*Nrfr,Ncol) #values are already repeated as necessary by unravel-index
        return(col_tot) 
    def projY(self,vSamples):        
        Nsym,K,Nrfr = self.currYDic.dimY
        K,Ncp,Na,Nd = self.dimH
        Lt,La,Ld = self.dimPhi
        mPhiY_tdoa,mPhiY_aoa,mPhiY_aod = self.currYDic.mPhiY
        vSamples = vSamples.reshape((Nsym,K,Nrfr,1))
        v2 = np.matmul(mPhiY_aoa.transpose((0,1,3,2)).conj(),vSamples)
        v3 = np.matmul(v2,mPhiY_aod.conj())
        v4 = np.matmul(mPhiY_tdoa.T.conj(), v3.reshape(Nsym,K,La*Ld) )
        c  = np.sum(v4,axis=0).reshape((Lt*La*Ld,1))
        return(c)


#------------------------------------------------------------------------------
# A multi dimensional dictionary with FFT acceleration in all axes
class CSMultiFFTDictionary(CSMultiDictionary):
    def createHDic(self,dimH=None,dimPhi=None):                
        K,Ncp,Na,Nd = dimH if dimH else self.dimH
        Lt,La,Ld =dimPhi if dimPhi else self.dimPhi
        TDoAdic=np.arange(0.0,Ncp,float(Ncp)/Lt)#in discrete samples Ds = Ts*Ncp, fractional TDoA supported
        AoAdic=np.fft.fftshift(np.arcsin(np.arange(-1.0,1.0,2.0/La)))
        AoDdic=np.fft.fftshift(np.arcsin(np.arange(-1.0,1.0,2.0/Ld)))
        # mPhiH_aoa=self.funAoAh(AoAdic,Na)
        # mPhiH_aod=self.funAoDh(AoDdic,Nd)
        # mPhiH=(mPhiH_aoa,mPhiH_aod)
        mPhiH=None
        return( self.typeHCacheItem(TDoAdic,AoDdic,AoAdic,mPhiH,{}) )    
    def getHCols(self,inds=None):
        inds = inds if inds else np.arange(np.prod(self.dimPhi),dtype=int)
        Ncol=len(inds)
        K,Ncp,Na,Nd = self.dimH
        Lt,La,Ld = self.dimPhi
        ind_tdoa,ind_aoa,ind_aod=np.unravel_index(inds, (Lt,La,Ld))
        TDoA_vals=self.currHDic.TDoAdic[ind_tdoa]
        col_tdoa=np.exp(-2j*np.pi*np.arange(0,1,1/K).reshape(K,1,1,1)*TDoA_vals) /np.sqrt(K)
        # col_tdoa = self.funTDoAh(TDoA_vals,Ncp,K)[:,None,None,:]
        AoA_vals=self.currHDic.AoAdic[ind_aoa]
        col_aoa = self.funAoAh(AoA_vals,Na)[None,:,None,:]
        AoD_vals=self.currHDic.AoDdic[ind_aod]
        col_aod = self.funAoDh(AoD_vals,Nd)[None,None,:,:]
        col_tot= (col_tdoa*col_aoa*col_aod).reshape(K*Na*Nd,Ncol) #values are already repeated as necessary by unravel-index
        return(col_tot)
    def createYDic(self,pilotPattern):
        wp,vp=pilotPattern
        Nsym,K,Nrfr=wp.shape[0:3]
        dimY=(Nsym,K,Nrfr)
        mPhiY = None
        return( self.typeYCacheItem( dimY, pilotPattern, mPhiY ) )  
    def getYCols(self,inds=None):
        inds = inds if inds else np.arange(np.prod(self.dimPhi),dtype=int)
        Ncol=len(inds)
        Nsym,K,Nrfr = self.currYDic.dimY
        wp,vp = self.currYDic.pilotPattern
        Lt,La,Ld = self.dimPhi
        K,Ncp,Na,Nd = self.dimH
        ind_tdoa,ind_aoa,ind_aod=np.unravel_index(inds, (Lt,La,Ld))
        TDoA_vals=self.currHDic.TDoAdic[ind_tdoa]
        col_tdoa=np.exp(-2j*np.pi*np.arange(0,1,1/K).reshape(1,K,1,1,1)*TDoA_vals)/np.sqrt(K)
        # col_tdoa = self.funTDoAh(TDoA_vals,Ncp,K)[None,:,None,None,:]
        AoA_vals=self.currHDic.AoAdic[ind_aoa]
        col_aoa = np.matmul( wp, self.funAoAh(AoA_vals,Na))[:,:,:,None,:]
        AoD_vals=self.currHDic.AoDdic[ind_aod]
        col_aod = np.matmul( self.funAoDh(AoD_vals,Nd).T ,vp ).transpose(0,1,3,2)[:,:,None,:,:]
        col_tot= (col_tdoa*col_aoa*col_aod).reshape(Nsym*K*Nrfr,Ncol) #values are already repeated as necessary by unravel-index
        return(col_tot)   
    def projY(self,vSamples):
        # Delay domain projection complexity K*Xt*(1+log(Ncp*Xt))
        K,Ncp,Na,Nd = self.dimH
        Lt,La,Ld = self.dimPhi
        Nsym,K,Nrfr=self.currYDic.dimY
        wp,vp = self.currYDic.pilotPattern
        vSamples = vSamples.reshape((Nsym,K,Nrfr,1))        
        v2 = np.fft.ifft(np.matmul(wp.transpose(0,1,3,2).conj(),vSamples),La,axis=2,norm='ortho')*np.sqrt(La/Na)
        v3 = np.fft.ifft(np.matmul(v2,vp.transpose(0,1,3,2).conj()),Ld,axis=3,norm='ortho')*np.sqrt(Ld/Nd)
        Ncomb=K//Ncp
        Ccorr=v3.reshape(Nsym,Ncp,Ncomb,La,Ld)#sum of Nrfr done by matmul
        Ccomb=np.sum(Ccorr,axis=0)
        Cfft=np.fft.ifft(Ccomb,Lt,axis=0)*Lt/np.sqrt(K)#fft conj
        Cbutterfly=Cfft*np.exp(2j*np.pi*np.arange(0,Ncp,Ncp/Lt).reshape(Lt,1,1,1)*np.arange(0,Ncomb/K,1/K).reshape(1,Ncomb,1,1))
        c=np.sum(Cbutterfly,axis=1).reshape(-1,1)
        # Cfft2=Ccomb*np.exp(2j*np.pi*np.arange(0,Ncomb/K,1/K).reshape(1,Ncomb,1,1))
        # Cfft2=np.fft.ifft(Cfft2,Lt,axis=0)*Lt/np.sqrt(K)#fft conj
        # Cbutterfly2=Cfft2*np.exp(2j*np.pi*np.arange(0,Ncp,Ncp/Lt).reshape(Lt,1,1,1))
        # print(f'Same result {np.max(np.abs(Cbutterfly2-Cbutterfly))}')
        return( c )

OMPInfoSet = col.namedtuple( "OMPInfoSet",[
        "multipaths",
        "Ycols",
        "Hcols",
    ])

class OMPCachedRunner:
    def __init__(self, dictionary=None):
        self.cachedDics={}
        self.dictionaryEngine = dictionary if dictionary else CSCachedDictionary()
    def getDictionary(self):
        return(self.dictionaryEngine)
    def setDictionary(self,d):
        self.dictionaryEngine = d
                
    def getBestProjDicInd(self,vSamples):
        c=self.dictionaryEngine.projY(vSamples)
        s_ind=np.argmax(np.abs(c))
        c_max=c[s_ind,0]        
        TDoA_new,AoA_new,AoD_new = self.dictionaryEngine.ind2Param(s_ind)            
        vRsupp_new=self.dictionaryEngine.getYCols([s_ind])[:,0]
        return(s_ind,c_max,TDoA_new,AoA_new,AoD_new,vRsupp_new)
    def getBestProjBR(self,vSamples,Xmu):
        s_ind,c_max,TDoA_new,AoA_new,AoD_new,vRsupp_new=self.getBestProjDicInd(vSamples)
        K,Ncp,Na,Nd = self.dictionaryEngine.dimH
        Lt,La,Ld = self.dictionaryEngine.dimPhi
        boolTable=np.unpackbits(np.arange(8,dtype=np.uint8).reshape(-1,1),axis=1,count=3,bitorder='little')
        muTable=boolTable-.5
        mid_mu=np.array([0,0,0],dtype=np.double)    
        while ( np.any(np.abs(mid_mu-muTable)>(.5/Xmu)) ):
            TDoA_mu = TDoA_new + muTable[:,0]*Ncp/Lt             
            AoA_mu = np.arcsin(np.mod( np.sin(AoA_new) - muTable[:,1]*2.0/La +1,2)-1)
            AoD_mu = np.arcsin(np.mod( np.sin(AoD_new) - muTable[:,2]*2.0/Ld +1,2)-1)
            vref=self.dictionaryEngine.createYCols(TDoA_mu,AoA_mu,AoD_mu)
            corr_mu=np.matmul( vref.transpose().conj() , vSamples )
            bestInd=np.argmax(np.abs(corr_mu))
            best_mu=muTable[bestInd,:]
            muTable=mid_mu*(1-boolTable)+best_mu*boolTable #[mid_mu,mid_mu,mid_mu] ... [best_mu,best_mu,best_mu]
            mid_mu=np.mean(muTable,axis=0)            
        #     print ( np.abs(c_max) , np.abs(corr_mu[bestInd,0]) )
        #     print((TDoA_new,AoA_new*180/np.pi,AoD_new*180/np.pi),(TDoA_mu[bestInd],AoA_mu[bestInd]*180/np.pi,AoD_mu[bestInd]*180/np.pi))
        # print ( "-"*30 ) 
        if ( np.abs(c_max) < np.abs(corr_mu[bestInd,0]) ):
            vRsupp_new    = vref[:,bestInd]
            TDoA_new =  TDoA_mu[bestInd]
            AoA_new   = AoA_mu[bestInd]
            AoD_new   = AoD_mu[bestInd]
            c_max = corr_mu[bestInd,0]
        return(s_ind,c_max,TDoA_new,AoA_new,AoD_new,vRsupp_new)
                
    def OMPBR(self,v,xi,pilotsID,vp,wp, Xt=1.0, Xd=1.0, Xa=1.0, Xmu=1.0, Ncp=None):
        Nsym=np.shape(vp)[0]
        K=np.shape(v)[1]
        Ncp = K if Ncp is None else Ncp
        Nd=np.shape(vp)[2]
#        Nrft=np.shape(vp)[3]
        Nrfr=np.shape(wp)[2]
        Na=np.shape(wp)[3]
        vflat=v.reshape(Nsym*K*Nrfr,1)
        
        Lt,La,Ld=(int(Ncp*Xt),int(Nd*Xd),int(Na*Xa))
        self.dictionaryEngine.setHDic((K,Ncp,Na,Nd),(Lt,La,Ld)) 
        self.dictionaryEngine.setYDic(pilotsID,(wp,vp))        
            
        r=vflat
        Rsupp=np.zeros(shape=(Nsym*K*Nrfr,Nsym*K*Nrfr),dtype=np.complex128)
        delay_supp=np.zeros(Nsym*K*Nrfr)
        aod_supp=np.zeros(Nsym*K*Nrfr)
        aoa_supp=np.zeros(Nsym*K*Nrfr)
        et=np.empty(shape=np.shape(r))
        ctr=0        
        while ((ctr<Nsym*K*Nrfr) and (np.sum(np.abs(r)**2)>xi)) or ctr==0:            
            if Xmu>1.0:
                ind,c_max,TDoA_new,AoA_new,AoD_new,vRsupp_new = self.getBestProjBR(r,Xmu)
            else:
                ind,c_max,TDoA_new,AoA_new,AoD_new,vRsupp_new = self.getBestProjDicInd(r)
            # print(ind,c_max,TDoA_new,AoA_new*180/np.pi,AoD_new*180/np.pi)
            delay_supp[ctr] = TDoA_new
            aoa_supp[ctr] = AoA_new
            aod_supp[ctr] = AoD_new
            Rsupp[:,ctr] = vRsupp_new
            vflat_proj,_,_,_=np.linalg.lstsq(Rsupp[:,0:ctr+1],vflat,rcond=None)
            et=np.matmul(Rsupp[:,0:ctr+1], vflat_proj )
            r=vflat-et
            # print('OMPBR %s ctr %d, |r|²= %f'%(self.dictionaryEngine.__class__.__name__,ctr,np.sum(np.abs(r)**2)))
            ctr=ctr+1
#        print('OMPBR ctr %d'%ctr)        
        Hsupp = self.dictionaryEngine.createHCols(delay_supp[0:ctr] , aoa_supp[0:ctr], aod_supp[0:ctr])
        multipath = pd.DataFrame({
            "coefs" : vflat_proj[:,0],
            "TDoA"  : delay_supp[0:ctr],
            "AoA"   : aoa_supp[0:ctr],
            "AoD"   : aod_supp[0:ctr]
            })
        hest= Hsupp @ vflat_proj
        return( hest.reshape(K,Na,Nd), multipath , Rsupp[:,0:ctr] , Hsupp )
        
    def Shrinkage(self,v,shrinkageParams,Nit,pilotsID,vp,wp, Xt=1.0, Xd=1.0, Xa=1.0, shrinkageAlg = "ISTA", Ncp=None):
        Nsym=np.shape(vp)[0]
        K=np.shape(v)[1]
        Ncp = K if Ncp is None else Ncp
        Nd=np.shape(vp)[2]
#        Nrft=np.shape(vp)[3]
        Nrfr=np.shape(wp)[2]
        Na=np.shape(wp)[3]
        vflat=v.reshape(Nsym*K*Nrfr,1)
        r=vflat
        
        Lt,La,Ld=(int(Ncp*Xt),int(Nd*Xd),int(Na*Xa))
        self.dictionaryEngine.setHDic((K,Ncp,Na,Nd),(Lt,La,Ld)) 
        self.dictionaryEngine.setYDic(pilotsID,(wp,vp))        
        
#        print("max beta %f"%(1.0/np.sum(np.abs(outputDic)**2)))
    
        et=1j*np.zeros((Lt*La*Ld,1))
        if  shrinkageAlg == "FISTA":
            old_et=et
        if  shrinkageAlg == "VAMP":
            A=self.dictionaryEngine.getYCols()
            (U,S,Vh)=np.linalg.svd(A)
            M1=(S*U).T.conj()               #faster implementation of np.matmul(np.diag(S),U.T.conj())
            r_mmse=r
            sigma2w=shrinkageParams[0]
            sigma2t_mmse=sigma2w
            alpha=shrinkageParams[1]
        for itctr in range(Nit):
            if shrinkageAlg == "FISTA":
                lamb=shrinkageParams[0]
                beta=shrinkageParams[1]
                c=et+beta*self.dictionaryEngine.projY( r )+(et-old_et)*(itctr-2.0)/(itctr+1)
                old_et=et
                et=np.exp(1j*np.angle(c))*np.maximum(np.abs(c)-lamb,0)
                r=vflat-self.dictionaryEngine.evalYDic(et)
            elif shrinkageAlg == "ISTA":
                lamb=shrinkageParams[0]
                beta=shrinkageParams[1]
                c=et+beta*self.dictionaryEngine.projY( r )
                et=np.exp(1j*np.angle(c))*np.maximum(np.abs(c)-lamb,0)
                r=vflat-self.dictionaryEngine.evalYDic(et)
            elif  shrinkageAlg == "AMP":
                alpha=shrinkageParams[0]
                bt=np.sum(np.abs(et)>0)/(Nsym*Ncp*Nrfr)
                c=et+self.dictionaryEngine.projY( r )
                lamb = alpha*np.sqrt(np.sum(np.abs(r)**2)/(Nsym*Ncp*Nrfr))
                et=np.exp(1j*np.angle(c))*np.maximum(np.abs(c)-lamb,0)
                old_r=r
                r=vflat-self.dictionaryEngine.evalYDic(et)+bt*old_r
            elif  shrinkageAlg == "VAMP":
                #LMMSE stage
                M2=sigma2w/sigma2t_mmse*Vh
#                M3=np.diag(S**2+sigma2w/sigma2t_mmse)
#                if np.any(np.isnan(np.matmul(M1,vflat)+np.matmul(M2,r_mmse))) or  np.any(np.isinf(np.matmul(M1,vflat)+np.matmul(M2,r_mmse))):
#                    print("aqui")
#                et_mmse=np.matmul(Vh.T.conj(),np.linalg.lstsq(M3,np.matmul(M1,vflat)+np.matmul(M2,r_mmse),rcond=None)[0])
                M3_diagInv=( 1/(S**2+sigma2w/sigma2t_mmse) ).reshape((Nsym*Ncp*Nrfr,1))
                et_mmse=np.matmul(Vh.T.conj(), M3_diagInv * np.matmul(M1,vflat)+np.matmul(M2,r_mmse) )
                vt_mmse=np.sum(1.0/((S**2)*sigma2t_mmse/sigma2w+1))/(Ncp*Nd*Na)
                r=et_mmse-vt_mmse*r_mmse//(1-vt_mmse)
                sigma2t=sigma2t_mmse*vt_mmse/(1-vt_mmse)
                #Shrinkage stage
                lamb=alpha*np.sqrt(sigma2t/(Nsym*Ncp*Nrfr))
                et=np.exp(1j*np.angle(r))*np.maximum(np.abs(r)-lamb,0)
                vt=np.sum(np.abs(et)>0)/(Nsym*Ncp*Nrfr)
                r_mmse=(et-vt*r)/(1-vt)
                sigma2t_mmse=sigma2t*vt/(1-vt)
            else:
                print("Unknown algoritm")
                return((0,0))
            
        lInds=np.where(et>0)[0]                
        TDoA,AoA,AoD = self.dictionaryEngine.ind2Param(lInds)
        Hsupp = self.dictionaryEngine.createHCols(TDoA,AoA,AoD)
        Ysupp = self.dictionaryEngine.createYFromH(Hsupp)
        multipath = pd.DataFrame({
            "coefs" : et[lInds],
            "TDoA"  : TDoA,
            "AoA"   : AoA,
            "AoD"   : AoD
            })
        hest= Hsupp @ et[lInds]
        return( hest.reshape(Ncp,Na,Nd), multipath , Ysupp , Hsupp )
            