import numpy as np
import collections as col

OMPInfoSet = col.namedtuple( "OMPInfoSet",[
        "coefs",
        "delays",
        "AoDs",
        "AoAs",
        "observations",
        "outputs",
    ])

class OMPCachedRunner:
    def __init__(self, preloadDics = []):
        self.cachedDics={}

    def createOutDicCols(self, delays,sinaods,sinaoas,Nt,Nd,Na):
        vout=np.empty(shape=(Nt*Nd*Na,len(delays)),dtype=np.cdouble)
        for i in range(len(delays)):
            vt=np.fft.fft(np.sinc( np.arange(0.0,Nt,1.0).reshape(Nt,1,1) - delays[i] ),axis=0)/np.sqrt(1.0*Nt)
            va=np.exp( -1j*np.pi * np.arange(0.0,Na,1.0).reshape(1,Na,1) * sinaoas[i] ) /np.sqrt(1.0*Na)
            vd=np.exp( -1j*np.pi * np.arange(0.0,Nd,1.0).reshape(1,1,Nd) * sinaods[i] ) /np.sqrt(1.0*Nd)
            vall=(vt*va*vd)
            vall=vall/np.sqrt(np.sum(np.abs(vall)**2)) #if vectors are not normalized, correlation-comparison is not valid
            vout[:,i]=vall.reshape((Nt*Nd*Na,))
        return( vout )
    def createDic(self, Nt,Nd,Na,Xt,Xd,Xa):
        Tdic=np.arange(0.0,Nt,1.0/Xt)
        Ddic=np.arange(-1.0,1.0,2.0/Nd/Xd)
        Adic=np.arange(-1.0,1.0,2.0/Na/Xa)
        outputDic=np.empty(shape=(Nt*Nd*Na,np.size(Tdic)*np.size(Ddic)*np.size(Adic)),dtype=np.cdouble)
#        ctr=0
#        for t in range(np.size(Tdic)):
#            for d in  range(np.size(Ddic)):
#                for a in  range(np.size(Adic)):
#                    outputDic[:,ctr]=self.createOutDicCols([Tdic[t]],[Ddic[d]],[Adic[a]],Nt,Nd,Na).reshape((Nt*Nd*Na,))
#                    ctr+=1
        outputDic=self.createOutDicCols(
                np.tile( Tdic.reshape(np.size(Tdic),1,1) , (1,np.size(Ddic),np.size(Adic)) ).reshape((np.size(Tdic)*np.size(Ddic)*np.size(Adic),)),
                np.tile( Ddic.reshape(1,np.size(Ddic),1) , (np.size(Tdic),1,np.size(Adic)) ).reshape((np.size(Tdic)*np.size(Ddic)*np.size(Adic),)),
                np.tile( Adic.reshape(1,1,np.size(Adic)) , (np.size(Tdic),np.size(Ddic),1) ).reshape((np.size(Tdic)*np.size(Ddic)*np.size(Adic),)),
                Nt,Nd,Na)
        result = OMPInfoSet(None,Tdic,Ddic,Adic,{},outputDic)
        self.cachedDics[ (Nt,Nd,Na,Xt,Xd,Xa) ] = result
        return(result)
    def createObsDicCols(self, vout,vp,wp ):
        Ncols=np.shape(vout)[1]
        Nt=np.shape(vp)[0]
        Nxp=np.shape(vp)[1]
        Nd=np.shape(vp)[2]
#        Nrft=np.shape(vp)[3]
        Nrfr=np.shape(wp)[2]
        Na=np.shape(wp)[3]
        vresult=np.empty(shape=(Nt*Nxp*Nrfr,Ncols),dtype=np.cdouble)
        for c in range(Ncols):
#            vtest=vout[:,c].reshape(Nt,1,Na,Nd)
#            vref=np.empty(shape=(Nt,Nxp,Nrfr),dtype=np.cdouble)
#            for k in range(0,Nt):
#                for p in range(0,Nxp):
#                    vref[k,p,:]=np.matmul(wp[k,p,:,:] , np.sum(np.matmul( vtest[k,:,:] ,vp[k,p,:,:]),axis=1,keepdims=True) ).reshape((Nrfr,))
            vref=np.matmul( wp,  np.sum( np.matmul( vout[:,c].reshape(Nt,1,Na,Nd) ,vp) ,axis=3,keepdims=True) )
            vresult[:,c]= vref.reshape((Nt*Nxp*Nrfr,))
        return( vresult )
    def createObsDic(self, pilotsID,vp,wp,Xt,Xd,Xa):
        Nt=np.shape(vp)[0]
#        Nxp=np.shape(vp)[1]
        Nd=np.shape(vp)[2]
#        Nrft=np.shape(vp)[3]
#        Nrfr=np.shape(wp)[2]
        Na=np.shape(wp)[3]
        if (Nt,Nd,Na,Xt,Xd,Xa) not in self.cachedDics:
            paramDic=self.createDic(Nt,Nd,Na,Xt,Xd,Xa)
        else:
            paramDic = self.cachedDics[(Nt,Nd,Na,Xt,Xd,Xa)]
        observDic=self.createObsDicCols( paramDic.outputs ,vp,wp)
        self.cachedDics[(Nt,Nd,Na,Xt,Xd,Xa)].observations[pilotsID]=observDic
        return(observDic)
        
        
    def create2DObsDicCol(self, vout,vp,wp ):
        Ncols=np.shape(vout)[1]
        K=np.shape(vp)[0]
        Nxp=np.shape(vp)[1]
        Nd=np.shape(vp)[2]
#        Nrft=np.shape(vp)[3]
        Nrfr=np.shape(wp)[2]
        Na=np.shape(wp)[3]
        vresult=np.empty(shape=(K*Nxp*Nrfr,Ncols),dtype=np.cdouble)
        for c in range(Ncols):
            vref=np.matmul( wp,  np.sum( np.matmul( vout[:,c].reshape(1,1,Na,Nd) ,vp) ,axis=3,keepdims=True) )
#            vresult[:,c]= np.fft.ifft(vref,K,axis=0).reshape((K*Nxp*Nrfr,))*np.sqrt(Nt)
            vresult[:,c]= vref.reshape((K*Nxp*Nrfr,))/np.sqrt(K)
        return( vresult )
        
    def create2DObsDic(self, pilotsID,vp,wp,Xt,Xd,Xa):
        Nt=np.shape(vp)[0]
#        Nxp=np.shape(vp)[1]
        Nd=np.shape(vp)[2]
#        Nrft=np.shape(vp)[3]
#        Nrfr=np.shape(wp)[2]
        Na=np.shape(wp)[3]
        if (Nt,Nd,Na,Xt,Xd,Xa) not in self.cachedDics:
            paramDic=self.createDic(1,Nd,Na,1.0,Xd,Xa)
        else:
            paramDic = self.cachedDics[(1,Nd,Na,1.0,Xd,Xa)]
        observDic=self.create2DObsDicCol( paramDic.outputs ,vp,wp)
        self.cachedDics[(1,Nd,Na,1.0,Xd,Xa)].observations[pilotsID]=observDic
        return(observDic)
        
    def freeCacheOfPilot(self,pilotsID,Nt,Nd,Na,Xt,Xd,Xa):
        if (Nt,Nd,Na,Xt,Xd,Xa) in self.cachedDics:
            if pilotsID in self.cachedDics[(Nt,Nd,Na,Xt,Xd,Xa)].observations:
                self.cachedDics[(Nt,Nd,Na,Xt,Xd,Xa)].observations.pop(pilotsID)  
                
    def OMPBR(self,v,xi,pilotsID,vp,wp, Xt=1.0, Xd=1.0, Xa=1.0, Xmu=1.0, accelDel = False):
        if (Xt!=int(Xt)):
        # if (Xt!=1.0):
            accelDel = False
        Nt=np.shape(v)[0]
        Nxp=np.shape(vp)[1]
        Nd=np.shape(vp)[2]
#        Nrft=np.shape(vp)[3]
        Nrfr=np.shape(wp)[2]
        Na=np.shape(wp)[3]
#        if (accelDel): #use FFTs and only antenna dictionary
#            vflat=np.fft.ifft(v,axis=0)
#        else:
#            vflat=v.reshape(Nt*Nxp*Nrfr,1)
        vflat=v.reshape(Nt*Nxp*Nrfr,1)
        r=vflat
        et=1j*np.zeros(np.shape(r))
    
        if (accelDel): #use FFTs and only antenna dictionary
            if (1,Nd,Na,1.0,Xd,Xa) not in self.cachedDics:
                dicParams=self.createDic(1,Nd,Na,1.0,Xd,Xa)
            else:
                dicParams = self.cachedDics[(1,Nd,Na,1.0,Xd,Xa)]
            if pilotsID not in dicParams.observations:
                observDic=self.create2DObsDic(pilotsID,vp,wp,1.0,Xd,Xa)
            else:
                observDic=dicParams.observations[pilotsID]
            Tdic=np.arange(0.0,Nt,1.0/Xt)
            Ddic=dicParams.AoDs
            Adic=dicParams.AoAs
        else:
            if (Nt,Nd,Na,Xt,Xd,Xa) not in self.cachedDics:
                dicParams=self.createDic(Nt,Nd,Na,Xt,Xd,Xa)
            else:
                dicParams = self.cachedDics[(Nt,Nd,Na,Xt,Xd,Xa)]
            if pilotsID not in dicParams.observations:
                observDic=self.createObsDic(pilotsID,vp,wp,Xt,Xd,Xa)
            else:
                observDic=dicParams.observations[pilotsID]                
            Tdic=dicParams.delays
            Ddic=dicParams.AoDs
            Adic=dicParams.AoAs
#        outputDic=dicParams.outputs 
        observDicConj=observDic.transpose().conj()#cache this value for speed
    
        Rsupp=np.zeros(shape=(Nt*Nxp*Nrfr,Nt*Nxp*Nrfr),dtype=np.complex)
        delay_supp=np.zeros(Nt*Nxp*Nrfr)
        aod_supp=np.zeros(Nt*Nxp*Nrfr)
        aoa_supp=np.zeros(Nt*Nxp*Nrfr)
        #Isupp=OMPInfoSet([],[],[],[],[],[])
        et=np.empty(shape=np.shape(r))
        ctr=0        
        if Xmu>1.0:#pregenerate this only once before the loop
            boolTable=np.array([ [(x%(2**(n+1)))//(2**n) for n in range(3)] for x in range(8)],dtype=np.double) #[0,0,0],[1,0,0],[0,1,0]...[1,1,1]
            muTableInit=boolTable-.5 #[-.5,-.5,-.5] ... [.5,.5,.5]
        while ((ctr<Nt*Nxp*Nrfr) and (np.sum(np.abs(r)**2)>xi)) or ctr==0:
            if (accelDel):
                Call=(observDicConj*r.T).reshape(np.size(Ddic)*np.size(Adic),Nt,Nxp*Nrfr)
                c=np.sum(np.fft.ifft(Call,int(Nt*Xt),axis=1)*Nt*Xt,axis=2).T.reshape(-1)
#                c_aux=self.cachedDics[(Nt,Nd,Na,1.0,Xd,Xa)].observations[pilotsID].transpose().conj()@r
#                print(np.all(np.isclose(c_aux.T,c)))
            else:    
                c=np.matmul( observDicConj , r )
            ind=np.argmax(np.abs(c))
            t=int(np.floor(ind/np.size(Ddic)/np.size(Adic)))
            d=int(np.mod(np.floor(ind/np.size(Adic)),np.size(Ddic)))
            a=int(np.mod(ind,np.size(Adic)))
            # print((t,d,a))
            if Xmu>1.0:
                muTable=muTableInit
                mid_mu=np.array([0,0,0],dtype=np.double)
                while ( np.any(np.abs(mid_mu-muTable)>(.5/Xmu)) ):
                    vall=self.createOutDicCols( Tdic[t] + muTable[:,0]/Xt , np.mod( Ddic[d] - muTable[:,2]*2.0/Nd/Xd +1,2)-1 , np.mod( Adic[a] - muTable[:,1]*2.0/Na/Xa +1,2)-1 ,Nt,Nd,Na)
                    vref=self.createObsDicCols(vall,vp,wp)
                    corr_mu=np.matmul( vref.transpose().conj() , r )
                    bestInd=np.argmax(np.abs(corr_mu))
                    best_mu=muTable[bestInd,:]
                    muTable=mid_mu*(1-boolTable)+best_mu*boolTable #[mid_mu,mid_mu,mid_mu] ... [best_mu,best_mu,best_mu]
                    mid_mu=np.mean(muTable,axis=0)
                mu=mid_mu
                vall=self.createOutDicCols( [Tdic[t] + mu[0]/Xt] , [Ddic[d] -mu[2]*2.0/Nd/Xd] , [Adic[a] -mu[1]*2.0/Na/Xa] ,Nt,Nd,Na)
                vref=self.createObsDicCols(vall,vp,wp)
                if ( np.abs(c[ind]) < np.abs(np.sum(vref.conj()*r)) ):
                    
#                    if (accelDel):
#                        Rsupp[:,ctr]    = (np.exp(2j*np.pi*np.arange(Nt)*Tdic[t]).reshape(Nt,1,1) * vref ).reshape(-1,1) 
#                    else:                                       
                    Rsupp[:,ctr]    = vref.reshape((Nt*Nxp*Nrfr,))
#                    Rsupp[:,ctr]    = vref.reshape((Nt*Nxp*Nrfr,))
                    delay_supp[ctr] = Tdic[t] + mu[0]/Xt
                    aod_supp[ctr]   = np.mod( Ddic[d] -mu[2]*2.0/Nd/Xd +1,2)-1
                    aoa_supp[ctr]   = np.mod( Adic[a] -mu[1]*2.0/Na/Xa +1,2)-1
#                    Rsupp=np.concatenate(( Rsupp, vref ), axis=1)
#                    Isupp.delays      .append( Tdic[t] + mu[0]/Xt )
#                    Isupp.AoDs        .append( Ddic[d] -mu[2]*2.0/Nd/Xd )
#                    Isupp.AoAs        .append( Adic[a] -mu[1]*2.0/Na/Xa )                
                else:
                    # print("Autopatched correlation decrease")                    
                    if (accelDel):
                        angle_ind = a+np.size(Adic)*d
                        Rsupp[:,ctr]  = (np.exp(-2j*np.pi*np.arange(0,1,1/Nt)*Tdic[t]).reshape(Nt,1,1)*observDic[:,angle_ind].reshape(Nt,Nxp,Nrfr) ).reshape(-1)                    
                    else:                    
                        Rsupp[:,ctr]    = observDic[:,ind]
                    delay_supp[ctr] = Tdic[t]
                    aod_supp[ctr]   = Ddic[d]
                    aoa_supp[ctr]   = Adic[a]
#                    Rsupp=np.concatenate(( Rsupp, observDic[:,ind].reshape(Nt*Nxp*Nrfr,1) ), axis=1)
#                    Isupp.delays      .append( Tdic[t] )
#                    Isupp.AoDs        .append( Ddic[d] )
#                    Isupp.AoAs        .append( Adic[a] )
            else:      
                if (accelDel):
                    angle_ind = a+np.size(Adic)*d
                    # vt=np.fft.fft(np.sinc( np.arange(0.0,Nt,1.0).reshape(Nt,1,1) - Tdic[t] ),axis=0)                
                    # Rsupp[:,ctr]  = (vt*observDic[:,angle_ind].reshape(Nt,Nxp,Nrfr) ).reshape(-1)                    
                    Rsupp[:,ctr]  = (np.exp(-2j*np.pi*np.arange(0,1,1/Nt)*Tdic[t]).reshape(Nt,1,1)*observDic[:,angle_ind].reshape(Nt,Nxp,Nrfr) ).reshape(-1)                    
                else:            
                    Rsupp[:,ctr]  = observDic[:,ind]        
                delay_supp[ctr] = Tdic[t]
                aod_supp[ctr] = Ddic[d]
                aoa_supp[ctr] = Adic[a]
#            et=np.matmul(Rsupp, np.linalg.lstsq(Rsupp,vflat,rcond=None)[0] )
            vflat_proj,_,_,_=np.linalg.lstsq(Rsupp[:,0:ctr+1],vflat,rcond=None)
#            vflat_proj=self.lsqNumbaWrapper(Rsupp,vflat)
            et=np.matmul(Rsupp[:,0:ctr+1], vflat_proj )
            r=vflat-et
#            print(np.sum(np.abs(r)**2))
#            print('OMPBR %s ctr %d, |r|²= %f'%("accel" if (accelDel) else "normal",ctr,np.sum(np.abs(r)**2)))
            ctr=ctr+1
#        print('OMPBR ctr %d'%ctr)        
        Isupp=OMPInfoSet(vflat_proj,delay_supp[0:ctr] , np.arcsin(np.clip( aod_supp[0:ctr], -1,1)) , np.arcsin(np.clip( aoa_supp[0:ctr], -1,1)) , Rsupp[:,0:ctr] , self.createOutDicCols( delay_supp[0:ctr] , aod_supp[0:ctr] , aoa_supp[0:ctr] , Nt,Nd,Na) )
        hest=np.matmul(Isupp.outputs, vflat_proj )
        return( hest.reshape(Nt,Na,Nd), Isupp )
        
    def Shrinkage(self,v,shrinkageParams,Nit,pilotsID,vp,wp, Xt=1.0, Xd=1.0, Xa=1.0, shrinkageAlg = "ISTA"):
        Nt=np.shape(v)[0]
        Nxp=np.shape(vp)[1]
        Nd=np.shape(vp)[2]
#        Nrft=np.shape(vp)[3]
        Nrfr=np.shape(wp)[2]
        Na=np.shape(wp)[3]
        vflat=v.reshape(Nt*Nxp*Nrfr,1)
        r=vflat
    
        if (Nt,Nd,Na,Xt,Xd,Xa) not in self.cachedDics:
            dicParams=self.createDic(Nt,Nd,Na,Xt,Xd,Xa)
        else:
            dicParams = self.cachedDics[(Nt,Nd,Na,Xt,Xd,Xa)]
        if pilotsID not in dicParams.observations:
            observDic=self.createObsDic(pilotsID,vp,wp,Xt,Xd,Xa)
        else:
            observDic=dicParams.observations[pilotsID]
    
        Tdic=dicParams.delays
        Ddic=dicParams.AoDs
        Adic=dicParams.AoAs
        outputDic=dicParams.outputs
        
#        print("max beta %f"%(1.0/np.sum(np.abs(outputDic)**2)))
    
        Rsupp=np.empty(shape=(Nt*Nxp*Nrfr,0))
        Isupp=OMPInfoSet([],[],[],[],[])
        et=1j*np.zeros((np.shape(observDic)[1],1))
        if  shrinkageAlg == "FISTA":
            old_et=et
        if  shrinkageAlg == "VAMP":
            A=observDic
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
                c=et+beta*np.matmul( observDic.transpose().conj() , r )+(et-old_et)*(itctr-2.0)/(itctr+1)
                old_et=et
                et=np.exp(1j*np.angle(c))*np.maximum(np.abs(c)-lamb,0)
                r=vflat-np.matmul(observDic,et)
            elif shrinkageAlg == "ISTA":
                lamb=shrinkageParams[0]
                beta=shrinkageParams[1]
                c=et+beta*np.matmul( observDic.transpose().conj() , r )
                et=np.exp(1j*np.angle(c))*np.maximum(np.abs(c)-lamb,0)
                r=vflat-np.matmul(observDic,et)
            elif  shrinkageAlg == "AMP":
                alpha=shrinkageParams[0]
                bt=np.sum(np.abs(et)>0)/(Nt*Nxp*Nrfr)
                c=et+np.matmul( observDic.transpose().conj() , r )
                lamb = alpha*np.sqrt(np.sum(np.abs(r)**2)/(Nt*Nxp*Nrfr))
                et=np.exp(1j*np.angle(c))*np.maximum(np.abs(c)-lamb,0)
                old_r=r
                r=vflat-np.matmul(observDic,et)+bt*old_r
            elif  shrinkageAlg == "VAMP":
                #LMMSE stage
                M2=sigma2w/sigma2t_mmse*Vh
#                M3=np.diag(S**2+sigma2w/sigma2t_mmse)
#                if np.any(np.isnan(np.matmul(M1,vflat)+np.matmul(M2,r_mmse))) or  np.any(np.isinf(np.matmul(M1,vflat)+np.matmul(M2,r_mmse))):
#                    print("aqui")
#                et_mmse=np.matmul(Vh.T.conj(),np.linalg.lstsq(M3,np.matmul(M1,vflat)+np.matmul(M2,r_mmse),rcond=None)[0])
                M3_diagInv=( 1/(S**2+sigma2w/sigma2t_mmse) ).reshape((Nt*Nd*Na,1))
                et_mmse=np.matmul(Vh.T.conj(), M3_diagInv * np.matmul(M1,vflat)+np.matmul(M2,r_mmse) )
                vt_mmse=np.sum(1.0/((S**2)*sigma2t_mmse/sigma2w+1))/(Nt*Nd*Na)
                r=et_mmse-vt_mmse*r_mmse//(1-vt_mmse)
                sigma2t=sigma2t_mmse*vt_mmse/(1-vt_mmse)
                #Shrinkage stage
                lamb=alpha*np.sqrt(sigma2t/(Nt*Nxp*Nrfr))
                et=np.exp(1j*np.angle(r))*np.maximum(np.abs(r)-lamb,0)
                vt=np.sum(np.abs(et)>0)/(Nt*Nxp*Nrfr)
                r_mmse=(et-vt*r)/(1-vt)
                sigma2t_mmse=sigma2t*vt/(1-vt)
            else:
                print("Unknown algoritm")
                return((0,0))
                    
#            et=np.matmul(Rsupp, np.linalg.lstsq(Rsupp,vflat,rcond=None)[0] )
        lInds=np.where(et>0)[0]        
        Rsupp=np.empty(shape=(Nt*Nxp*Nrfr,0))
        Isupp=OMPInfoSet([],[],[],[],[])
        for ind in lInds:
            t=int(np.floor(ind/np.size(Ddic)/np.size(Adic)))
            d=int(np.mod(np.floor(ind/np.size(Adic)),np.size(Ddic)))
            a=int(np.mod(ind,np.size(Adic)))
            Rsupp=np.concatenate(( Rsupp, observDic[:,ind].reshape(Nt*Nxp*Nrfr,1) ), axis=1)
            Isupp.delays      .append( Tdic[t] )
            Isupp.AoDs        .append( np.arcsin(Ddic[d]) )
            Isupp.AoAs        .append( np.arcsin(Adic[a]) )
        Isupp=OMPInfoSet(Isupp.delays , Isupp.AoDs , Isupp.AoAs , Rsupp , self.createOutDicCols( Isupp.delays , Isupp.AoDs , Isupp.AoAs ,Nt,Nd,Na) )
        hest=np.matmul(outputDic, et )
        return( hest.reshape(Nt,Na,Nd), Isupp )
            