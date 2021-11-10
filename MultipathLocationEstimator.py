#!/usr/bin/python

import numpy as np
import scipy.optimize as opt

class MultipathLocationEstimator:
    
    def __init__(self,Npoint=100,Nref=10,Ndiv=2,RootMethod='lm'):
        self.NLinePointsPerIteration=Npoint
        self.NLineRefinementIterations=Nref
        self.NLineRefinementDivision=Ndiv
        self.RootMethod=RootMethod
        self.c=3e8
        
    def computePosFrom3PathsKnownPhi0(self,AoD,AoA,dels,phi0_est):
        tgD = np.tan(AoD)
        tgA = np.tan(np.pi-AoA-phi0_est)
        siD = np.sin(AoD)
        siA = np.sin(np.pi-AoA-phi0_est)
        coD = np.cos(AoD)
        coA = np.cos(np.pi-AoA-phi0_est)
        
        T=(1/tgD+1/tgA)
        S=(1/siD+1/siA)
        P=S/T
        Q=P/tgA-1/siA
#        P=(siD+siA)/(coD*siA+coA*siD)
#        Q=((siA-coD*tgA)/(coD*coA+siD*siA))
        Dl = dels*self.c
        
        Idp=(Dl[0:-1]-Dl[1:])/(P[...,0:-1]-P[...,1:])
        Slp=(Q[...,0:-1]-Q[...,1:])/(P[...,0:-1]-P[...,1:])
        
        y0=(Idp[...,0:-1]-Idp[...,1:])/(Slp[...,0:-1]-Slp[...,1:])
        x0=Idp[...,0:-1]-y0*Slp[...,0:-1]
        
        l0Err=x0*P[...,0:-2]+y0*Q[...,0:-2]-Dl[0:-2]
        l0=np.sqrt(x0**2+y0**2)
        tauE=(l0-l0Err)/self.c
        
        return(x0,y0,tauE)
        
    def computeAllPathsWithParams(self,AoD,AoA,dels,x0,y0,phi0_est):
        tgD = np.tan(AoD)
        tgA = np.tan(np.pi-AoA-phi0_est)
        
#        T=(1/tgD+1/tgA)        
#        vy=(x0+y0/tgA)/T
        vy=np.where(tgD!=0, (tgA*x0+y0)/(tgA/tgD+1) , 0)
        vx=np.where(tgD!=0, vy/tgD, x0+y0/tgA)
            
        return(vx,vy)
        
    def computeAllPathsLinear(self,AoD,AoA,dels,phi0_est):
        tgD = np.tan(AoD)
        tgA = np.tan(np.pi-AoA-phi0_est)
        siD = np.sin(AoD)
        siA = np.sin(np.pi-AoA-phi0_est)
        coD = np.cos(AoD)
        coA = np.cos(np.pi-AoA-phi0_est)
        
#        T=(1/tgD+1/tgA)
#        S=(1/siD+1/siA)
#        P=S/T
        P=(siD+siA)/(coD*siA+coA*siD)
        P[np.isnan(P)]=1
        Q=P/tgA-1/siA
        Dl = dels*self.c
        
        result=np.linalg.lstsq(np.column_stack([P,Q,-np.ones_like(P)]),Dl,rcond=None)
        
        (x0est,y0est,l0err)=result[0]
        
        l0est=np.sqrt(x0est**2+y0est**2)
        tauEest=(l0est-l0err)/self.c
        
#        vyest=(x0est+y0est/tgA)/T
        vyest=np.where(tgD!=0, (tgA*x0est+y0est)/(tgA/tgD+1), 0)
        vxest=np.where(tgD!=0, vyest/tgD, x0est+y0est/tgA)
            
        return(x0est,y0est,tauEest,vxest,vyest)
          
    def feval_wrapper_AllPathsLinear_drop1(self,x,AoD,AoA,dels):
        Npath=AoD.size
        x0all=np.zeros(Npath)
        y0all=np.zeros(Npath)
        tauEall=np.zeros(Npath)
        for gr in range(Npath):
            (x0all[gr],y0all[gr],tauEall[gr],_,_)=self.computeAllPathsLinear(AoD[np.arange(Npath)!=gr],AoA[np.arange(Npath)!=gr],dels[np.arange(Npath)!=gr],x)
        return(np.sum(np.abs(x0all-np.mean(x0all,x0all.ndim-1,keepdims=True))**2+np.abs(y0all-np.mean(y0all,x0all.ndim-1,keepdims=True))**2+np.abs(self.c*tauEall-np.mean(self.c*tauEall,x0all.ndim-1,keepdims=True))**2,x0all.ndim-1))
    
    def feval_wrapper_AllPathsLinear_random(self,x,AoD,AoA,dels):
        Npath=AoD.size
        Nlines=5
        x0all=np.zeros(Nlines)
        y0all=np.zeros(Nlines)
        tauEall=np.zeros(Nlines)
        indices=np.random.choice(Npath,(Nlines,Npath//2))
        for gr in range(Nlines):
            (x0all[gr],y0all[gr],tauEall[gr],_,_)=self.computeAllPathsLinear(AoD[indices[gr,:]],AoA[indices[gr,:]],dels[indices[gr,:]],x)
        return(np.sum(np.abs(x0all-np.mean(x0all,x0all.ndim-1,keepdims=True))**2+np.abs(y0all-np.mean(y0all,x0all.ndim-1,keepdims=True))**2+np.abs(self.c*tauEall-np.mean(self.c*tauEall,x0all.ndim-1,keepdims=True))**2,x0all.ndim-1))
    
    def solvePhi0ForAllPaths_linear(self,AoD,AoA,dels):
        #coarse linear approximation for initialization
        init_phi0=self.bisectPhi0ForAllPaths(AoD,AoA,dels,Npoint=1000,Niter=1,Ndiv=2)
        res=opt.root(self.feval_wrapper_AllPathsLinear_drop1,x0=init_phi0,args=(AoD,AoA,dels),method=self.RootMethod)
        if not res.success:
#            print("Attempting to correct initialization problem")
            niter=0 
            while (not res.success) and (niter<1000):
                res=opt.root(self.feval_wrapper_AllPathsLinear_drop1,x0=2*np.pi*np.random.rand(1),args=(AoD,AoA,dels),method=self.RootMethod)
                success=res.success
                niter+=1
#            print("Final Niter %d"%niter)
        if res.success:
            return (res.x)
        else:
            print("ERROR: Phi0 root not found")
            return (np.array(0.0))
    
    def bisectPhi0ForAllPaths(self,AoD,AoA,dels,Npoint=None,Niter=None,Ndiv=None):        
        philow=0
        phihigh=2*np.pi
#        Ncurves=np.size(AoD)-2
        if not Npoint:
            Npoint=self.NLinePointsPerIteration
        if not Niter:
            Niter=self.NLineRefinementIterations
        if not Ndiv:
            Ndiv=self.NLineRefinementDivision
        for nit in range(Niter):
            interval=np.linspace(philow,phihigh,Npoint).reshape(-1,1)
#            x0all=np.zeros((Ncurves,Npoint))
#            y0all=np.zeros((Ncurves,Npoint))
#            for npath in range(Ncurves):
#                (x0all[npath,:],y0all[npath,:])= self.computePosFrom3PathsKnownPhi0(AoD[npath:npath+3],AoA[npath:npath+3],dels[npath:npath+3],interval)
            (x0all,y0all,tauEall)=self.computePosFrom3PathsKnownPhi0(AoD,AoA,dels,interval)
            dist=np.sum(np.abs(x0all-np.mean(x0all,x0all.ndim-1,keepdims=True))**2+np.abs(y0all-np.mean(y0all,x0all.ndim-1,keepdims=True))**2+np.abs(self.c*tauEall-np.mean(self.c*tauEall,x0all.ndim-1,keepdims=True))**2,x0all.ndim-1)
            distint=np.argmin(dist)
            philow=interval[distint]-np.pi/(Ndiv**nit)
            phihigh=interval[distint]+np.pi/(Ndiv**nit)
#        if (dist[distint]>1):
#            print("ERROR: phi0 recovery algorithm converged loosely phi0: %.2f d: %.2f"%((np.mod(interval[distint],2*np.pi),dist[distint])))
        return(interval[distint])
    
    def feval_wrapper_3PathPosFun(self,x,AoD,AoA,dels):
        (x0all,y0all,tauEall)=self.computePosFrom3PathsKnownPhi0(AoD,AoA,dels,np.asarray(x))
        return(np.sum(np.abs(x0all-np.mean(x0all,x0all.ndim-1,keepdims=True))**2+np.abs(y0all-np.mean(y0all,x0all.ndim-1,keepdims=True))**2+np.abs(self.c*tauEall-np.mean(self.c*tauEall,x0all.ndim-1,keepdims=True))**2,x0all.ndim-1))
        
    def solvePhi0ForAllPaths(self,AoD,AoA,dels):
        #coarse linear approximation for initialization
        init_phi0=self.bisectPhi0ForAllPaths(AoD,AoA,dels,Npoint=1000,Niter=1,Ndiv=2)
        res=opt.root(self.feval_wrapper_3PathPosFun,x0=init_phi0,args=(AoD,AoA,dels),method=self.RootMethod)
        if not res.success:
#            print("Attempting to correct initialization problem")
            niter=0 
            while (not res.success) and (niter<1000):
                res=opt.root(self.feval_wrapper_3PathPosFun,x0=2*np.pi*np.random.rand(1),args=(AoD,AoA,dels),method=self.RootMethod)
                niter+=1
#            print("Final Niter %d"%niter)
        if res.success:
            return (res.x)
        else:
            print("ERROR: Phi0 root not found")
            return (np.array(0.0))
   
    def computeAllLocationsFromPaths(self,AoD,AoA,dels,method='fsolve'):
        if method=='fsolve':
            phi0_est=self.solvePhi0ForAllPaths(AoD,AoA,dels)
        elif method=='bisec':
            phi0_est=self.bisectPhi0ForAllPaths(AoD,AoA,dels)
        elif method=='fsolve_linear':
            phi0_est=self.solvePhi0ForAllPaths_linear(AoD,AoA,dels)
        else:
            print("unsupported method")
            return(None)
        #at this point any 3 paths can be used
        if method=='fsolve_linear':
            (x0,y0,tauerr,vx,vy)= self.computeAllPathsLinear(AoD,AoA,dels,phi0_est)
        else:
            (x0all,y0all,tauEall)= self.computePosFrom3PathsKnownPhi0(AoD,AoA,dels,phi0_est.reshape(-1,1))
            x0=np.mean(x0all,1)
            y0=np.mean(y0all,1)
            (vx,vy)= self.computeAllPathsWithParams(AoD,AoA,dels,x0,y0,phi0_est)
        return(phi0_est,x0,y0,vx,vy)