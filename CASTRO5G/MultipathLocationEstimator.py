#!/usr/bin/python

import numpy as np
import scipy.optimize as opt
import argparse
import pandas as pd
from collections.abc import Iterable
from tqdm import tqdm

class MultipathLocationEstimator:
    """Class used to calculate 5G UE (User Equipment) location in 2D and 3D. 
    
    The main goal of the MultipathLocationEstimator class is try to recover the UE position trigonometrically, defined as 
    (d0x, d0y), estimating the value of user offset orientation (phi_0) from the knowledge of the set of the Angles of Departure 
    (AoD), Angles of Arrival (DAoA) and delays introduced by the multipath channels. 
    This class includes several algorithms to obtain the value of phi_0, from a brute force searching method to a 
    non-linear system equation solver. Also, different methods have been included to find the UE position dealing with clock
    and orientation error.
    """
    
    tableMethodsScipyRoot=[
        'hybr',
        'lm',
        'broyden1',
        'broyden2',
        'anderson',
        'linearmixing',
        'diagbroyden',
        'excitingmixing',
        'krylov',
        'df-sane'
        ]
    tableMethodsScipyMinimize=[
        'Nelder-Mead',
        'Powell',
        'CG',
        'BFGS',
        'Newton-CG',
        'L-BFGS-B',
        'TNC',
        'COBYLA',
        'COBYQA',
        'SLSQP',
        'trust-constr',
        'dogleg',
        'trust-ncg',
        'trust-exact',
        'trust-krylov',
        ]
   
    def __init__(self, orientationMethod='lm', nPoint=100, groupMethod='drop1', disableTQDM=True):
        """ MultipathLocationEstimator constructor.
        
        Parameters
        ----------

        nPoint : int, optional
            Total points in the search range for brute-force orientation finding.

        orientationMethod: str, optional
            Type of solver algorithm for non-linear minimum mean squared error orientation finding.
            ** brute : Brute force search with nPoint uniform samples of the range 0..2π
            ** lm    : Levenberg–Marquardt algorithm for non-linear least squares provided by scipy.optimize.root()
            ** hybr  : Powell's dog leg method AKA "hybrid" method  provided by scipy.optimize.root(), an iterative
                       optimization algorithm for solving non-linear least squares problem.
            
        """
        self.c = 3e8
        self.orientationMethod = orientationMethod
        #default args
        self.nPoint = nPoint
        self.groupMethod = groupMethod
        self.disableTQDM = disableTQDM
    
    def computeAllPathsV1(self, AoD, DAoA, TDoA, AoA0_est):
        """Calculates the possible UE vector positions in 2D using the LS method.
        
        ----------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        AoD  : ndarray
            Azimuths of Departure of the NLOS ray propagation, measured in the BS from the positive x-axis in the 
            counter-clockwise sense.
            
        DAoA : ndarray
            Azimuths of Arrival of the NLOS ray propagation, measured in the UE from the positive x'-axis in the 
            counter-clockwise sense. The value of phi_0 can modify the orientation of the x-axis.
            
        TDoA : ndarray
            Time-diference of arrival introduced by the NLOS ray propagations. The absolute light travel time of
            first path is unknown, TDoAs are measured w.r.t. an arbitrary 'zero instant' of the receiver clock.
            
        AoA0_est: ndarray
            Offset AoA of the UE orientation in the counter-clockwise sense.

        Returns
        -------
        x0_est, y0_est : ndarray
            x,y-coordinates of the UE estimated position.
            
        tauEest : ndarray
            initial receiver clock offset measured as difference between the TDoA 'measurement zero' and the light
            travel time in the LoS direction l0 := √(x0_est² + y0_est²)
            
        x_est,y_est : ndarray
           x,y-coordinates of the scatterers' estimated positions.
        """
        from warnings import warn
        warn("This is the deprecated OLD V1 2D location algoritm of MultipathLocationEstimator. Please migrate your code to the joint 2D/3D implementation computeAllPaths(...)", DeprecationWarning, stacklevel=2)
        tgD = np.tan(AoD)
        tgA = np.tan(DAoA + AoA0_est)
        sinD = np.sin(AoD)
        sinA = np.sin(DAoA + AoA0_est)
        cosD = np.cos(AoD)
        cosA = np.cos(DAoA + AoA0_est)
        
        P = (sinD + sinA)/(cosD*sinA - sinD*cosA)
        P_mask = np.invert((np.isinf(P) + np.isnan(P)), dtype=bool)
        P = P*P_mask
        P[np.isnan(P)] = 1

        Q = (cosA + cosD)/(sinD*cosA - cosD*sinA)
        Q_mask = np.invert((np.isinf(Q) + np.isnan(Q)), dtype=bool)
        Q = Q*Q_mask
        Q[np.isnan(Q)] = 1

        Dl = TDoA*self.c
        
        result = np.linalg.lstsq(np.column_stack([P, Q, -np.ones_like(P)]), Dl, rcond=None)
        (d0xest, d0yest, l0err) = result[0]

        l0est = np.sqrt(d0xest**2 + d0yest**2)
        ToA0_est = (l0est - l0err)/self.c
        
        vyest = np.where(tgD!=0, (-tgA*d0xest + d0yest)/(-tgA/tgD + 1), 0)
        vxest = np.where(tgD!=0, vyest/tgD, d0xest - d0yest/tgA)
            
        return(d0xest, d0yest, ToA0_est, vxest, vyest)
    
    #TODO implement these functions from a common library in all classes
    def uVector(self,A,Z=None):
        """Computes unitary directional column vectors. 
        
        ----------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        A  : ndarray
            N Azimuths of the desired vectors
            
        Z : ndarray or None
            if None, the outputs are 2D row vectors, otherwise the N Zeniths of the desired vectors

        Returns
        -------
        D : ndarray
            2xN or 3xN matrix with row values given by
            [cos(A), sin(A)] if Z is None
            [cos(A)*sin(Z), sin(A)sin(Z), np.cos(Z)]
        """
        if Z is None:
            return( np.vstack([np.cos(A),np.sin(A)]) )
        else:
            return( np.vstack([np.cos(A)*np.sin(Z),np.sin(A)*np.sin(Z),np.cos(Z)]) )
    def angVector(self,v):
            AoA=np.arctan2(v[...,1],v[...,0])
            if v.shape[-1]>2:
                l=np.linalg.norm(v[...,0:2],axis=-1)
                ZoA=np.arctan2(l,v[...,2])#recall we use 3GPP Zenith angle convention
                return(np.array([AoA,ZoA]))
            else:
                return(AoA)
        
    def rMatrix2D(self,A):
        """Computes a 2D rotation matrix 
        
        ----------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        A  : float
            N Angle of rotation

        Returns
        -------
        R : 2x2
            Rotation matrix
        """
        return( np.array([
            [np.cos(A),-np.sin(A)],
            [np.sin(A),np.cos(A)]
            ]) )
    def rMatrix3D(self,A,ax=2):
        """Computes a 3D single-axis rotation matrix 
        
        ----------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        A  : float
            Angle of rotation
        ax  : int, optional, default = 2
            Axis were the rotation should take place

        Returns
        -------
        R : 3x3
            Rotation matrix that leaves axis 'ax' unchanged and rotates the
            other 2 axes
        """
        basis=self.rMatrix2D(A)
        R=np.insert(basis,ax,np.zeros(2),axis=0)
        R=np.insert(R,ax,np.zeros(3),axis=1)
        R[ax,ax]=1
        return(R)
    def rMatrix(self,A,Z=None,S=None):
        """Computes a 2D or 3D full rotation matrix 
        
        ----------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        A  : float
            Azimuth, "bearing" yaw of rotation measured counterclockwise from the X axis, around the Z axis
        Z  : float, optional, default = None
            Zenith, "downtilt" π/2 minius pitch of rotation measured downwards from the Z axis, around the "facing-to-A" horizontal axis. If None, a 2D matrix is returned
        S  : float, optional, default = None
            "Slant" roll of rotation measured form up from the Y axis, around the "facing-to-AZ" axis). If None, a 3D matrix rotated in A,Z is returned

        Returns
        -------
        R : 3x3
            3 Axis Rotation matrix with azimuth A, zenith Z and slant S
        """
        if Z is None:
            return( self.rMatrix2D(A) )
        else:
            E=np.pi/2-Z#The single-axis rotaiton matrix function is alwas positive, we must replace zenith with elevation
            if S is None:
                return( self.rMatrix3D(A,2)@self.rMatrix3D(E,1) )
            else:
                return( self.rMatrix3D(A,2)@self.rMatrix3D(E,1)@self.rMatrix3D(S,0) )
    def computeAllPaths(self, paths , rotation=None):       
        Npath=paths.shape[0]
        #TODO choose a library to provide uVector globally to the project
        mode3D = ('ZoD' in paths.columns) and ('DZoA' in paths.columns)
        if mode3D:
            DoD = self.uVector(paths.AoD,paths.ZoD).T
            if rotation is None:
                DoA = ( self.uVector(paths.DAoA,paths.DZoA) ).T
            elif isinstance(rotation, Iterable):
                Rm=self.rMatrix(*rotation) 
                DoA = ( Rm@self.uVector(paths.DAoA,paths.DZoA) ).T
            else:#if its not iterable it must be a number describing the AoA0
                DoA = ( self.uVector(paths.DAoA+rotation,paths.DZoA) ).T
        else:
            DoD = self.uVector(paths.AoD).T
            DoA = self.uVector(paths.DAoA+ (rotation if rotation is not None else 0) ).T
        C12= np.sum(-DoD*DoA,axis=1,keepdims=True)
        # print(C12)
        M=np.column_stack([(DoD-DoA)/(1+C12),-np.ones((Npath,1))])
        if np.any(C12[:,0]==-1):#fix the indetermination case DoD=-DoA
            M[C12[:,0]==-1,0:-1]=DoD[C12[:,0]==-1,:]
        
        result_est=np.linalg.lstsq(M, paths.TDoA*self.c, rcond=None)[0]
        d0_est = result_est[0:-1]
        l0err = result_est[-1]
        l0est = np.linalg.norm(d0_est)
        ToA0_est = (l0est - l0err)/self.c
        Vi=(DoD+C12*DoA)/(1-C12**2)
        liD_est=Vi@d0_est
        d_est=liD_est[:,None]*DoD
        return(d0_est,ToA0_est,d_est)
    
    def computeAllPathsV1wrap(self, paths, rotation=None):
        """Computes computeAllPathsV1 with the input-output convention of the new version
        
        ----------------------------------------------------------------------------------------------------------
        """
        AoA0_est = 0 if rotation is None else rotation
        d0xest, d0yest, ToA0_est, vxest, vyest = self.computeAllPathsV1(paths.AoD, paths.DAoA, paths.TDoA, AoA0_est)
        d0_est = np.array([d0xest,d0yest])
        d_est = np.vstack([vxest,vyest]).T
        return(d0_est, ToA0_est, d_est)
        
    def genKPathGroup(self, Npath, K=3):
        """Returns a Npath-K+1 x Npath boolean table representing paths belonging to K-path groups.
        
        Divides the set {1, 2, 3, . . ., Npath} into Npath-K groups of paths G1={1, 2, 3},
        G2={2, 3, 4} , G3{3, 4, 5}, . . ., Gm={Npath-2, Npath-1, Npath}. 
        
        E.g.:
        Npath = 5, K=3 the groups includes paths as:        
                                                G1 = {1,2,3}
                                                G2 = {2,3,4}
                                                G3 = {3,4,5}        
        For this examples, the table is generated as:
                                 |True  True  True False False|
                                 |False True  True  True False|
                                 |False False True  True  True|
                                 
        ---------------------------------------------------------------------------------------------------------

        Parameters
        ----------
        Npath  : int
            Total number of paths.

        Returns
        -------
        table_group : ndarray
            Boolean array that contains all the 3-path groups.
        """        
        table_group = np.empty((Npath-K+1, Npath), dtype=bool)

        for gr in range(Npath-K+1):
            path_indices = gr+np.arange(0,K).astype(int)
            table_group[gr,:] = np.isin(np.arange(Npath), path_indices)        
        
        return table_group

    def genDrop1Group(self, Npath):
        """Returns a Npath x Npath boolean table representing paths belonging to drop-1 groups.
        
        Divides the set {1, 2, 3, . . ., Npath} into Npath groups of paths G1={2, 3,...,Npath},
        G2={1,3,...,Npath}, G3={1,2,4,...,Npath}, . . . , Gm={1,...,Npath-2, Npath-1}. 
        E.g.:
        Npath = 5, the groups include paths as:      
                                    G1 = {2,3,4,5}     G4 = {1,2,3,5}
                                    G2 = {1,3,4,5}     G5 = {1,2,3,4}
                                    G3 = {1,2,4,5}       
        For this example, the table is generated as:
                                 |False  True  True  True  True|
                                 |True  False  True  True  True|
                                 |True  True  False  True  True|
                                 |True  True  True  False  True|
                                 |True  True  True  True  False|
                              
        ---------------------------------------------------------------------------------------------------------
                         
        Parameters
        ----------
        Npath  : int
            Total number of paths.

        Returns
        -------
        table_group : ndarray
            Boolean array that contains all the drop1 groups.     
        """
        table_group = np.empty((Npath, Npath), dtype=bool)

        for gr in range(Npath):
            table_group[gr,:] = np.isin(np.arange(Npath), [gr], invert=True)
        
        return table_group
    
    def genOrthoGroup(self, Npath, K=2):
        """Returns a K x Npath boolean table representing paths evenly split in K orthogonal groups.
                              
        -------------------------------------------------------------------------------------------
        """
        table_group = np.empty((K, Npath), dtype=bool)
    
        for gr in range(K):
            path_indices = gr*(Npath//(K+1))+np.arange(K).astype(int)
            table_group[gr,:] = np.isin(np.arange(Npath), path_indices)        
        
        return table_group
        
    def genRandomGroup(self, Npath, Ngroups, Nmembers):
        """Returns a Ngroup x Npath boolean table representing paths belonging to random groups.
        
        Divides the set {1, 2, 3, . . ., Npath} into Ngroup groups of Nmembers paths uniformly
        selected at random.
          
        ---------------------------------------------------------------------------------------------------------

        Parameters
        ----------
        Npath  : int
            Total number of paths.        
        Ngroup : int
            Number of groups to make
        Nmembers : int
            Number of members per group.

        Returns
        -------
        table_group : ndarray
            Boolean array that contains all the random groups.
        """
        table_group = np.empty((Ngroups, Npath), dtype=bool)

        for gr in range(Ngroups):
            path_indices = np.random.choice(Npath,Nmembers,replace=False)
            table_group[gr,:] = np.isin(np.arange(Npath), path_indices)        
        
        return table_group

    
    def locEvalByPathGroups(self, x, paths, groupMethod='drop1'):
        """Evaluates all linear location solutions obtained by multiple groups
        of paths, as a function of the rotation of the receiver.
        
        ---------------------------------------------------------------------------------------------------------
   
        Parameters
        ----------
        x: ndarray
            Receiver rotation.
            
        paths  : dataframe
            multipath components
            
        groupMethod : ndarray, optional
            Path grouping strategy.
            *** Options: 'drop1', '3-path', 'random'
            *** Default value is 'drop1'.
            
        Returns
        -------
        Returns the value of the mean square error (MSE) between all location solutions and
        the average        
                        sum_i sum_d ( d[i,d]-mean_j(d[j,d]) )²
        """
        Npath = paths.index.size
        if isinstance(groupMethod,str):
            if 'path' in groupMethod:
                table_group = self.genKPathGroup(Npath,K=int(groupMethod.strip('path')))    
            elif 'ortho' in groupMethod:
                table_group = self.genKPathGroup(Npath,K=int(groupMethod.strip('ortho')))      
            elif 'drop' in groupMethod:
                table_group = self.genDrop1Group(Npath)            
            elif groupMethod == 'random':
                Ngroups = Npath
                Nmembers = Npath//2
                table_group = self.genRandomGroup(Npath, Ngroups, Nmembers)
            else:
                raise(TypeError("MultipathLocationEstimator requires an explicit boolean group table or a known group method name in str"))
        else:
            table_group = groupMethod

        Ngroup = table_group.shape[0]
        mode3D = ('ZoD' in paths.columns) and ('DZoA' in paths.columns)    
        Ndim = 3 if mode3D else 2
        d0_all = np.zeros((Ngroup,Ndim))
        tauEall = np.zeros(Ngroup)
        for gr in range(Ngroup):
            # (d0_all[gr,:], tauEall[gr],_) = self.computeAllPathsV1wrap(paths[table_group[gr, :]], rotation=x)
            (d0_all[gr,:], tauEall[gr],_) = self.computeAllPaths(paths[table_group[gr, :]], rotation=x)
        v_all=np.concatenate([d0_all,self.c*tauEall[:,None]],axis=1)        
        return(v_all)
    
    def locMSEByPathGroups(self, x, paths, groupMethod='drop1'):
        """Evaluates the mean squared error (MSE) distance among all linear location solutions
        obtained by multiple groups of paths, as a function of the rotation of the receiver.
        """
        v_all=self.locEvalByPathGroups(x, paths,groupMethod)
        v_all-=np.mean(v_all,axis=0)
        return( np.sum( np.abs(v_all)**2 ) )
    
    def locDifByPathGroups(self, x, paths, groupMethod='drop1'):
        """Evaluates all pairs of differences of all linear location solutions
        obtained by multiple groups of paths, as a function of the rotation of the receiver.
        """
        v_all=self.locEvalByPathGroups(x, paths,groupMethod)
        # v_all=np.diff(v_all,axis=0).reshape(-1)
        # v_all=(v_all[1:,:]-np.mean(v_all,axis=0)).reshape(-1)
        # return( v_all )
        Ng=v_all.shape[0]
        Ndim=v_all.shape[1]
        v_big=np.zeros(Ng*(Ng-1)*Ndim)
        for n1 in range(Ng):
            for n2 in range(Ng-1):
                v_big[n2*Ng+n1:n2*Ng+n1+Ndim]=v_all[n1]-v_all[n2+1*(n2>=n1)]
        return( v_big )
    
    def locCheckAtRotation(self,rotation,paths,d0,tauE):
        DoD = self.uVector(paths.AoD,paths.ZoD).T
        Rm=self.rMatrix(*rotation)
        DDoA = self.uVector(paths.DAoA,paths.DZoA).T
        DoA=DDoA@Rm.T        
        C12= np.sum(-DoD*DoA,axis=1,keepdims=True)
        li=np.linalg.norm(d0)+tauE+paths.TDoA*self.c
        return( li*(1+C12)+DDoA@Rm.T@d0-DoD@d0 )       
        

    def bruteAoA0ByPathGroups(self, paths, nPoint=None, groupMethod='drop1'):
        """Estimates the value of the receiver Azimuth AoA0 by brute force by minimizing the
        mean squared distance between location solutions of multiple groups. Finds the best
        of nPoints between 0 and 2π
        """        
        if nPoint is None:
            nPoint = self.nPoint            
        interval = np.linspace(0,  2*np.pi, nPoint)
        MSE = np.zeros(nPoint)
        for n in tqdm(range(nPoint), desc='MultipathLocationEstimator brute force 2D rotation estimation' ,disable=self.disableTQDM):
            MSE[n] = self.locMSEByPathGroups(interval[n], paths, groupMethod)        
        distind = np.argmin(MSE)

        return(interval[distind])
    
    def brute3DRotByPathGroups(self, paths, nPoint=None, groupMethod='drop1'):
        """Estimates the value of the receiver Asimuth, Zenith and Spin by brute force by minimizing the
        mean squared distance between location solutions of multiple groups. Finds the best
        of nPoints^3 in the intervals [0,2π)x[0,π)x[0,2π) 
        """        
        if nPoint is None:
            nPoint = self.nPoint
        if isinstance(nPoint,tuple):
            dims=nPoint
        else:
            dims=(nPoint,nPoint,nPoint)
        intervalAoA0 = np.linspace(0,  2*np.pi, dims[0])
        intervalZoA0 = np.linspace(0,  np.pi, dims[1])
        intervalSoA0 = np.linspace(0,  2*np.pi, dims[2])
        
        MSE = np.zeros(dims)
        for n in tqdm(range(np.prod(dims)), desc='MultipathLocationEstimator brute force 3D rotation estimation' ,disable=self.disableTQDM, leave=False, position=1):
            n1,n2,n3=np.unravel_index(n,(dims))
            MSE[n1,n2,n3] = self.locMSEByPathGroups((intervalAoA0[n1],intervalZoA0[n2],intervalSoA0[n3]), paths, groupMethod)
        n1,n2,n3 = np.unravel_index(np.argmin(MSE),dims)

        return((intervalAoA0[n1],intervalZoA0[n2],intervalSoA0[n3]))
    
    def numericRot0ByPathGroups(self, paths, init_AoA0, groupMethod='drop1', orientationMethod='lm'):
        """Performs the estimation of the value of AoA0 using the scipy.optimize.root function.        
        The value of the estimated offset angle of the UE orientation is obtained by finding the zeros of the 
        minimum mean square error (MMSE) equation defined by feval_wrapper_AllPathsByGroups. For this purpose, it is used
        the method root() to find the solutions of this function.
        In this case, it is used the root method of scipy.optimize.root() to solve a non-linear equation with parameters.                  
        """
        if orientationMethod in self.tableMethodsScipyRoot:
            # res = opt.root(self.locMSEByPathGroups, x0=init_AoA0, args=(paths, groupMethod), method=orientationMethod)
            res = opt.root(self.locDifByPathGroups, x0=init_AoA0, args=(paths, groupMethod), method=orientationMethod)
        elif orientationMethod in self.tableMethodsScipyMinimize:
             #generally the 'lm' method above is recommended to minimize non-linear least squares, before the methods in the following
            res = opt.minimize(self.locMSEByPathGroups, x0=init_AoA0, args=(paths, groupMethod), method=orientationMethod)
        # if not res.success:
        #     print("Attempting to correct initialization problem")
        #     niter = 0 
        #     while (not res.success) and (niter<100):
        #         if len(init_AoA0)==3:
        #             random_init=np.random.rand(3)*np.pi*[2,1,2]
        #         else:
        #             random_init=2*np.pi*np.random.rand(len(init_AoA0))#should only be called with 1 but just in case
        #         res = opt.root(self.locMSEByPathGroups, x0=random_init, args=(paths, groupMethod), method=orientationMethod)
        #         niter += 1
        #print("Final Niter %d"%niter)
        # print(res)
        if res.success:
            if hasattr(res, 'cov_x'):
                return (res.x,res.cov_x)
            else:
                return (res.x,None)
        else:
            print("ERROR: AoA0 root not found")
            return (np.array(0.0),np.inf)
                

    def computeAllLocationsFromPaths(self, paths, orientationMethod=None, orientationMethodArgs={'groupMethod':'drop1','hintRotation':None}):
        """Performs the estimation of the phi_0 especified by the parameter method, and returns the position 
        of the UE for this angle.
            
        ---------------------------------------------------------------------------------------------------------

        Parameters
         ----------
        paths  : dataframe
            Containing at least the columns AoD, DAoA and TDoA
            Same as computeAllPaths                    
    
        orientationMethod: str, optional
            Overrides namesake global parameter.
            
        orientationMethodArgs : dic, optional
            dictionally containing key-value pairs defining the arguments of the orientation method
            ** groupMethod : ndarray, optional Path grouping strategy among 'drop1', '3-path', 'random'.
                             Applicable to methods lm, hybr and brute
            ** nPoint: number of points ot divide the range of search [0-2π] in brute force method.
            ** hintRotation: initialization point for non-linear solvers lm and hybr
            
        Returns
        -------
            
        d0 : ndarray
            x,y-coordinate of the possible position of the UE.
            
        d : ndarray
            x,y-coordinate of the reflectors' positions in the NLOS paths.
            
        rotation_est: ndarray
            Offset angle estimated of the UE orientation.
            
        rotation_cov: ndarray
            The inverse of the Hessian matrix of AoA0.
                        
        """
        mode3D = ('ZoD' in paths.columns) and ('DZoA' in paths.columns)
        themethod = orientationMethod if orientationMethod is not None else self.orientationMethod
        if themethod == 'brute':
            if "groupMethod" in orientationMethodArgs:
                groupMethod = orientationMethodArgs["groupMethod"]        
            else:
                groupMethod = self.groupMethod
            if "nPoint" in orientationMethodArgs:
                nPoint =  orientationMethodArgs["nPoint"]
            else:
                nPoint = self.nPoint
            if mode3D:
                rotation_est = self.brute3DRotByPathGroups(paths, nPoint, groupMethod)
            else:
                rotation_est = self.bruteAoA0ByPathGroups(paths, nPoint, groupMethod)                
            if isinstance(nPoint,tuple):
                rotation_cov = np.pi/nPoint[0]
            else:
                rotation_cov = np.pi/nPoint
        elif themethod in (self.tableMethodsScipyRoot+self.tableMethodsScipyMinimize):
            if "groupMethod" in orientationMethodArgs:
                groupMethod = orientationMethodArgs["groupMethod"]        
            else:
                groupMethod = self.groupMethod    
            if "initRotation" in orientationMethodArgs:
                initRotation = orientationMethodArgs["initRotation"]
            else:
                #coarse brute approximation for initialization
                if mode3D:
                    #use random init because brute force is unbearably slow in 3D and performs poorly anayways
                    # initRotation = np.random.rand(3)*[2,1,2]*np.pi
                    initRotation = self.brute3DRotByPathGroups(paths, (10,10,10), groupMethod)
                else:
                    initRotation = self.bruteAoA0ByPathGroups(paths, 100, groupMethod)
            (rotation_est, rotation_cov) = self.numericRot0ByPathGroups(paths, initRotation, groupMethod, themethod)
            if not mode3D:
                rotation_est=rotation_est[0]
        else:
            print("unsupported method")
            return(None)
        
        (d0, tauerr, d) = self.computeAllPaths(paths, rotation=rotation_est)

        return(d0, tauerr, d, rotation_est, rotation_cov)
        
    def getTParamToLoc(self,x0,y0,tauE,AoA0,x,y,dAxes,vAxes):
        dfun={
            ('dTau','dx0') : lambda x0,y0,x,y: x0/3e8/np.sqrt((x0-x[:,None,:])**2+(y0-y[:,None,:])**2),
            ('dTau','dy0') : lambda x0,y0,x,y: y0/3e8/np.sqrt((x0-x[:,None,:])**2+(y0-y[:,None,:])**2),
            ('dTau','dTauE') : lambda x0,y0,x,y: -np.ones_like(x[:,None,:]),
            ('dTau','dAoA0') : lambda x0,y0,x,y: np.zeros_like(x[:,None,:]),
            ('dTau','dx') : lambda x0,y0,x,y: ((x[:,None,:]==x)*x)/3e8*(1/np.sqrt((x0-x)**2+(y0-y)**2)+1/np.sqrt((x)**2+(y)**2)),
            ('dTau','dy') : lambda x0,y0,x,y: ((y[:,None,:]==y)*y)/3e8*(1/np.sqrt((x0-x)**2+(y0-y)**2)+1/np.sqrt((x)**2+(y)**2)),
            
            ('dAoD','dx0') : lambda x0,y0,x,y: np.zeros_like(x[:,None,:]),
            ('dAoD','dy0') : lambda x0,y0,x,y: np.zeros_like(x[:,None,:]),
            ('dAoD','dTauE') : lambda x0,y0,x,y: np.zeros_like(x[:,None,:]),
            ('dAoD','dAoA0') : lambda x0,y0,x,y: np.zeros_like(x[:,None,:]),
            ('dAoD','dx') : lambda x0,y0,x,y: (-y*(y[:,None,:]==y))/((x)**2+(y)**2),
            ('dAoD','dy') : lambda x0,y0,x,y: (x*(x[:,None,:]==x))/((x)**2+(y)**2),
            
            ('dDAoA','dx0') : lambda x0,y0,x,y: (y[:,None,:]-y0)/((x0-x[:,None,:])**2+(y0-y[:,None,:])**2),
            ('dDAoA','dy0') : lambda x0,y0,x,y: (x0-x[:,None,:])/((x0-x[:,None,:])**2+(y0-y[:,None,:])**2),
            ('dDAoA','dTauE') : lambda x0,y0,x,y: np.zeros_like(x[:,None,:]),
            ('dDAoA','dAoA0') : lambda x0,y0,x,y: -np.ones_like(x[:,None,:]),
            ('dDAoA','dx') : lambda x0,y0,x,y: (y0-y[:,None,:])/((x0-x)**2+(y0-y)**2),
            ('dDAoA','dy') : lambda x0,y0,x,y: (x[:,None,:]-x0)/((x0-x)**2+(y0-y)**2)
              }
        T= np.concatenate([np.vstack([dfun[term,var](x0,y0,x,y) for term in dAxes]) for var in vAxes],axis=1)
#        T=np.concatenate(listOfPartials,axis=0)
        return(T)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--load', help='CVS name file lo load channel parameters')
    parser.add_argument('--AoD', help='Array with Angle of Departure values')
    parser.add_argument('--DAoA', help='Array with Angle of Arrival values')
    parser.add_argument('--TDoA', help='Array with NLOS paths delays values')
    
    args = parser.parse_args()
    
    if args.load:
        #Load CSV file
        data = pd.read_csv(args.load, header = 0)
        AoD = data['AoD']
        DAoA = data['DAoA']
        TDoA = data['TDoA']

    else:
        AoD = args.AoD
        DAoA = args.DAoA
        TDoA = args.TDoA

    loc = MultipathLocationEstimator(nPoint = 100, orientationMethod = "lm")
    (AoA0_fsolve, d_0x_fsolve, d_0y_fsolve,_,_,_,_) = loc.computeAllLocationsFromPaths(AoD, DAoA, TDoA)
    print(AoA0_fsolve, d_0x_fsolve, d_0y_fsolve)

