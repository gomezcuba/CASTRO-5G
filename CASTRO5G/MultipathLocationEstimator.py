#!/usr/bin/python

import numpy as np
import scipy.optimize as opt
import argparse
import pandas as pd

class MultipathLocationEstimator:
    """Class used to calculate 5G UE (User Equipment) location in 2D and 3D. 
    
    The main goal of the MultipathLocationEstimator class is try to recover the UE position trigonometrically, defined as 
    (d0x, d0y), estimating the value of user offset orientation (phi_0) from the knowledge of the set of the Angles of Departure 
    (AoD), Angles of Arrival (DAoA) and delays introduced by the multipath channels. 
    This class includes several algorithms to obtain the value of phi_0, from a brute force searching method to a 
    non-linear system equation solver. Also, different methods have been included to find the UE position dealing with clock
    and orientation error.
    
    ...

    Attributes
    ---------
    Npoint : int, optional
        Total point divisions in the minimization range of search.
        
    RootMethod: str, optional
        Type of solver.
        ** lm (Levenberg–Marquardt algorithm): specified for solving non-linear least squares problem.
        ** hybr (Hybrid method): it uses Powell's dog leg method, an iterative optimization algorithm for the solution of 
        non-linear least squares problem.
    
    """
   
    def __init__(self, Npoint=100, RootMethod='lm'):
        """ MultipathLocationEstimator constructor.
        
        Parameters
        ----------

        Npoint : int, optional
            Total points in the search range for brute-force orientation finding.

        RootMethod: str, optional
            Type of solver algorithm for scipy.optimize.root() LS orientation finding.
            ** lm : Levenberg–Marquardt algorithm for solving non-linear least squares problems.
            ** hybr : Hybrid method using Powell's dog leg method, an iterative optimization algorithm for solving non-linear least squares problem.
            
        """
        self.NLinePointsPerIteration = Npoint
        self.RootMethod = RootMethod
        self.c = 3e8
    
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
    def uVectorT(self,A,Z=None):
        """Computes unitary directional ROW transposed vectors. 
        
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
            return( np.column_stack([np.cos(A),np.sin(A)]) )
        else:
            return( np.column_stack([np.cos(A)*np.sin(Z),np.sin(A)*np.sin(Z),np.cos(Z)]) )
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
            Zenith, "downtilt" pitch of rotation measured downwards from the Z axis, around the "facing-to-A" horizontal axis. If None, a 2D matrix is returned
        S  : float, optional, default = None
            "Slant" roll of rotation measured form up from the Y axis, around the "facing-to-AZ" axis). If None, a 3D matrix rotated in A,Z is returned

        Returns
        -------
        R : 3x3
            3 Axis Rotation matrix with azimuth A, zenith Z and slant S
        """
        if Z is None:
            return( self.rMatrix2D(A) )
        elif S is None:
            return( self.rMatrix3D(A,2)@self.rMatrix3D(Z,1) )
        else:
            return( self.rMatrix3D(A,2)@self.rMatrix3D(Z,1)@self.rMatrix3D(S,0) )
    def computeAllPaths(self, AoD, DAoA, TDoA, ZoD=None, DZoA=None, rotation=None):       
        Npath=len(AoD)    
        #TODO choose a library to provide uVector globally to the project
        DoD = self.uVectorT(AoD,ZoD)
        if rotation is None:
            DoA = self.uVectorT(DAoA,DZoA)
        else:
            if (ZoD is None) and (DZoA is None):
                AoA0_est=rotation
                Rm=self.rMatrix(AoA0_est)
            else:
                AoA0_est,ZoA0_est,SoA0_est=rotation
                Rm=self.rMatrix(AoA0_est,ZoA0_est,SoA0_est)
            DoA = self.uVectorT(DAoA,DZoA)@Rm.T
            
        C12= np.sum(-DoD*DoA,axis=1,keepdims=True)
        M=np.column_stack([(DoD-DoA)/(1+C12),-np.ones((Npath,1))])
        result_est=np.linalg.lstsq(M, TDoA*self.c, rcond=None)[0]
        d0_est = result_est[0:-1]
        l0err = result_est[-1]
        l0est = np.linalg.norm(d0_est)
        ToA0_est = (l0est - l0err)/self.c
        Vi=(DoD+C12*DoA)/(1-C12**2)
        liD_est=Vi@d0_est
        d_est=liD_est[:,None]*DoD
        return(d0_est,ToA0_est,d_est)
    
    def computeAllPathsV1wrap(self, AoD, DAoA, TDoA, rotation=None):
        """Computes computeAllPathsV1 with the input-output convention of the new version
        
        ----------------------------------------------------------------------------------------------------------
        """
        AoA0_est = 0 if rotation is None else rotation
        d0xest, d0yest, ToA0_est, vxest, vyest = self.computeAllPathsV1(AoD, DAoA, TDoA, AoA0_est)
        d0_est = np.array([d0xest,d0yest])
        d_est = np.vstack([vxest,vyest]).T
        return(d0_est, ToA0_est, d_est)
        
    def gen3PathGroup(self, Npath):
        """Returns a Npath-2 x Npath boolean table representing paths belonging to 3-path groups.
        
        Divides the set {1, 2, 3, . . ., Npath} into Npath-2 groups of paths G1={1, 2, 3},
        G2={2, 3, 4} , G3{3, 4, 5}, . . ., Gm={Npath-2, Npath-1, Npath}. 
        
        E.g.:
        Npath = 5, the groups includes paths as:        
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
        table_group = np.empty((Npath-2, Npath), dtype=bool)

        for gr in range(Npath-2):
            path_indices = [gr, gr+1, gr+2]
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


    def feval_wrapper_AllPathsByGroupsFun(self, x, AoD, DAoA, TDoA, ZoD=None, DZoA=None, group_method='drop1'):
        """Evaluates the minimum mean squared (MSE) distance among all linear location solutions
        obtained by multiple groups of paths, as a function of the AoA0 of the receiver.
            
        ---------------------------------------------------------------------------------------------------------
   
        Parameters
        ----------
        x: ndarray
            Function unknown (AoA0).
            
        AoD  : ndarray
            Angles of Departure of the NLOS ray propagation, measured in the BS from the positive x-axis in the 
            counter-clockwise.
            
        DAoA  : ndarray
            Angles of Arrival of the NLOS ray propagation, measured in the UE from the positive x-axis in the 
            counter-clockwise sense. The value of phi_0 can modify the orientation of the x-axis.
            
        TDoA : ndarray
            Delays introduced by the NLOS ray propagations.
            
        group_method : ndarray, optional
            Path grouping strategy.
            *** Options: 'drop1', '3-path', 'random'
            *** Default value is 'drop1'.
            
        Returns
        -------
        Returns the mathematical polinomial function based in the minimum mean square error (MMSE) equation.

        """
        Npath = AoD.size
        if group_method == '3path':
            table_group = self.gen3PathGroup(Npath)            
        elif group_method == 'drop1':
            table_group = self.genDrop1Group(Npath)            
        elif group_method == 'random':
            Ngroups = Npath
            Nmembers = Npath//2
            table_group = self.genRandomGroup(Npath, Ngroups, Nmembers)
        else:
            table_group = group_method

        Ngroup = table_group.shape[0]
        d0xall = np.zeros(Ngroup)
        d0yall = np.zeros(Ngroup)
        tauEall = np.zeros(Ngroup)
        for gr in range(Ngroup):
            (d0xall[gr], d0yall[gr], tauEall[gr],_,_) = self.computeAllPathsV1(AoD[table_group[gr, :]], DAoA[table_group[gr, :]], TDoA[table_group[gr, :]], x)
        return(np.sum(np.abs(d0xall-np.mean(d0xall,d0xall.ndim-1,keepdims=True))**2+np.abs(d0yall-np.mean(d0yall,d0xall.ndim-1,keepdims=True))**2+np.abs(self.c*tauEall-np.mean(self.c*tauEall,d0xall.ndim-1,keepdims=True))**2,d0xall.ndim-1))
        # Ndim = 2 if (ZoD is None) or (DZoA is None) else 3
        # d0_all = np.zeros((Ngroup,Ndim))
        # tauEall = np.zeros(Ngroup)
        # for gr in range(Ngroup):
        #     (d0_all[gr,:], tauEall[gr],_) = self.computeAllPathsV1wrap(AoD[table_group[gr, :]], DAoA[table_group[gr, :]], TDoA[table_group[gr, :]], rotation=x)
        # v_all=np.concatenate([d0_all,self.c*tauEall[:,None]],axis=1)
        # v_all-=np.mean(v_all,axis=0)
        # return( np.sum( np.abs(v_all)**2 ) )

    def bruteAoA0ForAllPaths(self, AoD, DAoA, TDoA, Npoint=None, group_method='drop1'):
        """Estimates the value of the receiver Azimuth AoA0 by brute force by minimizing the
        mean squared distance between location solutions of multiple groups. Finds the best
        of Npoints between 0 and 2π
        
        ---------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        AoD  : ndarray
            Angles of Departure of the NLOS ray propagation, measured in the BS from the positive x-axis in the 
            counter-clockwise.
            
        DAoA  : ndarray
            Angles of Arrival of the NLOS ray propagation, measured in the UE from the positive x-axis in the 
            counter-clockwise sense. The value of phi_0 can modify the orientation of the x-axis.
            
        TDoA : ndarray
            Delays introduced by the NLOS ray propagations.
            
        Npoint : int, optional
            Total point divisions in the minimization range of search.
            *** The range of search is [0-2pi]
            *** Default value is 100
            
        group_method : ndarray, optional
            Path grouping strategy.
            *** Options: 'drop1', '3-path', 'random'
            *** Default value is 'drop1'.

        Returns
        -------
        AoA0_est: ndarray
            Receiver UE Azimuth orientation angle

        """        
        if Npoint is None:
            Npoint = self.NLinePointsPerIteration            
        interval = np.linspace(0,  2*np.pi, Npoint)
        dist = np.zeros(Npoint)
        for npoint in range(Npoint):
            dist[npoint] = self.feval_wrapper_AllPathsByGroupsFun(interval[npoint], AoD, DAoA, TDoA, group_method)        
        distind = np.argmin(dist)

        return(interval[distind])
    
    def solveAoA0ForAllPaths(self, AoD, DAoA, TDoA, init_AoA0, group_method='drop1', RootMethod='lm'):
        """Performs the estimation of the value of AoA0 using the scipy.optimize.root function.
        
        The value of the estimated offset angle of the UE orientation is obtained by finding the zeros of the 
        minimum mean square error (MMSE) equation defined by feval_wrapper_AllPathsByGroups. For this purpose, it is used
        the method root() to find the solutions of this function.

        In this case, it is used the root method of scipy.optimize.root() to solve a non-linear equation with parameters.
        ---------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        AoD  : ndarray
            Angles of Departure of the NLOS ray propagation, measured in the BS from the positive x-axis in the 
            counter-clockwise.
            
        DAoA  : ndarray
            Angles of Arrival of the NLOS ray propagation, measured in the UE from the positive x-axis in the 
            counter-clockwise sense. The value of phi_0 can modify the orientation of the x-axis.
            
        TDoA : ndarray
            Delays introduced by the NLOS ray propagations.
        
        init_AoA0 : ndarray
            Hint or guess about the value of AoA0.
            
        group_method : ndarray, optional
            Path grouping strategy.
            *** Options: 'drop1', '3-path', 'random'
            *** Default value is 'drop1'.
        
        RootMethod: str, optional
            Type of solver.
            ** lm (Levenberg–Marquardt algorithm): especified for solving non-linear least squares problem.
            ** hybr (Hybrid method): it uses Powell's dog leg method, an iterative optimization algorithm for the solution of 
            non-linear least squares problem.

        Returns
        -------
        AoA0_est: ndarray
            Offset angle estimated of the UE orientation.
                  
        """
       
        res = opt.root(self.feval_wrapper_AllPathsByGroupsFun, x0=init_AoA0, args=(AoD, DAoA, TDoA, group_method), method=self.RootMethod)
        if not res.success:
        #print("Attempting to correct initialization problem")
            niter = 0 
            while (not res.success) and (niter<1000):
                res = opt.root(self.feval_wrapper_AllPathsByGroupsFun, x0=2*np.pi*np.random.rand(1), args=(AoD, DAoA, TDoA, group_method), method=self.RootMethod)
                niter += 1
        #print("Final Niter %d"%niter)
        if res.success:
            return (res.x,res.cov_x)
        else:
            print("ERROR: AoA0 root not found")
            return (np.array(0.0),np.inf)

    def computeAllLocationsFromPaths(self, AoD, DAoA, TDoA, Npoint=100, hint_AoA0=None, AoA0_method='fsolve', group_method='drop1', RootMethod='lm'):
        """Performs the estimation of the phi_0 especified by the parameter method, and returns the position 
        of the UE for this angle.

        The parameter method calls:
        - 'fsolve' : 
            calls solveAoA0ForAllPaths() to estimate AoA0.

        - 'brute' : 
            calls bisectAoA0ForAllPaths() to estimate AoA0.
            
        ---------------------------------------------------------------------------------------------------------

        Parameters
         ----------
        AoD  : ndarray
            Angles of Departure of the NLOS ray propagation, measured in the BS from the positive x-axis in the 
            counter-clockwise.
            
        DAoA  : ndarray
            Angles of Arrival of the NLOS ray propagation, measured in the UE from the positive x-axis in the 
            counter-clockwise sense. The value of phi_0 can modify the orientation of the x-axis.
            
        TDoA : ndarray
            Delays introduced by the NLOS ray propagations.
            
        Npoint : int, optional
            Total point divisions in the minimization range of search.
            ** The range of search is [0-2pi]
            
        hint_AoA0 : ndarray, optional
            Hint or guess about the value of AoA0. 
    
        AoA0_method: str, optional
            Method used to performs the value estimation of AoA0.
            *** Options: 'fsolve', 'bisec', 'fsolve_linear'
            *** Default value is 'fsolve'.
        
        group_method : ndarray, optional
            Path grouping strategy.
            *** Options: 'drop1', '3-path', 'random'
            *** Default value is 'drop1'.
            
        Returns
        -------
        AoA0_est: ndarray
            Offset angle estimated of the UE orientation.
            
        d0x : ndarray
            x-coordinate of the possible position of the UE.
            
        d0y : ndarray
            y-coordinate of the possible position of the UE.
            
        vy : ndarray
            y-coordinate of the reflector position in the NLOS path.
            
        vx : ndarray
            x-coordinate of the reflector position in the NLOS path.
            
        cov_AoA0: ndarray
            The inverse of the Hessian matrix of AoA0.
                        
        """

        if AoA0_method == 'fsolve':
            
            if (hint_AoA0 == None):
                #coarse linear approximation for initialization
                init_AoA0 = self.bruteAoA0ForAllPaths(AoD, DAoA, TDoA, Npoint, group_method)
                (AoA0_est,cov_AoA0) = self.solveAoA0ForAllPaths(AoD, DAoA, TDoA, init_AoA0, group_method, RootMethod)

            else:
                init_AoA0 = hint_AoA0
                (AoA0_est, cov_AoA0) = self.solveAoA0ForAllPaths(AoD, DAoA, TDoA, init_AoA0, group_method, RootMethod)
            cov_AoA0=None
        elif AoA0_method == 'brute':
            AoA0_est = self.bruteAoA0ForAllPaths(AoD, DAoA, TDoA, Npoint, group_method)
            cov_AoA0 = np.pi/self.NLinePointsPerIteration
        else:
            print("unsupported method")
            return(None)
        
        (d0, tauerr, d) = self.computeAllPathsV1wrap(AoD, DAoA, TDoA, AoA0_est)

        return(AoA0_est, d0[0],  d0[1], tauerr, d[0], d[1], cov_AoA0)
        
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

    loc = MultipathLocationEstimator(Npoint = 100, RootMethod = "lm")
    (AoA0_fsolve, d_0x_fsolve, d_0y_fsolve,_,_,_,_) = loc.computeAllLocationsFromPaths(AoD, DAoA, TDoA)
    print(AoA0_fsolve, d_0x_fsolve, d_0y_fsolve)

