#!/usr/bin/python


import numpy as np
import scipy.optimize as opt
import argparse
import pandas as pd


class MultipathLocationEstimator:
    """Class used to perform the calculation and estimation of the UE (User Equipment) position in 2D. 
    
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


    Methods
    -------
    computeAllPaths(AoD, DAoA, TDoA, AoA0_est)
        Performs the calculation of the possible UE vector positions using the linear method.
     
    gen3PathGroup(Npath)
        Returns the table with the path groups using the 3-path method.
    
    genDrop1Group(Npath)
        Returns the table with the path groups using the drop1 method.
    
    genRandomGroup(Npath, Nlines)
        Returns the table with the path groups using the random method.
    
    feval_wrapper_AllPathsByGroupsFun(x, AoD, DAoA, TDoA, group_method='drop1')
        Defines the minimum mean squared (MSE) distance function to solve all the possible UE vector positions using the 
        linear method by grouping the paths into sets defined by the group_method.
    
    bruteAoA0ForAllPaths(self, AoD, DAoA, TDoA, Npoint, group_method='drop1')
        Performs the estimation of the value of AoA0 using the brute force method.
    
    solveAoA0ForAllPaths(self, AoD, DAoA, TDoA, init_AoA0, group_method='drop1', RootMethod='lm')
        Performs the estimation of the value of AoA0 using the scipy.optimize.root function.
    
    computeAllLocationsFromPaths(self, AoD, DAoA, TDoA, Npoint, hint_AoA0=None, AoA0_method='fsolve', group_method='drop1', RootMethod='lm')
        Performs the estimation of the phi_0 specified by the parameter method, and returns the position of the UE for this angle.
    
    """
   
    def __init__(self, Npoint=100, RootMethod='lm'):
        """Constructs all the attributes for the MultipathLocationEstimator object.


        Parameters
        ----------


        Npoint : int, optional
            Total point divisions in the minimization range of search.


        RootMethod: str, optional
            Type of solver.
            ** lm (Levenberg–Marquardt algorithm): specified for solving non-linear least squares problems.
            ** hybr (Hybrid method): it uses Powell's dog leg method, an iterative optimization algorithm for the solution of 
            non-linear least squares problem.
            
        """


        self.NLinePointsPerIteration = Npoint
        self.RootMethod = RootMethod
        self.c = 3e8
    
    def computeAllPaths(self, AoD, DAoA, TDoA, AoA0_est):
        """Performs the calculation of the possible UE vector positions using the linear method.
        
        The value of the UE position is obtained by using the linear algorithm estimation. For this purpose the 
        algorithm takes the sets of the DAoA, AoD and delays values of the all the NLOS Npaths and obtained all 
        the possibles position vectors (d0x, d0y) of the UE, for the value specified by AoA0_est.
        
        
        ----------------------------------    GENERALIZED LINEAR METHOD    -------------------------------------


        Considering having the DAoA, AoD and delays values for Npaths: {1,2,3, . . . , Npath}, generalized linear method 
        is divided into the following steps:
        
        STEP (1):
            Obtain the unknown values for P, Q and Dl unknowns for the Npaths.
        
        STEP (2): 
            With Npaths, it can be written the following system of linear equations:
            
                                                            A @ x = b      (1)
            
            
                                    | P{1}     Q{1}     -1|              |  Dl[1]  |
                                    | P{2}     Q{1}     -1|              |  Dl[2]  |
                                    | P{3}     Q{1}     -1|   |d0xest|   |  Dl[3]  |
                                    | P{4}     Q{1}     -1| @ |d0yest| = |  Dl[4]  |  (2)
                                    |                     |   |l0err |   |         |
                                    |           ...       |              |   ...   |
                                    |                     |              |         |
                                    | P{Npath} P{Npath} -1|              |Dl[Npath]|




        STEP (3):
            Compute the minimum least-squares solution.
            
            *** The method "numpy.linalg.lstsq" computes the least-squares solution. What means that obtains the vector x that 
            approximately solves the equation (1). 
            In this case, the system is over-determined (the number of linearly independent rows of a can be less than, 
            equal to, or greater than its number of linearly independent columns). So if there are multiple minimizing 
            solutions, the one with the smallest euclidean 2-norm (min|b - A x|) is returned.
        
        ----------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        AoD  : ndarray
            Angles of Departure of the NLOS ray propagation, measured in the BS from the positive x-axis in the 
            non-clockwise.
            
        DAoA  : ndarray
            Angles of Arrival of the NLOS ray propagation, measured in the UE from the positive x-axis in the 
            non-clockwise sense. The value of phi_0 can modify the orientation of the x-axis.
            
        TDoA : ndarray
            Delays introduced by the NLOS ray propagations.
            
        AoA0_est: ndarray
            Offset angle of the UE orientation in the non-clockwise sense.


        Returns
        -------
        d0xest : ndarray
            x-coordinate of the UE estimated position.
            
        d0yest : ndarray
            y-coordinate of the UE estimated position.
            
        tauEest : ndarray
            initial delay difference between the LOS path and the NLOS path propagation.
            
        vyest : ndarray
           x-coordinate of the scatter estimated position.
            
        vxest : ndarray
           y-coordinate of the scatter estimated position.
          
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
        tauEest = (l0est - l0err)/self.c
        
        vyest = np.where(tgD!=0, (-tgA*d0xest + d0yest)/(-tgA/tgD + 1), 0)
        vxest = np.where(tgD!=0, vyest/tgD, d0xest - d0yest/tgA)
            
        return(d0xest, d0yest, tauEest, vxest, vyest)
    
    def gen3PathGroup(self, Npath):
        """Returns the table with the path groups using the 3-path method.
        
        This function divides the set {1, 2, 3, . . ., Npath}, where Npath is the total number of NLOS paths into several 
        groups of paths G1(1, 2, 3), G2(2, 3, 4) , G3(3, 4, 5), . . ., Gm(Npath-2, Npath-1, Npath). Each group is defined as 
        the result of combination of 3 paths in a total of (Npath-2) groups.
          
        E.g.:
        The total number of paths, Npath = 10, so groups includes paths as:
        
                                                G1 = {1,2,3}       G5 = {5,6,7}
                                                G2 = {2,3,4}       G6 = {6,7,8}
                                                G3 = {3,4,5}       G7 = {7,8,9}
                                                G4 = {4,5,6}       G8 = {8,9,10}
        
        For this examples, the table is generated as:
        
                                 |True  True  True False False False False False False False|
                                 |False True  True  True False False False False False False|
                                 |False False True  True  True False False False False False|
                                 |False False False True  True  True False False False False|
                                 |False False False False True  True  True False False False|
                                 |False False False False False True  True  True False False|
                                 |False False False False False False True  True  True False|
                                 |False False False False False False False True  True  True|
        
        ---------------------------------------------------------------------------------------------------------


        Parameters
        ----------
        Npath  : int
            Total number of NLOS paths.


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
        """Returns the table with the path groups using the drop1 method.
        
        This function divides the set {1, 2, 3, . . ., Npath}, where Npath is the total number of NLOS paths into several 
        groups of paths of paths G1, G2 , G3, . . ., Gm. Each group is defined as the group of all paths except de m-th one.
        So the Gm group will inclue: Gm = {1, . . . ,m-1 , m+1, . . ., Npath}
          
        E.g.:
        The total number of paths, Npath = 10, so groups includes paths as:
        
                                    G1 = {2,3,4,5,6,7,8,9,10}       G6 = {1,2,3,4,5,7,8,9,10}
                                    G2 = {1,3,4,5,6,7,8,9,10}       G7 = {1,2,3,4,5,6,8,9,10}
                                    G3 = {1,2,4,5,6,7,8,9,10}       G8 = {1,2,3,4,5,6,7,9,10}
                                    G4 = {1,2,3,5,6,7,8,9,10}       G9 = {1,2,3,4,5,6,7,8,10}
                                    G5 = {1,2,3,4,6,7,8,9,10}       G10 = {1,2,3,4,5,6,7,8,9}
        
        For this example, the table is generated as:
        
                                 |False  True  True  True  True  True  True  True  True  True|
                                 |True  False  True  True  True  True  True  True  True  True|
                                 |True  True  False  True  True  True  True  True  True  True|
                                 |True  True  True  False  True  True  True  True  True  True|
                                 |True  True  True  True  False  True  True  True  True  True|
                                 |True  True  True  True  True  False  True  True  True  True|
                                 |True  True  True  True  True  True  False  True  True  True|
                                 |True  True  True  True  True  True  True  False  True  True|
                                 |True  True  True  True  True  True  True  True  False  True|
                                 |True  True  True  True  True  True  True  True  True  False|
                              
        ---------------------------------------------------------------------------------------------------------
                         
        Parameters
        ----------
        Npath  : int
            Total number of NLOS paths.


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
        """Returns the table with the path groups using the random method.
        
        This function divides the set {1, 2, 3, .Npath}, where Npath is the total number of NLOS paths into several 
        groups of paths G1(1, 2, 7), G2(2, 3, 4) , G3(3, 4, 5), . . ., Gm(Npath-2, Npath-1, Npath). Each group is defined as 
        the result of a random choice picking Nmembers out of Npath with uniform distribution.
          
        E.g.:
        The total number of paths, Npath = 10, Ngroups=6, Nmembers=4 so groups look like:
        
                                                G1 = {3,4,6,8}       G4 = {3,5,6,7}
                                                G2 = {1,3,6,8}       G5 = {3,4,5,8}
                                                G3 = {1,4,5,10}      G6 = {1,2,4,6}
        
        For this example, the table is generated as:
        
                                  |False False  True  True False  True False False  True False|
                                  | True False  True False False  True False False  True False|
                                  | True False False  True  True False False False False  True|
                                  |False False  True False  True  True  True False False False|
                                  |False False  True  True  True False False  True False False|
                                  | True  True False  True False  True False False False False|
        
        ---------------------------------------------------------------------------------------------------------


        Parameters
        ----------
        Npath  : int
            Total number of NLOS paths.


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


    def feval_wrapper_AllPathsByGroupsFun(self, x, AoD, DAoA, TDoA, group_method='drop1'):
        """Determines the minimum mean squared (MSE) distance function of solving all the possible UE vector positions using the 
        linear method by grouping the paths into sets defined by the group_method.
        
        The group_method method calls:
        - '3path' : 
            gen3PathGroup() to create the path groups.

        - 'drop1' : 
            genDrop1Group() to create the path groups.

        - 'random' : 
            genRandomGroup() to create the path groups.
            
        ---------------------------------------------------------------------------------------------------------
   
        Parameters
        ----------
        x: ndarray
            Function unknown (AoA0).
            
        AoD  : ndarray
            Angles of Departure of the NLOS ray propagation, measured in the BS from the positive x-axis in the 
            non-clockwise.
            
        DAoA  : ndarray
            Angles of Arrival of the NLOS ray propagation, measured in the UE from the positive x-axis in the 
            non-clockwise sense. The value of phi_0 can modify the orientation of the x-axis.
            
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
            (d0xall[gr], d0yall[gr], tauEall[gr],_,_) = self.computeAllPaths(AoD[table_group[gr, :]], DAoA[table_group[gr, :]], TDoA[table_group[gr, :]], x)
        return(np.sum(np.abs(d0xall-np.mean(d0xall,d0xall.ndim-1,keepdims=True))**2+np.abs(d0yall-np.mean(d0yall,d0xall.ndim-1,keepdims=True))**2+np.abs(self.c*tauEall-np.mean(self.c*tauEall,d0xall.ndim-1,keepdims=True))**2,d0xall.ndim-1))

    def bruteAoA0ForAllPaths(self, AoD, DAoA, TDoA, Npoint, group_method='drop1'):
        """Performs the estimation of the value of AoA0 using the brute force method.
        
        The value of the estimated offset angle of the UE orientation is obtained by using the bisection algorithm.
        For this purpose the method divides the range of the possible values of phi_0, among 0 and 2pi into Npoints 
        and minimize the error. The method reduces recurrently the range till minimize the error in phi_0 estimation.
        
        ----------------------------------------   BRUTE FORCE SEARCH    ----------------------------------------
        
        The brute force solution generates thousands os points in the interval (0, 2pi) and picks the one with the
        lowest minimun square error (MSE).
        
        ---------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        AoD  : ndarray
            Angles of Departure of the NLOS ray propagation, measured in the BS from the positive x-axis in the 
            non-clockwise.
            
        DAoA  : ndarray
            Angles of Arrival of the NLOS ray propagation, measured in the UE from the positive x-axis in the 
            non-clockwise sense. The value of phi_0 can modify the orientation of the x-axis.
            
        TDoA : ndarray
            Delays introduced by the NLOS ray propagations.
            
        Npoint : int, optional
            Total point divisions in the minimization range of search.
            ** The range of search is [0-2pi]
            
        group_method : ndarray, optional
            Path grouping strategy.
            *** Options: 'drop1', '3-path', 'random'
            *** Default value is 'drop1'.

        Returns
        -------
        AoA0_est: ndarray
            Offset angle estimated of the UE orientation.

        """
        
        if not Npoint:
            Npoint = self.NLinePointsPerIteration
            
        philow = 0
        phihigh = 2*np.pi
        interval = np.linspace(philow, phihigh, Npoint).reshape(-1,1)
        dist = np.zeros(Npoint)

        for npoint in range(Npoint):
            dist[npoint] = self.feval_wrapper_AllPathsByGroupsFun(interval[npoint], AoD, DAoA, TDoA, group_method)
        
        distint = np.argmin(dist)

        return(interval[distint])
    
    def solveAoA0ForAllPaths(self, AoD, DAoA, TDoA, init_AoA0, group_method='drop1', RootMethod='lm'):
        """Performs the estimation of the value of AoA0 using the scipy.optimize.root function.
        
        The value of the estimated offset angle of the UE orientation is obtained by finding the zeros of the 
        minimum mean square error (MMSE) equation defined by feval_wrapper_AllPathsByGroups. For this purpose, it is used
        the method root() to find the solutions of this function.


        ---------------------------------    MULTIDIMENSIONAL ROOT METHOD    ------------------------------------
        
        In this case, it is used the root method of scipy.optimize.root() to solve a non-linear equation with parameters.
        
        TO USE THE METHOD:
        sol = scipy.optimize.root(fun, x0, args=(), method='lm').
        
        Parameters
        ----------
        fun : callable
            The vector function to find a root of.
        
        x0 : ndarray
            Initial guess.
    
        args : tuple, optional
            Extra arguments passed to the objective function and its Jacobian. These are other variables 
            that we aren't finding the roots for. The parameters have to appear in the same order as the function.
            
        method : str
            Type of solver.
            *** lm (Levenberg–Marquardt algorithm): especified for solving non-linear least squares problem.
        
        
        ---------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        AoD  : ndarray
            Angles of Departure of the NLOS ray propagation, measured in the BS from the positive x-axis in the 
            non-clockwise.
            
        DAoA  : ndarray
            Angles of Arrival of the NLOS ray propagation, measured in the UE from the positive x-axis in the 
            non-clockwise sense. The value of phi_0 can modify the orientation of the x-axis.
            
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
            non-clockwise.
            
        DAoA  : ndarray
            Angles of Arrival of the NLOS ray propagation, measured in the UE from the positive x-axis in the 
            non-clockwise sense. The value of phi_0 can modify the orientation of the x-axis.
            
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
        
        (d0x, d0y, tauerr, vx, vy) = self.computeAllPaths(AoD, DAoA, TDoA, AoA0_est)

        return(AoA0_est, d0x, d0y, tauerr, vx, vy, cov_AoA0)
        
    def getTParamToLoc(self,x0,y0,tauE,AoA0,x,y,dAxes,vAxes):
        dfun={
            ('dTau','dx0') : lambda x0,y0,x,y: x0/3e-1/np.sqrt((x0-x[:,None,:])**2+(y0-y[:,None,:])**2),
            ('dTau','dy0') : lambda x0,y0,x,y: y0/3e-1/np.sqrt((x0-x[:,None,:])**2+(y0-y[:,None,:])**2),
            ('dTau','dTauE') : lambda x0,y0,x,y: -np.ones_like(x[:,None,:]),
            ('dTau','dAoA0') : lambda x0,y0,x,y: np.zeros_like(x[:,None,:]),
            ('dTau','dx') : lambda x0,y0,x,y: ((x[:,None,:]==x)*x)/3e-1*(1/np.sqrt((x0-x)**2+(y0-y)**2)+1/np.sqrt((x)**2+(y)**2)),
            ('dTau','dy') : lambda x0,y0,x,y: ((y[:,None,:]==y)*y)/3e-1*(1/np.sqrt((x0-x)**2+(y0-y)**2)+1/np.sqrt((x)**2+(y)**2)),
            
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

