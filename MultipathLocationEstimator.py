#!/usr/bin/python

import numpy as np
import scipy.optimize as opt
import argparse
import pandas as pd

class MultipathLocationEstimator:
    """Class used to perform the calculation and estimation of the UE (User Equipment) position in 2D. 
    
    The main goal of the MultipathLocationEstimator class is try to recover the UE position trigonometrically, defined as 
    (d0x, d0y), estimating the value of user offset orientation (phi_0) from the knowledge of the set of the Angles Of Departure 
    (AoD), Angles Of Arrival (AoA) and delays introduced by the multipath channels. 
    This class includes several algorithms to obtain the value of phi_0, from a brute force searching method to a 
    non-linear system ecuation solver. Also differents methods have been included to find the UE position dealing with clock
    and orientation error.
    
    ...
    

    Attributes
    ---------
    Npoint : int, optional
        Total point divisions in the minimization range of search.
        
    RootMethod: str, optional
        Type of solver.
        ** lm (Levenberg–Marquardt algorithm): especified for solving non-linear least squares problems.
        ** hybr (Hybrid method): it uses Powell's dog leg method, an iterative optimisation algorithm for the solution of 
        non-linear least squares problems.

    Methods
    -------
    computeAllPaths(AoD, AoA, dels, phi0_est)
        Performs the calculation of the posible UE vector positions using the linear method.
     
    gen3PathGroup(Npath)
        Returns the table with the path groups using the 3-path method.
    
    genDrop1Group(Npath)
        Returns the table with the path groups using the drop1 method.
    
    genRandomGroup(Npath, Nlines)
        Returns the table with the path groups using the random method.
    
    feval_wrapper_AllPathsByGroupsFun(x, AoD, AoA, dels, group_method='drop1')
        Defines the minimum mean squared (MSE) distance function to solve all the posible UE vector positions using the 
        linear method by grouping the paths into sets defined by the group_method.
    
    brutePhi0ForAllPaths(self, AoD, AoA, dels, Npoint, group_method='drop1')
        Performs the estimation of the value of phi0 using the brute force method.
    
    solvePhi0ForAllPaths(self, AoD, AoA, dels, init_phi0, group_method='drop1', RootMethod='lm')
        Performs the estimation of the value of phi0 using the scipy.optimize.root function.
    
    computeAllLocationsFromPaths(self, AoD, AoA, dels, Npoint, hint_phi0=None, phi0_method='fsolve', group_method='drop1', RootMethod='lm')
        Performs the estimation of the phi_0 especified by the parameter method, and returns the position 
        of the UE for this angle.
    
    """
   
    def __init__(self, Npoint=1000, RootMethod='lm'):
        """Constructs all the attributes for the MultipathLocationEstimator object.

        Parameters
        ----------

        Npoint : int, optional
            Total point divisions in the minimization range of search.

        RootMethod: str, optional
            Type of solver.
            ** lm (Levenberg–Marquardt algorithm): especified for solving non-linear least squares problems.
            ** hybr (Hybrid method): it uses Powell's dog leg method, an iterative optimisation algorithm for the solution of 
            non-linear least squares problems.
            
        """

        self.NLinePointsPerIteration = Npoint
        self.RootMethod = RootMethod
        self.c = 3e8
    
    def computeAllPaths(self, AoD, AoA, dels, phi0_est):
        """Performs the calculation of the posible UE vector positions using the linear method.
        
        The value of the UE position is obtained by using the linear algorithm estimation. For this purpose the 
        algorithm takes the sets of the AoA, AoD and delays values of the all the NLOS Npaths and obtained all 
        the posibles position vectors (d0x, d0y) of the UE, for the value especified by phi0_est.
        
        
        ----------------------------------    GENERALIZED LINEAR METHOD    -------------------------------------

        Considering having the AoA, AoD and delays values for Npaths: {1,2,3, . . . , Npath}, generalized linear method 
        is divided into the following steps:
        
        STEP (1):
            Obtain the unknown values for P, Q and Dl unknowns for the Npaths.
        
        STEP (2): 
            With Npaths, it can be wrotten the following system of linear equations:
            
                                                            A @ x = b      (1)
            
            
                                    | P{1}     Q{1}     -1|             |  Dl[1]  |
                                    | P{2}     Q{1}     -1|             |  Dl[2]  |
                                    | P{3}     Q{1}     -1|   |d0xest|   |  Dl[3]  |
                                    | P{4}     Q{1}     -1| @ |d0yest| = |  Dl[4]  |  (2)
                                    |                     |   |l0err|   |         |
                                    |           ...       |             |   ...   |
                                    |                     |             |         |
                                    | P{Npath} P{Npath} -1|             |Dl[Npath]|


        STEP (3):
            Compute the minimun least-squares solution.
            
            *** The method "numpy.linalg.lstsq" computes the least-squares solution. What means that obtains the vector x that 
            approximately solves the equation (1). 
            In this case, the system is over-determined (the number of linearly independent rows of a can be less than, 
            equal to, or greater than its number of linearly independent columns). So if there are multiple minimizing 
            solutions, the one with the smallest euclidean 2-norm (min|b - A x|) is returned.
        
        ----------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        AoD  : ndarray
            Angles of departure of the NLOS ray propagation, mesured in the BS from de positive x-axis in the 
            non-colckwise.
            
        AoA  : ndarray
            Angles of arrival of the NLOS ray propagation, mesured in the UE from de positive x-axis in the 
            non-colckwise sense. The value of phi_0 can modify the orientation of the x-axis.
            
        dels : ndarray
            Delays introduced by the NLOS ray propagations.
            
        phi0_est: ndarray
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
        tgA = np.tan(AoA + phi0_est)
        sinD = np.sin(AoD)
        sinA = np.sin(AoA + phi0_est)
        cosD = np.cos(AoD)
        cosA = np.cos(AoA + phi0_est)
        
        P = (sinD + sinA)/(sinD*cosA + cosD*sinA)
        P_mask = np.invert((np.isinf(P) + np.isnan(P)), dtype=bool)
        P = P*P_mask
        P[np.isnan(P)] = 1

        Q = (cosA - cosD)/(sinD*cosA + cosD*sinA)
        Q_mask = np.invert((np.isinf(Q) + np.isnan(Q)), dtype=bool)
        Q = Q*Q_mask
        Q[np.isnan(Q)] = 1

        Dl = dels*self.c
        
        result = np.linalg.lstsq(np.column_stack([P, Q, -np.ones_like(P)]), Dl, rcond=None)
        (d0xest, d0yest, l0err) = result[0]
        
        l0est = np.sqrt(d0xest**2 + d0yest**2)
        tauEest = (l0est - l0err)/self.c
        
        vyest = np.where(tgD!=0, (tgA*d0xest + d0yest)/(tgA/tgD + 1), 0)
        vxest = np.where(tgD!=0, vyest/tgD, d0xest + d0yest/tgA)
            
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
        
        For this examples, the table is generated as:
        
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
            Boolean array that contains all the 3-path groups.
           
        """
        table_group = np.empty((Npath, Npath), dtype=bool)

        for gr in range(Npath):
            table_group[gr,:] = np.isin(np.arange(Npath), [gr], invert=True)
        
        return table_group

    def genRandomGroup(self, Npath, Nlines):
        """Returns the table with the path groups using the random method.
        
        This function divides the set {1, 2, 3, . . ., Npath}, where Npath is the total number of NLOS paths into several 
        groups of paths G1, G2 , G3, . . ., Gm. So the table will include Nlines groups with random Npath values.
          
        E.g.:
        With Npath = 10 and Nlines = 5, groups could be generated as:
        
                                                    G1 = {2,6,7}       
                                                    G2 = {1,3,8,9}       
                                                    G3 = {1,2,5,9,10}       
                                                    G4 = {1,3,5,8,9}
                                                    G5 = {1,3,4}
        
        For this examples, the table is generated as:
        
                                |False True False False False True True False False False]
                                |True False True False False False False  True True False]
                                |True   True False False True False False False True True]
                                |True False   True False True False False True True False]
                                |True False True True False False False False False False]
        
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
        path_indices = np.random.choice(Npath,(Nlines, Npath//2))
        table_group = np.empty((Nlines, Npath), dtype=bool)
        
        for gr in range(Nlines):
            table_group[gr, :] = np.isin(np.arange(Npath), path_indices[gr])
        
        return table_group

    def feval_wrapper_AllPathsByGroupsFun(self, x, AoD, AoA, dels, group_method='drop1'):
        """Defines the minimum mean squared (MSE) distance function to solve all the posible UE vector positions using the 
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
            Function unknown (phi0).
            
        AoD  : ndarray
            Angles of departure of the NLOS ray propagation, mesured in the BS from de positive x-axis in the 
            non-colckwise.
            
        AoA  : ndarray
            Angles of arrival of the NLOS ray propagation, mesured in the UE from de positive x-axis in the 
            non-colckwise sense. The value of phi_0 can modify the orientation of the x-axis.
            
        dels : ndarray
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
            Nlines = 5
            table_group = self.genRandomGroup(Npath, Nlines)

        else:
            table_group = group_method

        Ngroup = table_group.shape[0]
        d0xall = np.zeros(Ngroup)
        d0yall = np.zeros(Ngroup)
        tauEall = np.zeros(Ngroup)

        for gr in range(Ngroup):
            (d0xall[gr], d0yall[gr], tauEall[gr],_,_) = self.computeAllPaths(AoD[table_group[gr, :]], AoA[table_group[gr, :]], dels[table_group[gr, :]], x)
        return(np.sum(np.abs(d0xall-np.mean(d0xall,d0xall.ndim-1,keepdims=True))**2+np.abs(d0yall-np.mean(d0yall,d0xall.ndim-1,keepdims=True))**2+np.abs(self.c*tauEall-np.mean(self.c*tauEall,d0xall.ndim-1,keepdims=True))**2,d0xall.ndim-1))

    def brutePhi0ForAllPaths(self, AoD, AoA, dels, Npoint, group_method='drop1'):
        """Performs the estimation of the value of phi0 using the brute force method.
        
        The value of the estimated offset angle of the UE orientation is obtained by using the bisection algorithm.
        For this purpose the method divides the range of the posible values of phi_0, among 0 and 2pi into Npoints 
        and minimize the error. The method reduces recurrently the range till minimize the error in phi_0 estimation.
        
        ----------------------------------------   BRUTE FORCE SEARCH    ----------------------------------------
        
        The brute force solution generates thousands os points in the interval (0, 2pi) and picks the one with the
        lowest minimun square error (MSE).
        
        ---------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        AoD  : ndarray
            Angles of departure of the NLOS ray propagation, mesured in the BS from de positive x-axis in the 
            non-colckwise.
            
        AoA  : ndarray
            Angles of arrival of the NLOS ray propagation, mesured in the UE from de positive x-axis in the 
            non-colckwise sense. The value of phi_0 can modify the orientation of the x-axis.
        
        dels : ndarray
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
        phi0_est: ndarray
            Offset angle estimated of the UE orientation.

        """
        
        if not Npoint:
            Npoint = self.NLinePointsPerIteration
            
        philow = 0
        phihigh = 2*np.pi
        interval = np.linspace(philow, phihigh, Npoint).reshape(-1,1)
        dist = np.zeros(Npoint)

        for npoint in range(Npoint):
            dist[npoint] = self.feval_wrapper_AllPathsByGroupsFun(interval[npoint], AoD, AoA, dels, group_method)
        
        distint = np.argmin(dist)

        return(interval[distint])
    
    def solvePhi0ForAllPaths(self, AoD, AoA, dels, init_phi0, group_method='drop1', RootMethod='lm'):
        """Performs the estimation of the value of phi0 using the scipy.optimize.root function.
        
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
            *** lm (Levenberg–Marquardt algorithm): especified for solving non-linear least squares problems.
        
        
        ---------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        AoD  : ndarray
            Angles of departure of the NLOS ray propagation, mesured in the BS from de positive x-axis in the 
            non-colckwise.
            
        AoA  : ndarray
            Angles of arrival of the NLOS ray propagation, mesured in the UE from de positive x-axis in the 
            non-colckwise sense. The value of phi_0 can modify the orientation of the x-axis.
            
        dels : ndarray
            Delays introduced by the NLOS ray propagations.
        
        init_phi0 : ndarray
            Hint or guess about the value of phi0.
            
        group_method : ndarray, optional
            Path grouping strategy.
            *** Options: 'drop1', '3-path', 'random'
            *** Default value is 'drop1'.
        
        RootMethod: str, optional
            Type of solver.
            ** lm (Levenberg–Marquardt algorithm): especified for solving non-linear least squares problems.
            ** hybr (Hybrid method): it uses Powell's dog leg method, an iterative optimisation algorithm for the solution of 
            non-linear least squares problems.

        Returns
        -------
        phi0_est: ndarray
            Offset angle estimated of the UE orientation.
                  
        """
       
        res = opt.root(self.feval_wrapper_AllPathsByGroupsFun, x0=init_phi0, args=(AoD, AoA, dels, group_method), method=self.RootMethod)
        if not res.success:
        #print("Attempting to correct initialization problem")
            niter = 0 
            while (not res.success) and (niter<1000):
                res = opt.root(self.feval_wrapper_AllPathsByGroupsFun, x0=2*np.pi*np.random.rand(1), args=(AoD, AoA, dels, group_method), method=self.RootMethod)
                niter += 1
        #print("Final Niter %d"%niter)
        if res.success:
            return (res.x,res.cov_x)
        else:
            print("ERROR: phi0 root not found")
            return (np.array(0.0),np.inf)

    def computeAllLocationsFromPaths(self, AoD, AoA, dels, Npoint=None, hint_phi0=None, phi0_method='fsolve', group_method='drop1', RootMethod='lm'):
        """Performs the estimation of the phi_0 especified by the parameter method, and returns the position 
        of the UE for this angle.

        The parameter method calls:
        - 'fsolve' : 
            calls solvePhi0ForAllPaths() to estimate phi0.

        - 'brute' : 
            calls bisectPhi0ForAllPaths() to estimate phi0.
            
        ---------------------------------------------------------------------------------------------------------

        Parameters
         ----------
        AoD  : ndarray
            Angles of departure of the NLOS ray propagation, mesured in the BS from de positive x-axis in the 
            non-colckwise.
            
        AoA  : ndarray
            Angles of arrival of the NLOS ray propagation, mesured in the UE from de positive x-axis in the 
            non-colckwise sense. The value of phi_0 can modify the orientation of the x-axis.
            
        dels : ndarray
            Delays introduced by the NLOS ray propagations.
            
        hint_phi0 : ndarray, optional
            Hint or guess about the value of phi0. 
    
        phi0_method: str, optional
            Method used to performs the value estimation of phi0.
            *** Options: 'fsolve', 'bisec', 'fsolve_linear'
            *** Default value is 'fsolve'.
        
        group_method : ndarray, optional
            Path grouping strategy.
            *** Options: 'drop1', '3-path', 'random'
            *** Default value is 'drop1'.
            
        Returns
        -------
        phi0_est: ndarray
            Offset angle estimated of the UE orientation.
            
        d0x : ndarray
            x-coordinate of the posible position of the UE.
            
        d0y : ndarray
            y-coordinate of the posible position of the UE.
            
        vy : ndarray
            y-coordinate of the reflector position in the NLOS path.
            
        vx : ndarray
            x-coordinate of the reflector position in the NLOS path.
            
        cov_phi0: ndarray
            The inverse of the Hessian matrix of phi0.
                        
        """

        if phi0_method == 'fsolve':
            
            if (hint_phi0 == None):
                #coarse linear approximation for initialization
                init_phi0 = self.brutePhi0ForAllPaths(AoD, AoA, dels, Npoint, group_method)
                (Phi0_est,cov_Phi0) = self.solvePhi0ForAllPaths(AoD, AoA, dels, init_phi0, group_method, RootMethod)

            else:
                init_phi0 = hint_phi0
                (phi0_est, cov_phi0) = self.solvePhi0ForAllPaths(AoD, AoA, dels, init_phi0, group_method, RootMethod)

        elif phi0_method == 'brute':
            phi0_est = self.brutePhi0ForAllPaths(AoD, AoA, dels, Npoint, group_method)
            cov_phi0 = np.pi/self.NLinePointsPerIteration
        else:
            print("unsupported method")
            return(None)
        
        (d0x, d0y, tauerr, vx, vy) = self.computeAllPaths(AoD, AoA, dels, phi0_est)

        return(phi0_est, d0x, d0y, tauerr, vx, vy, cov_phi0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--load',  help='CVS name file lo load channel parameters')
    parser.add_argument('--AoD', help='Array with Angle of Departure values')
    parser.add_argument('--AoA', help='Array with Angle of Arrival values')
    parser.add_argument('--dels', help='Array with NLOS paths delays values')
    
    args = parser.parse_args()
    
    if args.load:
        #cargamos fichero CSV
        data = pd.read_csv(args.load, header = 0)
        AoD = data['AoD']
        AoA = data['AoA']
        dels = data['dels']
    
    else:
        AoA = args.AoA
        AoD = args.AoD
        dels = args.dels
