#!/usr/bin/python

import numpy as np
import scipy.optimize as opt

class MultipathLocationEstimator:
    """Class used to perform the calculation and estimation of the UE (User Equipment) position in 2D. 
    
    The main goal of the MultipathLocationEstimator class is try to recover the UE position, define as (x0, y0), 
    estimating the value of user offset (psi_0) from the knowledge of the set of the Angles Of Departure (AoD), 
    Angles Of Arrival (AoA) and the delays introduced by the multipath channels.
    
    ...
    

    Attributes
    ---------
    Npoint : int, optional
        Total point divisions in the minimization range of search.
        
    Nref : int, optional
        Number of iterations of the algorithm to find and obtain the minimal value from the destiny fucntion.
        
    Ndiv : int, optional
        Total divisions of the new range of search where the minimal solution was found in one iteration.
        
    RootMethod: str, optional
        Type of solver.
        ** lm (Levenberg–Marquardt algorithm): especified for solving non-linear least squares problems.
    

    Methods
    -------
    computePosFrom3PathsKnownPsi0(self,AoD,AoA,dels,psi0_est)
        Performs the calculation of all the posible UE vector positions using the 3-path method.
        
    computeAllPathsWithParams(self,AoD,AoA,dels,x0,y0,psi0_est)
        Performs the calculation of all the scatters vector positions.
    
    computeAllPathsLinear(self,AoD,AoA,dels,psi0_est)
        Performs the calculation of all the posible UE vector positions using the linear method.
    
    feval_wrapper_AllPathsLinear_drop1(self,x,AoD,AoA,dels)
        Defines the function to solve all the posible UE vector positions using the linear method by grouping 
        the paths into sets of (Npath-1) paths to compute (Npath-1) estimations.
    
    feval_wrapper_AllPathsLinear_random(x,AoD,AoA,dels)
        Defines the function to solve all the posible UE vector positions using the linear method by grouping 
        the paths into sets of (Npath//2) random paths to compute (Nlines) estimations.
        
    solvePsi0ForAllPaths_linear(self,AoD,AoA,dels,init_psi0)
        Performs the estimation of the value of psi0 using the scipy.optimize.root method and 
        feval_wrapper_AllPathsLinear_drop1 function.
    
    bisectPsi0ForAllPaths(self,AoD,AoA,dels,Npoint=None,Niter=None,Ndiv=None)
        Performs the estimation of the value of psi0 using the bisection method.
    
    feval_wrapper_3PathPosFun(self,x,AoD,AoA,dels)
        Defines the function to solve all the posible UE vector positions using the 3-path method.
    
    solvePsi0ForAllPaths(self,AoD,AoA,dels,init_psi0)
        Performs the estimation of the value of psi0 using the scipy.optimize.root method and
        feval_wrapper_3PathPosFun function.
    
    computeAllLocationsFromPaths(self,AoD,AoA,dels,method='fsolve',hint_psi0=None)
        Performs the estimation of the psi0 especified by the parameter method, and returns the position 
        of the UE for this angle.
    
    """
    
    
    def __init__(self,Npoint=100,Nref=10,Ndiv=2,RootMethod='lm'):
        """Constructs all the attributes for the MultipathLocationEstimator object.

        Parameters
        ----------

        Npoint : int, optional
            Total point divisions in the minimization range of search.

        Nref : int, optional
            Number of iterations of the algorithm to find and obtain the minimal value from the destiny fucntion.

        Ndiv : int, optional
            Total divisions of the new range of search where the minimal solution was found in one iteration.

        RootMethod: str, optional
            Type of solver.
            ** lm (Levenberg–Marquardt algorithm): especified for solving non-linear least squares problems.
        
        """
        
        self.NLinePointsPerIteration=Npoint
        self.NLineRefinementIterations=Nref
        self.NLineRefinementDivision=Ndiv
        self.RootMethod=RootMethod
        self.c=3e8
    
    
    def computePosFrom3PathsKnownPsi0(self,AoD,AoA,dels,psi0_est):
        """Performs the calculation of all the posible UE vector positions using the 3-path method.
        
        The value of the UE position is obtained by using the 3-path algorithm estimation. For this purpose the 
        algorithm takes the sets of the AoA, AoD and delays values of 3 different NLOS paths and obtaine all 
        the posibles position vectors (x0, y0) of the UE, for the range of values especified by psi0_est.
        
        --------------------------------------    3-PATH METHOD    ----------------------------------------
        
        Considering having the AoA, AoD and delays values for Npaths: {1,2,3, . . . , Npath}, the algorithm is 
        divided in the following steps:
        
        STEP (1):
            Obtain the unknown values for T, S, P, Q and Dl for the Npaths.
        
        STEP (2):
        To understand how to compute Idp, Slp and y0 we define:
            i = 1,2, . . ., Npath-2
            j = 2,3, . . ., Npath-1
            k = 3,4, . . ., Npath
                
            To compute Idp:
            Idp(i,j) = (Dl(i) - Dl(j)) / (P(i) - P(j))
            
            To compute Slp:
            Slp(i,j) = (Q(j) - Q(i)) / (P(i) - P(j))
            
            With this, we obtain the values from the combination of 2 paths:
            e.g: 
                Idp{1,2}, Idp{2,3} ... Idp{Npath-1,Npath}
                Slp{1,2}, Slp{2,3} ... Slp{Npath-1,Npath}
            
        STEP (3):  
            To make the system linear we need to add one more path to add one new ecuation:
            
            To compute y0:
            y0(i,j,k) = (Idp(i,j) - Idp(j,k)) / (Slp(i,j) - Slp(j,k))
                
            With this, we obtain the values from the combination of 3 paths:
            e.g: 
                y0{1,2,3}, y0{2,3,4} ... Idp{Npath-2,Npath-1,Npath}
                        
        ** For Npaths we can obtain Npath-2 posible positions estimation, because paths parameters are agruped by 3
    
        Parameters
        ----------
        AoD  : ndarray
            Angles of departure of the NLOS ray propagation, mesured in the BS from de positive x-axis in the 
            non-colckwise.
            
        AoA  : ndarray
            Angles of arrival of the NLOS ray propagation, mesured in the UE from de positive x-axis in the 
            non-colckwise sense. The value of psi_0 can modify the orientation of the x-axis.
            
        dels : ndarray
            Delays introduced by the NLOS ray propagations.
            
        psi0_est: ndarray
            Offset angle of the UE orientation in the clockwise sense.

        Returns
        -------
        x0 : ndarray
            Array with all the posibles x-coordinate componets of the posible position of the UE.
            
        y0 : ndarray
            Array with all the posibles y-coordinate componets of the posible position of the UE.
            
        tauE : ndarray
            Delay difference between the LOS path and the NLOS path propagation.
              
        """
        
        tgD = np.tan(AoD)
        tgA = np.tan(np.pi-AoA-psi0_est)
        siD = np.sin(AoD)
        siA = np.sin(np.pi-AoA-psi0_est)
        coD = np.cos(AoD)
        coA = np.cos(np.pi-AoA-psi0_est)
        
        T=(1/tgD+1/tgA)
        S=(1/siD+1/siA)
        P=S/T
        Q=P/tgA-1/siA
        #P=(siD+siA)/(coD*siA+coA*siD)
        #Q=((siA-coD*tgA)/(coD*coA+siD*siA))
        Dl = dels*self.c
         
        Idp=(Dl[0:-1]-Dl[1:])/(P[...,0:-1]-P[...,1:])
        Slp=(Q[...,0:-1]-Q[...,1:])/(P[...,0:-1]-P[...,1:])
        
        y0=(Idp[...,0:-1]-Idp[...,1:])/(Slp[...,0:-1]-Slp[...,1:])
        x0=Idp[...,0:-1]-y0*Slp[...,0:-1]
        
        l0Err=x0*P[...,0:-2]+y0*Q[...,0:-2]-Dl[0:-2]
        l0=np.sqrt(x0**2+y0**2)
        tauE=(l0-l0Err)/self.c
        
        return(x0,y0,tauE)
    
    
    def computeAllPathsWithParams(self,AoD,AoA,dels,x0,y0,psi0_est):
        """Performs the calculation of all the scatters vector positions.
        
        Parameters
        ----------
        AoD  : ndarray
            Angles of departure of the NLOS ray propagation, mesured in the BS from de positive x-axis in the 
            non-colckwise.
            
        AoA  : ndarray
            Angles of arrival of the NLOS ray propagation, mesured in the UE from de positive x-axis in the 
            non-colckwise sense. The value of psi_0 can modify the orientation of the x-axis.
            
        dels : ndarray
            Delays introduced by the NLOS ray propagations.
            
        x0 : ndarray
            x-coordinate of the posible position of the of UE.
            
        y0 : ndarray
            y-coordinate of the posible position of the of UE.
            
        psi0_est: ndarray
            Offset angle of the UE orientation in the clockwise sense.

        Returns
        -------
        vy : ndarray
            y-coordinate of the posible scatter position in the NLOS path propagation.
            
        vx : ndarray
            x-coordinate of the posible scatter position in the NLOS path propagation.
                  
        """
            
        tgD = np.tan(AoD)
        tgA = np.tan(np.pi-AoA-psi0_est)
        
        #T=(1/tgD+1/tgA)        
        #vy=(x0+y0/tgA)/T
        vy=np.where(tgD!=0, (tgA*x0+y0)/(tgA/tgD+1) , 0)
        vx=np.where(tgD!=0, vy/tgD, x0+y0/tgA)
            
        return(vx,vy)
    
    
    def computeAllPathsLinear(self,AoD,AoA,dels,psi0_est):
        """Performs the calculation of all the posible UE vector positions using the linear method.
        
        The value of the UE position is obtained by using the linear algorithm estimation. For this purpose the 
        algorithm takes the sets of the AoA, AoD and delays values of the all the NLOS Npaths and obtained all 
        the posibles position vectors (x0, y0) of the UE, for the range of values especified by psi0_est.
        
        ----------------------------------    GENERALIZED LINEAR METHOD    -------------------------------------

        The generalized linear method is divided into the following steps:
        
        STEP (1): 
            With Npaths, the totaL number of NLOS paths it, can be wrotten the following system of linear equations:
            
                                                            A @ x = b      (1)
            
            
                                    | P{1}     Q{1}     -1|             |  Dl[1]  |
                                    | P{2}     Q{1}     -1|             |  Dl[2]  |
                                    | P{3}     Q{1}     -1|   |x0est|   |  Dl[3]  |
                                    | P{4}     Q{1}     -1| @ |y0est| = |  Dl[4]  |  (2)
                                    |                     |   |l0err|   |         |
                                    |           ...       |             |   ...   |
                                    |                     |             |         |
                                    | P{Npath} P{Npath} -1|             |Dl[Npath]|


        STEP (2):
            Compute the minimun least-squares solution:
            
            The method numpy.linalg.lstsq: compute least-squares solution to equation . What means that obtains
            the vector x that approximately solves the equation (1). 
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
            non-colckwise sense. The value of psi_0 can modify the orientation of the x-axis.
            
        dels : ndarray
            Delays introduced by the NLOS ray propagations.
            
        psi0_est: ndarray
            Offset angle of the UE orientation in the clockwise sense.

        Returns
        -------
        x0est : ndarray
            Array with all the posibles x-coordinate componets of the posible position of the UE.
            
        y0est : ndarray
           Array with all the posibles y-coordinate componets of the posible position of the UE.
            
        tauEest : ndarray
            Delay difference between the LOS path and the NLOS path propagation.
            
        vyest : ndarray
           Array with all the posibles x-coordinate componets of the posible position of the scatters.
            
        vxest : ndarray
            Array with all the posibles x-coordinate componets of the posible position of the scatters.

        """
        
        tgD = np.tan(AoD)
        tgA = np.tan(np.pi-AoA-psi0_est)
        siD = np.sin(AoD)
        siA = np.sin(np.pi-AoA-psi0_est)
        coD = np.cos(AoD)
        coA = np.cos(np.pi-AoA-psi0_est)
        
        #T=(1/tgD+1/tgA)
        #S=(1/siD+1/siA)
        #P=S/T
        P=(siD+siA)/(coD*siA+coA*siD)
        P[np.isnan(P)]=1
        Q=P/tgA-1/siA
        Dl = dels*self.c
        
        result=np.linalg.lstsq(np.column_stack([P,Q,-np.ones_like(P)]),Dl,rcond=None)
        
        (x0est,y0est,l0err)=result[0]
        
        l0est=np.sqrt(x0est**2+y0est**2)
        tauEest=(l0est-l0err)/self.c
        
        #vyest=(x0est+y0est/tgA)/T
        vyest=np.where(tgD!=0, (tgA*x0est+y0est)/(tgA/tgD+1), 0)
        vxest=np.where(tgD!=0, vyest/tgD, x0est+y0est/tgA)
            
        return(x0est,y0est,tauEest,vxest,vyest)
    
    
    def feval_wrapper_AllPathsLinear_drop1(self,x,AoD,AoA,dels):
        """Defines the function to solve all the posible UE vector positions using the linear method by grouping 
        the paths into sets of (Npath-1) paths to compute (Npath-1) estimations.
        
        The value of the UE position is obtained by using the linear algorithm estimation. For this purpose the 
        algorithm takes the sets of the AoA, AoD and delays values of the all the NLOS (Npath-1) paths combinations 
        and obtained all the posibles position vectors (x0, y0) of the UE, for the range of values especified by 
        psi_0 and returns the value of the function with the non-linear MMSE.
        
        
        -------------------------    WRAPPING AND DROP ONE PATH LINEAR METHOD    ----------------------------
        
        The wrapping and drop one path linear method is divided into the following steps:
        
        STEP (1): 
            Divide the set {1,2,3 . . . Npath}, where Npath is the total number of NLOS paths  into several groups 
            of paths G1, G2 , G3. . . Gm. Each group is defined as the group of all paths except de m-th one.
            So the Gm group will inclue: Gm = {1, . . . , m-1 , m+1 , . . ., Npath}
            
            E.g.:
            The total number of paths, Npath = 4, so groups includes paths as:
            G1 = {1,3,4}
            G2 = {1,2,4}
            G3 = {1,2,3}

        STEP (2):
            Obtain separate solutions of the linear system for each group G1, G2 . . . Gm using the 
            computeAllPathsLinear method.
        
        STEP (3):
            Find the non-linear MMSE solution

        ----------------------------------------    ADVANTAGES    -------------------------------------------
        
        * Better position estimation than the 3-path algorithm.
        * The algorithm does not diverge as sharply as 3paths.
        
        
        -----------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        x: ndarray
            Function unknown (psi0).
            
        AoD  : ndarray
            Angles of departure of the NLOS ray propagation, mesured in the BS from de positive x-axis in the 
            non-colckwise.
            
        AoA  : ndarray
            Angles of arrival of the NLOS ray propagation, mesured in the UE from de positive x-axis in the 
            non-colckwise sense. The value of psi_0 can modify the orientation of the x-axis.
            
        dels : ndarray
            Delays introduced by the NLOS ray propagations.
            
        Returns
        -------
        Returns the mathematical polinomial function based in the minimum mean square error (MMSE) equation.

        """
        
        Npath=AoD.size
        x0all=np.zeros(Npath)
        y0all=np.zeros(Npath)
        tauEall=np.zeros(Npath)
        for gr in range(Npath):
            # AoD[np.arange(Npath)!=gr takes all AoD path angles less the gr path
            (x0all[gr],y0all[gr],tauEall[gr],_,_)=self.computeAllPathsLinear(AoD[np.arange(Npath)!=gr],AoA[np.arange(Npath)!=gr],dels[np.arange(Npath)!=gr],x)
        return(np.sum(np.abs(x0all-np.mean(x0all,x0all.ndim-1,keepdims=True))**2+np.abs(y0all-np.mean(y0all,x0all.ndim-1,keepdims=True))**2+np.abs(self.c*tauEall-np.mean(self.c*tauEall,x0all.ndim-1,keepdims=True))**2,x0all.ndim-1))
    
    
    def feval_wrapper_AllPathsLinear_random(self,x,AoD,AoA,dels):
        """Defines the function to solve all the posible UE vector positions using the linear method by grouping 
        the paths into sets of (Npath//2) random paths to compute (Nlines) estimations.
        
        The value of the UE position is obtained by using the linear algorithm estimation. For this purpose the 
        algorithm takes the sets of the AoA, AoD and delays values of the (Npath//2) NLOS random paths combinations 
        and obtained all the posibles position vectors (x0, y0) of the UE, for the range of values especified by 
        psi_0 and returns the value of the function with the non-linear MMSE.
        
        
        -------------------------    RANDOM WRAPPING PATH LINEAR METHOD    ----------------------------------
        
        The wrapping and drop one path linear method is divided into the following steps:
        
        STEP (1): 
            Divide the set {1,2,3 . . . Npath}, where Npath is the total number of NLOS paths into Nlines groups 
            of randomly paths distribution G1, G2 , G3. . . Gm. Each group is defined as the group of Npath//2 paths
            randomly distributed.
            
            E.g.:
            The total number of paths, Npath = 4, and the number of iterations, Nlines = 2 groups includes paths
            randomly as:
            G1 = {3,1}
            G2 = {2,1}
            
        STEP (2):
            Obtain separate solutions of the linear system for each group G1, G2 . . . Gm using the 
            computeAllPathsLinear method.
        
        STEP (3):
            Find the non-linear MMSE solution
        
        -----------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        x : ndarray
            Function unknown (psi0).
            
        AoD  : ndarray
            Angles of departure of the NLOS ray propagation, mesured in the BS from de positive x-axis in the 
            non-colckwise.
            
        AoA  : ndarray
            Angles of arrival of the NLOS ray propagation, mesured in the UE from de positive x-axis in the 
            non-colckwise sense. The value of psi_0 can modify the orientation of the x-axis.
            
        dels : ndarray
            Delays introduced by the NLOS ray propagations.
            
        Returns
        -------
        Returns the mathematical polinomial function based in the minimum mean square error (MMSE) equation.

        """
        
        Npath=AoD.size
        Nlines=5
        x0all=np.zeros(Nlines)
        y0all=np.zeros(Nlines)
        tauEall=np.zeros(Nlines)
        indices=np.random.choice(Npath,(Nlines,Npath//2))
        for gr in range(Nlines):
            #AoD[indices[gr,:] takes all random AoD angles from the random paths index generated
            (x0all[gr],y0all[gr],tauEall[gr],_,_)=self.computeAllPathsLinear(AoD[indices[gr,:]],AoA[indices[gr,:]],dels[indices[gr,:]],x)
        return(np.sum(np.abs(x0all-np.mean(x0all,x0all.ndim-1,keepdims=True))**2+np.abs(y0all-np.mean(y0all,x0all.ndim-1,keepdims=True))**2+np.abs(self.c*tauEall-np.mean(self.c*tauEall,x0all.ndim-1,keepdims=True))**2,x0all.ndim-1))
    
    
    def solvePsi0ForAllPaths_linear(self,AoD,AoA,dels,init_psi0):
        """Performs the estimation of the value of psi0 using the scipy.optimize.root method and 
        feval_wrapper_AllPathsLinear_drop1 function.
        
        The value of the estimated offset angle of the UE orientation is obtained by finding the zeros of the 
        given vector function defined by feval_wrapper_AllPathsLinear_drop1. For this purpose, we use the method 
        root() which is used to find the solutions of the this vector function.


        ---------------------------------    MULTIDIMENSIONAL ROOT METHOD    ------------------------------------
        
        In this case we make use of the root method of scipy.optimize.root(), used to solving a non-linear equation 
        with parameters.
        
        TO USE THE METHOD:
        sol = scipy.optimize.root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None, options=None).
        
        Parameters
        ----------
        fun : callable
            The vector function to find a root of.
        
        x0 : numpy.ndarray
            Initial guess.

        args : tuple, optional
            Extra arguments passed to the objective function and its Jacobian. These are other variables 
            that we aren't finding the roots for. The parameters have to appear in the same order as the function.
            
        method : str
            Type of solver.
            *lm (Levenberg–Marquardt algorithm): especified for solving non-linear least squares problems.
        
        
        ---------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        AoD  : ndarray
            Angles of departure of the NLOS ray propagation, mesured in the BS from de positive x-axis in the 
            non-colckwise.
        
        AoA  : ndarray
            Angles of arrival of the NLOS ray propagation, mesured in the UE from de positive x-axis in the 
            non-colckwise sense. The value of psi_0 can modify the orientation of the x-axis.
            
        dels : ndarray
            Delays introduced by the NLOS ray propagations.

        init_psi0 : ndarray
            Hint about the value of psi0.

        Returns
        -------
        psi0_est: ndarray
            Offset angle estimated of the UE orientation.
        
        """
    
        res=opt.root(self.feval_wrapper_AllPathsLinear_drop1,x0=init_psi0,args=(AoD,AoA,dels),method=self.RootMethod)
        #print(res)
        if not res.success:
        #print("Attempting to correct initialization problem")
            niter=0 
            while (not res.success) and (niter<1000):
                res=opt.root(self.feval_wrapper_AllPathsLinear_drop1,x0=2*np.pi*np.random.rand(1),args=(AoD,AoA,dels),method=self.RootMethod)
                niter+=1
        #print("Final Niter %d"%niter)
        if res.success:
            return (res.x,res.cov_x)
        else:
            print("ERROR: Psi0 root not found")
            return (np.array(0.0),np.inf)
    
    
    def bisectPsi0ForAllPaths(self,AoD,AoA,dels,Npoint=None,Niter=None,Ndiv=None):
        """Performs the estimation of the value of psi0 using the bisection method.
        
        The value of the estimated offset angle of the UE orientation is obtained by using the bisection algorithm.
        For this purpose the method divides the range of the posible values of psi0, among 0 and 2pi into Npoints 
        and minimize the error. The method reduces recurrently the range till minimize the error in psi0.
        
        ----------------------------------------   BRUTE FORCE SEARCH    ----------------------------------------
        
        The brute force solution generates thousands os points in the interval (0, 2pi) and picks the one with the
        lowest minimun square error (MSE)
        
        
        ----------------------------------------    BISECTION METHOD    ------------------------------------------
        
        The bisection method is divided into the following steps:

        STEP (1):
            Firstly divide the vector psi0 in Npoints for the range from 0 to 2pi.
        
        STEP (2):
            The function computePosFrom3PathsKnownPsi0 is called to compute the values of the posible x0, y0 and tauE
            for the psi0 points.
            
        STEP (3):
            The MMSE error is computed
            
        STEP (4):
            The point of the psi0 interval which the MMSE is minimized is obtained.
            
        STEP (5):
            The search value range of psi0 around the calculated minimum point is reduced and divided again into
            Ndivs points.
        
        STEP (6):
            Return to STEP 2 for Niter
        
        ---------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        AoD  : ndarray
            Angles of departure of the NLOS ray propagation, mesured in the BS from de positive x-axis in the 
            non-colckwise.
            
        AoA  : ndarray
            Angles of arrival of the NLOS ray propagation, mesured in the UE from de positive x-axis in the 
            non-colckwise sense. The value of psi_0 can modify the orientation of the x-axis.
        
        dels : ndarray
            Delays introduced by the NLOS ray propagations.

        Returns
        -------
        psi0_est: ndarray
            Offset angle estimated of the UE orientation.

        """
            
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
#                (x0all[npath,:],y0all[npath,:])= self.computePosFrom3PathsKnownPsi0(AoD[npath:npath+3],AoA[npath:npath+3],dels[npath:npath+3],interval)
            (x0all,y0all,tauEall)=self.computePosFrom3PathsKnownPsi0(AoD,AoA,dels,interval)
            dist=np.sum(np.abs(x0all-np.mean(x0all,x0all.ndim-1,keepdims=True))**2+np.abs(y0all-np.mean(y0all,x0all.ndim-1,keepdims=True))**2+np.abs(self.c*tauEall-np.mean(self.c*tauEall,x0all.ndim-1,keepdims=True))**2,x0all.ndim-1)
            distint=np.argmin(dist)
            philow=interval[distint]-np.pi/(Ndiv**nit)
            phihigh=interval[distint]+np.pi/(Ndiv**nit)
#        if (dist[distint]>1):
#            print("ERROR: psi0 recovery algorithm converged loosely psi0: %.2f d: %.2f"%((np.mod(interval[distint],2*np.pi),dist[distint])))
        return(interval[distint])
    
    
    def feval_wrapper_3PathPosFun(self,x,AoD,AoA,dels):
        """Defines the function to solve all the posible UE vector positions using the 3-path method.
        
        Parameters
        ----------
        x : ndarray
            Function unknown (psi0).
            
        AoD  : ndarray
            Angles of departure of the NLOS ray propagation, mesured in the BS from de positive x-axis in the 
            non-colckwise.
            
        AoA  : ndarray
            Angles of arrival of the NLOS ray propagation, mesured in the UE from de positive x-axis in the 
            non-colckwise sense. The value of psi_0 can modify the orientation of the x-axis.
            
        dels : ndarray
            Delays introduced by the NLOS ray propagations.

        Returns
        -------
        Returns the mathematical polinomial function based in the minimum mean square error (MMSE) equation.

        """
        
        (x0all,y0all,tauEall)=self.computePosFrom3PathsKnownPsi0(AoD,AoA,dels,np.asarray(x))
        return(np.sum(np.abs(x0all-np.mean(x0all,x0all.ndim-1,keepdims=True))**2+np.abs(y0all-np.mean(y0all,x0all.ndim-1,keepdims=True))**2+np.abs(self.c*tauEall-np.mean(self.c*tauEall,x0all.ndim-1,keepdims=True))**2,x0all.ndim-1))
    
    
    def solvePsi0ForAllPaths(self,AoD,AoA,dels,init_psi0):
        """Performs the estimation of the value of psi0 using the scipy.optimize.root method and
        feval_wrapper_3PathPosFun function.
        
        The value of the estimated offset angle of the UE orientation is obtained by finding the zeros of the 
        given vector function defined by feval_wrapper_3PathPosFun. For this purpose, we use the method root()
        which is used to find the solutions of the this vector function.


        ---------------------------------    MULTIDIMENSIONAL ROOT METHOD    ------------------------------------
        
        In this case we make use of the root method of scipy.optimize.root(), used to solving a non-linear equation 
        with parameters.
        
        TO USE THE METHOD:
        sol = scipy.optimize.root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None, options=None).
        
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
            *lm (Levenberg–Marquardt algorithm): especified for solving non-linear least squares problems.
        
        
        ---------------------------------------------------------------------------------------------------------
        
        Parameters
        ----------
        AoD  : ndarray
            Angles of departure of the NLOS ray propagation, mesured in the BS from de positive x-axis in the 
            non-colckwise.
            
        AoA  : ndarray
            Angles of arrival of the NLOS ray propagation, mesured in the UE from de positive x-axis in the 
            non-colckwise sense. The value of psi_0 can modify the orientation of the x-axis.
            
        dels : ndarray
            Delays introduced by the NLOS ray propagations.
        
        init_psi0 : ndarray
            Hint or guess about the value of psi0.


        Returns
        -------
        psi0_est: ndarray
            Offset angle estimated of the UE orientation.
                  
        """
        
        res=opt.root(self.feval_wrapper_3PathPosFun,x0=init_psi0,args=(AoD,AoA,dels),method=self.RootMethod)
        if not res.success:
        #print("Attempting to correct initialization problem")
            niter=0 
            while (not res.success) and (niter<1000):
                res=opt.root(self.feval_wrapper_3PathPosFun,x0=2*np.pi*np.random.rand(1),args=(AoD,AoA,dels),method=self.RootMethod)
                niter+=1
        #print("Final Niter %d"%niter)
        if res.success:
            return (res.x,res.cov_x)
        else:
            print("ERROR: Psi0 root not found")
            return (np.array(0.0),np.inf)
    
    def computeAllLocationsFromPaths(self,AoD,AoA,dels,method='fsolve',hint_psi0=None):
        """Performs the estimation of the psi0 especified by the parameter method, and returns the position 
        of the UE for this angle.

        The parameter method calls:
        - 'fsolve' : 
            solvePsi0ForAllPaths() to estimate psi0 and computePosFrom3PathsKnownPsi0 to return the UE position.

        - 'biscec' : 
            bisectPsi0ForAllPaths() to estimate psi0 and computePosFrom3PathsKnownPsi0 to return the UE position.

        - 'fsolve_linear' : 
            solvePsi0ForAllPaths_linear() to estimate psi0 and computeAllPathsWithParams to return the UE 
            position.

        Parameters
         ----------
        AoD  : ndarray
            Angles of departure of the NLOS ray propagation, mesured in the BS from de positive x-axis in the 
            non-colckwise.
            
        AoA  : ndarray
            Angles of arrival of the NLOS ray propagation, mesured in the UE from de positive x-axis in the 
            non-colckwise sense. The value of psi_0 can modify the orientation of the x-axis.
            
        dels : ndarray
            Delays introduced by the NLOS ray propagations.
    
        method: str, optional
            Method used to performs the value estimation of psi0.
            ** Options: 'fsolve', 'bisec', 'fsolve_linear'
            ** Default value is 'fsolve'.
        
        hint_psi0 : ndarray, optional
            Hint or guess about the value of psi0.   
         
            
        Returns
        -------
        psi0_est: ndarray
            Offset angle estimated of the UE orientation.
            
        x0 : ndarray
            x-coordinate of the posible position of the UE.
            
        y0 : ndarray
            y-coordinate of the posible position of the UE.
            
        vy : ndarray
            y-coordinate of the reflector position in the NLOS path.
            
        vx : ndarray
            x-coordinate of the reflector position in the NLOS path.
            
        cov_psi0: ndarray
            The inverse of the Hessian matrix of psi0. A value of None indicates a singular matrix, which means the 
            curvature in parameter psi0 is numerically flat. To obtain the covariance matrix of the parameter psi0, cov_x 
            must be multiplied by the variance of the residuals.
                        
        """
        
        if (hint_psi0==None):
            #coarse linear approximation for initialization
            init_psi0=self.bisectPsi0ForAllPaths(AoD,AoA,dels,Npoint=1000,Niter=1,Ndiv=2)
        else:
            init_psi0=hint_psi0
        if method=='fsolve':
            (psi0_est,cov_psi0)=self.solvePsi0ForAllPaths(AoD,AoA,dels,init_psi0)
        elif method=='bisec':
            psi0_est=self.bisectPsi0ForAllPaths(AoD,AoA,dels)
            cov_psi0=np.pi/self.NLinePointsPerIteration
        elif method=='fsolve_linear':
            (psi0_est,cov_psi0)=self.solvePsi0ForAllPaths_linear(AoD,AoA,dels,init_psi0)
        else:
            print("unsupported method")
            return(None)
        #at this point any 3 paths can be used
        if method=='fsolve_linear':
            (x0,y0,tauerr,vx,vy)= self.computeAllPathsLinear(AoD,AoA,dels,psi0_est)
        else:
            (x0all,y0all,tauEall)= self.computePosFrom3PathsKnownPsi0(AoD,AoA,dels,psi0_est.reshape(-1,1))
            x0=np.mean(x0all,1)
            y0=np.mean(y0all,1)
            (vx,vy)= self.computeAllPathsWithParams(AoD,AoA,dels,x0,y0,psi0_est)
        return(psi0_est,x0,y0,vx,vy,cov_psi0)