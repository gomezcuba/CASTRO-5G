#!/usr/bin/python
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np

import time
import os
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='OMP Zak Simulation')
parser.add_argument('-D', type=int,help='Delay dimension')
parser.add_argument('-V', type=int,help='Doppler dimension')
parser.add_argument('-F', type=int,help='No. independent frames')
parser.add_argument('-S', type=int,help='No. SNR points')
parser.add_argument('-J', type=int,help='No. files to join')
parser.add_argument('--label', type=str,help='Run label')
parser.add_argument('--redo',help='Do not generate new inputs, repeat the prior file', action='store_true')
parser.add_argument('--norun',help='Do not perform simulation, load prior file', action='store_true')
parser.add_argument('--nosave', help='Do not save simulation data to file', action='store_true')
parser.add_argument('--print', help='Generate plots', action='store_true')
parser.add_argument('--show', help='Put plots in windows', action='store_true')
args = parser.parse_args()
#args = parser.parse_args("-D 16 -V 16 -S 11 -F 25 --label part --norun -J 4".split(' '))
#args = parser.parse_args("-D 16 -V 16 -S 11 -F 125 --label largepart --norun -J 8".split(' '))
#args = parser.parse_args("-D 16 -V 16 -S 11 -F 125 --label redopart --norun -J 3".split(' '))
#args = parser.parse_args("-D 16 -V 16 -S 11 -F 125 --label largeOnServer --norun -J 3 --show".split(' '))
#args = parser.parse_args("-D 32 -V 32 -S 11 -F 20 --label extra --norun -J 3 --show".split(' '))
#args = parser.parse_args("-D 8 -V 8 -S 11 -F 125 --label lowDpart --norun -J 8".split(' '))
#args = parser.parse_args("-D 16 -V 16 -S 11 -F 25 --label speedpart --norun -J 8".split(' '))
#args = parser.parse_args("-D 16 -V 16 -S 3 -F 5 --label speedtest".split(' '))
#args = parser.parse_args("-D 32 -V 16 -S 1 -F 5 --label speedtest".split(' '))
#args = parser.parse_args("-D 8 -V 8 -S 5 -F 100 --label test --norun --show --print".split(' '))
args = parser.parse_args("-D 4 -V 4 -S 5 -F 100 --label example --show --print".split(' '))

plt.close("all")

if args.D:
    D=args.D
else:
    D=4
if args.V:
    V=args.V
else:
    V=4
if args.S:
    Npoint=args.S
else:
    Npoint=5
if args.F:
    Nframes=args.F
else:
    Nframes=10
if args.J:
    Nfiles=args.J
else:
    Nfiles=0
    
if args.label:
    outfoldername="./OMPZresults%s-%d-%d-%d-%d"%(args.label,D,V,Npoint,Nframes)
else:
    outfoldername="./OMPZresults-%d-%d-%d-%d"%(D,V,Npoint,Nframes)
if not os.path.isdir(outfoldername):
    os.mkdir(outfoldername)
    
L=D*V

def dirichlet(t,L):
    #implements dirichletL(t/L), always with zeros at integers
    if isinstance(t,np.ndarray):
#        return np.array([dirichlet(titem,L) for titem in t])
#        print(phase,sinnum,sinden)
#        return np.divide(np.exp(1j*np.pi*(L-1)*t/L)*np.sin(np.pi*t)/L,np.sin(np.pi*t/L),out=np.ones_like(sinden), where=np.sin(np.pi*t/L)!=0)
        phase=np.exp(1j*np.pi*(L-1)*t/L)
        sinnum=0j+np.sin(np.pi*t)
        sinden=0j+np.sin(np.pi*t/L)
        res=phase*sinnum/sinden/L
        res[sinden==0]=1
        return(res)
    elif isinstance(t, (list,tuple)):            
        return [dirichlet(titem,L) for titem in t]
    else:
        return 1 if t==0 else np.exp(1j*np.pi*(L-1)*t/L)*np.sin(np.pi*t)/np.sin(np.pi*t/L)/L
    
def zakRecChanBase(tau,nu,D,V):
    L=D*V
    Hout=np.zeros((L,L),dtype=np.complex64)
    dprim=np.arange(D).reshape(D,1)
    vprim=np.arange(V).reshape(1,V)
    
    for d in range(D):#TODO all flat kroenecker at once
        for u in range(V):
            hcoef=np.exp(-2j*np.pi*( (nu)*( d/D ) ))
#accelerated as below
            w1= dirichlet( d-dprim-tau*D ,D).reshape((D,1))
            w2= dirichlet( u-vprim-nu*V ,V).reshape((1,V))
#            dw=d-dprim-tau*D
#            vw=u-vprim-nu*V
#            w1=np.divide(np.exp(1j*np.pi*(D-1)*dw/D)*np.sin(np.pi*dw)/D,np.sin(np.pi*dw/D),out=np.ones(np.shape(dw),dtype=np.complex64), where=np.sin(np.pi*dw/D)!=0)
#            w2=np.divide(np.exp(1j*np.pi*(D-1)*vw/V)*np.sin(np.pi*vw)/V,np.sin(np.pi*vw/V),out=np.ones(np.shape(vw),dtype=np.complex64), where=np.sin(np.pi*vw/V)!=0)
                
#            TODO accelerate all
            hlocal = hcoef*w1*w2
            hlocal[0:(D-d),:]=hlocal[0:(D-d),:]*np.exp(-2j*np.pi*vprim/V)
#            hlocal[0:d,:]=hlocal[0:d,:]*np.exp(-2j*np.pi*vprim/V)
            Hout[d*V+u,:]=(hlocal).reshape(L)            
    return(Hout)
    
def zakRecChanIShift(x,Dd,Dv,D,V):
    L=D*V
    xout=np.roll(x.reshape(D,V),[Dd,Dv],axis=(0,1))
    xout[0:(D-Dd),:]=xout[0:(D-Dd),:]*np.exp(-2j*np.pi*np.arange(V)/V)#TODO revise
    return(xout.reshape(L))
    

def zakRecChanAsMatrix(dels,dops,gains,D,V):
    L=D*V
    Npath=len(dels)
    Hout=np.zeros((L,L),dtype=np.complex64)
    dprim=np.arange(D).reshape(D,1)
    vprim=np.arange(V).reshape(1,V)
    for d in range(D):
        for u in range(V):
            hlocal=np.zeros((D,V),dtype=np.complex64)
            for p in range(Npath):
                hcoef=gains[p]*np.exp(-2j*np.pi*( (dops[p])*( d/D ) ))
#accelerated as below
                w1= dirichlet( d-dprim-dels[p]*D ,D)
                w2= dirichlet( u-vprim-dops[p]*V ,V)
#                dw=d-dprim-dels[p]*D
#                vw=u-vprim-dops[p]*V
#                w1=np.divide(np.exp(1j*np.pi*(D-1)*dw/D)*np.sin(np.pi*dw)/D,np.sin(np.pi*dw/D),out=np.ones(np.shape(dw),dtype=np.complex64), where=np.sin(np.pi*dw/D)!=0)
#                w2=np.divide(np.exp(1j*np.pi*(D-1)*vw/V)*np.sin(np.pi*vw)/V,np.sin(np.pi*vw/V),out=np.ones(np.shape(vw),dtype=np.complex64), where=np.sin(np.pi*vw/V)!=0)
                hlocal = hlocal + hcoef*w1*w2
#accelerated as below
#            quasiperiod=np.exp(-2j*np.pi*((np.arange(V).reshape(1,V))/V)*np.floor((d-np.arange(D).reshape(D,1))/D))
#            Hout[d*V+u,:]=(hlocal*quasiperiod).reshape(L)
            hlocal[0:(D-d),:]=hlocal[0:(D-d),:]*np.exp(-2j*np.pi*vprim/V)
            Hout[d*V+u,:]=(hlocal).reshape(L) 
#TODO this seems slower?
#    Hout=np.zeros((L,L),dtype=np.complex64)
#    for p in range(Npath):
#        Hout=Hout+gains[p]*zakRecChanBase(dels[p],dops[p],D,V)
    return(Hout)
    
class ompCachedZakRunner:
    dicCache={}
    def __init__(self):        
        self.baseDicCache={}    
        self.outDicCache={}
    def freeCacheOfPilot(self,x2Dflat,D,V,X):
        outDicKey=(D,V,X,hash(x2Dflat.data.tobytes())) #optimistic no collision handling
        if outDicKey in self.outDicCache:
            self.outDicCache.pop(outDicKey)
    def ompEstZakChanMatrix(self,x2Dflat,y2Dflat,D,V,tolStop=0,X=1,Nref=0,NiterStop=100):
        L=D*V
        delDic=np.arange(0,1,1/D/X)
        dopDic=np.arange(0,1,1/V/X)
        baseDicKey=(D,V,X)
        outDicKey=(D,V,X,hash(x2Dflat.data.tobytes())) #optimistic no collision handling
#        if X==1:
#            np.roll(x2Dflat.reshape(V,D), [nv,nd], axis=(0,1))
#        else:
        if baseDicKey not in self.baseDicCache:
            Hdic=np.zeros((L,L,L*X*X),dtype=np.complex64)
            for nd in range(len(delDic)):            
                for nv in range(len(dopDic)):
                    Hbasis=zakRecChanBase(delDic[nd],dopDic[nv],D,V)
                    Hdic[:,:,nv+X*V*nd]=Hbasis
            self.baseDicCache[baseDicKey]=Hdic
        else:
            Hdic=self.baseDicCache[baseDicKey]
        if outDicKey not in self.outDicCache:
            YDic=np.zeros((L,L*X*X),dtype=np.complex64)
            for nd in range(len(delDic)):
                for nv in range(len(dopDic)):
                    Hbasis=Hdic[:,:,nv+X*V*nd]
                    YDic[:,nv+X*V*nd]=np.matmul(Hbasis,x2Dflat)
            self.outDicCache[outDicKey]=YDic
        else:
            YDic=self.outDicCache[outDicKey]
        indSup=np.array([],dtype=np.int32)
        gainSup=np.array([],dtype=np.complex64)    
        res=y2Dflat
        currentIter=0
        res_decay=np.sum(np.abs(res)**2)
        if Nref>0:         
            delSup=np.array([])
            dopSup=np.array([])
            baseYSup=np.zeros((L,0),dtype=np.complex64)       
        while (np.sum(np.abs(res)**2)>tolStop) and (currentIter<NiterStop):
#        while ((res_decay)>tolStop*currentIter/L) and (currentIter<NiterStop):
            corr=np.matmul(YDic.transpose().conj(),res)
            imax=np.argmax(np.abs(corr))
            indSup=np.concatenate((indSup,[imax]))                
            if Nref>0:
                dmean=delDic[imax//(V*X)]
                vmean=dopDic[np.mod(imax,V*X)]
                mud=0.5
                muv=0.5
                dlow=dmean-mud*1/D/X
                dhigh=dmean+mud*1/D/X
                vlow=vmean-muv*1/V/X
                vhigh=vmean+muv*1/V/X
                for re in range(Nref):
                    baseLL=np.matmul( zakRecChanBase(dlow,vlow,D,V) , x2Dflat )
                    baseLH=np.matmul( zakRecChanBase(dlow,vhigh,D,V) , x2Dflat )
                    baseHL=np.matmul( zakRecChanBase(dhigh,vlow,D,V) , x2Dflat )
                    baseHH=np.matmul( zakRecChanBase(dhigh,vhigh,D,V) , x2Dflat )
                    corrLL=np.abs(np.matmul(baseLL.conj().T,res))
                    corrLH=np.abs(np.matmul(baseLH.conj().T,res))
                    corrHL=np.abs(np.matmul(baseHL.conj().T,res))
                    corrHH=np.abs(np.matmul(baseHH.conj().T,res))
                    quadrant = np.argmax([corrLL, corrLH, corrHL, corrHH])
                    if quadrant == 0:
                        dhigh=dmean
                        vhigh=vmean
                    elif quadrant == 1:
                        dhigh=dmean
                        vlow=vmean
                    elif quadrant ==2:
                        dlow=dmean
                        vhigh=vmean
                    elif quadrant ==3:
                        dlow=dmean
                        vlow=vmean
                    else:
                        print("ERROR in quadrant")
                    dmean=(dlow+dhigh)/2
                    vmean=(vlow+vhigh)/2
                newbase=np.matmul( zakRecChanBase(dmean,vmean,D,V) , x2Dflat )
                baseYSup=np.concatenate((baseYSup,newbase.reshape(L,1)),1)
                delSup=np.concatenate((delSup,[dmean])) 
                dopSup=np.concatenate((dopSup,[vmean]))
#                    print("OMP pure %f %f, ref %f %f"%(delDic[imax//(M*X)],dopDic[np.mod(imax,M*X)],dmean,vmean))
            else:
                baseYSup=YDic[:,indSup]
            gainSup=np.linalg.lstsq(baseYSup,y2Dflat,rcond=None)[0]
            proj=np.matmul(baseYSup, gainSup )
            res_decay=np.sum(np.abs(res-y2Dflat+proj)**2)
            res=y2Dflat-proj
            currentIter=currentIter+1
        if Nref==0:
            delSup=delDic[indSup//(V*X)]
            dopSup=dopDic[np.mod(indSup,V*X)]
        return(delSup,dopSup,gainSup)

if Npoint>1:
    SNRstep=20/(Npoint-1)
    SNRdB=np.arange(0,Npoint*SNRstep,SNRstep)
else:
    SNRstep=0
    SNRdB=np.array([10])

if not args.norun:
    if args.redo:
        data=np.load(outfoldername+'/numpyresults.npz')
        Allx2D=data["Allx2D"]
        Allz2D=data["Allz2D"]
        AllsymQPSK=data["AllsymQPSK"]
        AllxQPSK=data["AllxQPSK"]
        Alldops=data["Alldops"]
        Alldels=data["Alldels"]
        Allgains=data["Allgains"]
    else:
        Allx2D=np.random.randn(L*Nframes, 2).view(np.complex128).reshape(L,Nframes)/np.sqrt(2)
        Allx2D=Allx2D*np.sqrt(L/np.sum(np.abs(Allx2D)**2,0)).reshape(1,Nframes)
        Allz2D=np.random.randn(L*Nframes, 2).view(np.complex128).reshape(L,Nframes)/np.sqrt(2)
        QPSK=np.exp(2j*np.pi*np.arange(4)/4)
        AllsymQPSK=np.random.randint(0,4,(L,Nframes))
        AllxQPSK=np.exp(2j*np.pi*AllsymQPSK/4)
        
        Npath=3;
        
        Alldops=np.random.rand(Npath,Nframes)
        Alldels=np.random.rand(Npath,Nframes)
        Allgains=np.random.randn(Npath*Nframes, 2).view(np.complex128).reshape(Npath,Nframes)
    
    OMPcases=[
            (1,0,1e6),
            (4,0,1e6),
            (1,2,1e6),
    #        (4,2),
    #        (1,4),
            (1,10,1e6),
            ]
    Nxcases=len(OMPcases)
    
    AllHMSE=np.zeros((Npoint,Nframes,Nxcases))
    AllNpEst=np.zeros((Npoint,Nframes,Nxcases))
    Allruntime=np.zeros((Npoint,Nframes,Nxcases))
    
    AllSER=np.zeros((Npoint,Nframes,Nxcases))
    AlltrueSER=np.zeros((Npoint,Nframes))
    AllXMSE=np.zeros((Npoint,Nframes,Nxcases))
    AlltrueXMSE=np.zeros((Npoint,Nframes))
    
    omprunner = ompCachedZakRunner()
    
    t_start_sim = time.time()
    for fra in range (Nframes):
        t_start_frame = time.time()
        
        x2Dflat=Allx2D[:,fra]
        z2Dflat=Allz2D[:,fra]
        dels=Alldels[:,fra]
        dops=Alldops[:,fra]
        gains=Allgains[:,fra]
        gains=gains/np.sqrt(np.sum(np.abs(gains)**2))
        Htrue=zakRecChanAsMatrix(dels,dops,gains,D,V)
           
        y2DflatNoiseless=np.matmul(Htrue,x2Dflat)
        
        xQPSK=AllxQPSK[:,fra]
        symQPSK=AllsymQPSK[:,fra]
        y2DflatNoiselessData=np.matmul(Htrue,xQPSK)    
        
        for poin in range(Npoint):
            t_start_point = time.time()
            sigma2Z=1/(10**(SNRdB[poin]/10))
            z2DflatSNR = np.sqrt(sigma2Z)*z2Dflat
            y2Dflat=y2DflatNoiseless+z2DflatSNR
            x2Deqflattrue=( np.linalg.lstsq(np.matmul(Htrue.T.conj(),Htrue)+sigma2Z*np.eye(L),np.matmul(Htrue.T.conj(),y2Dflat),rcond=None)[0] )
            AlltrueXMSE[poin,fra]=np.sum(np.abs(x2Dflat-x2Deqflattrue)**2)/np.sum(np.abs(x2Dflat)**2)
            
            y2DflatData=y2DflatNoiselessData+z2DflatSNR
            x2Drvflattrue=( np.linalg.lstsq(np.matmul(Htrue.T.conj(),Htrue)+sigma2Z*np.eye(L),np.matmul(Htrue.T.conj(),y2DflatData),rcond=None)[0] )
            symDecflattrue=np.argmin(np.abs(x2Drvflattrue.reshape(L,1)-QPSK.reshape((1,4))),1)
            AlltrueSER[poin,fra]=np.mean(symDecflattrue!=symQPSK)
            
            for xcas in range(Nxcases):
                X=OMPcases[xcas][0]
                Nref=OMPcases[xcas][1]
                Nmaxiter=OMPcases[xcas][2]
                t_start_case=time.time()
                (delSup,dopSup,gainSup)=omprunner.ompEstZakChanMatrix(x2Dflat,y2Dflat,D,V,sigma2Z*L,X,Nref,Nmaxiter) 
                Allruntime[poin,fra,xcas]=time.time()-t_start_case
                Hest = zakRecChanAsMatrix(delSup,dopSup,gainSup,D,V)
                AllNpEst[poin,fra,xcas]=np.size(delSup)
                AllHMSE[poin,fra,xcas]=np.sum(np.abs(Hest-Htrue)**2)/np.sum(np.abs(Htrue)**2)
            
                x2Drvflat=( np.linalg.lstsq(np.matmul(Hest.T.conj(),Hest)+sigma2Z*np.eye(L),np.matmul(Hest.T.conj(),y2DflatData),rcond=None)[0] )
                AllXMSE[poin,fra,xcas]=np.sum(np.abs(xQPSK-x2Drvflat)**2)/np.sum(np.abs(xQPSK)**2)
                symDecflat=np.argmin(np.abs(x2Drvflat.reshape(L,1)-QPSK.reshape((1,4))),1)
                AllSER[poin,fra,xcas]=np.mean(symDecflat!=symQPSK)
            print("\t%d/%d point frame %d/%d sim %s in %s seconds"%(poin,Npoint,fra,Nframes,args.label,time.time()-t_start_point), flush=True)
#        for xcas in range(Nxcases):
#            X=OMPcases[xcas][0]
#            omprunner.freeCacheOfPilot(x2Dflat,D,V,X)
        print("%d/%d frame sim %s in %s seconds"%(fra,Nframes,args.label,time.time()-t_start_frame),flush=True)
    
    if not args.nosave: 
        np.savez(outfoldername+'/numpyresults.npz',
                 SNRdB=SNRdB,
                 Allx2D=Allx2D,
                 Allz2D=Allz2D,
                 AllsymQPSK=AllsymQPSK,
                 AllxQPSK=AllxQPSK,
                 Alldops=Alldops,
                 Alldels=Alldels,
                 Allgains=Allgains,
                 OMPcases=OMPcases,
                 AllHMSE=AllHMSE,
                 AllNpEst=AllNpEst,
                 Allruntime=Allruntime,
                 AlltrueSER=AlltrueSER,
                 AllSER=AllSER,
                 AlltrueXMSE=AlltrueXMSE,
                 AllXMSE=AllXMSE)
    print("Total Simulation Time:%s seconds"%(time.time()-t_start_sim))
    precomp_indices=[0]
else:
    if Nfiles==0:
        data=np.load(outfoldername+'/numpyresults.npz')        
        SNRdB=data["SNRdB"]
        OMPcases=data["OMPcases"]
        Nxcases=len(OMPcases)
        AllHMSE=data["AllHMSE"]
        AllNpEst=data["AllNpEst"]
        Allruntime=data["Allruntime"]
        AlltrueSER=data["AlltrueSER"]
        AllSER=data["AllSER"]
        AlltrueXMSE=data["AlltrueXMSE"]
        AllXMSE=data["AllXMSE"]
    else:
        for fil in range(Nfiles):
            outfoldername="./OMPZresults%s%d-%d-%d-%d-%d"%(args.label,fil,D,V,Npoint,Nframes)
            data=np.load(outfoldername+'/numpyresults.npz')
            if fil==0:                        
                SNRdB=data["SNRdB"]
                OMPcases=data["OMPcases"]
                Nxcases=len(OMPcases)
                AllHMSE=data["AllHMSE"]
                AllNpEst=data["AllNpEst"]
                Allruntime=data["Allruntime"]
                AlltrueSER=data["AlltrueSER"]
                AllSER=data["AllSER"]
                AlltrueXMSE=data["AlltrueXMSE"]
                AllXMSE=data["AllXMSE"]
            else:       
                AllHMSE=np.concatenate((AllHMSE,data["AllHMSE"]),1)
                AllNpEst=np.concatenate((AllNpEst,data["AllNpEst"]),1)
                Allruntime=np.concatenate((Allruntime,data["Allruntime"]),1)
                AlltrueSER=np.concatenate((AlltrueSER,data["AlltrueSER"]),1)
                AllSER=np.concatenate((AllSER,data["AllSER"]),1)
                AlltrueXMSE=np.concatenate((AlltrueXMSE,data["AlltrueXMSE"]),1)
                AllXMSE=np.concatenate((AllXMSE,data["AllXMSE"]),1)
            
           #the joined out folder for figures 
        outfoldername="./OMPZresults%sJOIN%d-%d-%d-%d-%d"%(args.label,Nfiles,D,V,Npoint,Nframes)
        if not os.path.isdir(outfoldername):
            os.mkdir(outfoldername)

if args.print or args.show:
    plt.figure(1)
    plt.plot(SNRdB,10*np.log10(np.mean(AllHMSE,1)))
    omplabels=['OMP $\\kappa_\\tau=%d, \\kappa_\\nu= %d$'%(x[0],x[0]) if x[1]==0 else 'OMPBR $\kappa_\\tau=2^{%d}, \kappa_\\nu= 2^{%d}$'%(x[1],x[1]) for x in OMPcases]
    plt.legend(omplabels)
    plt.xlabel('SNR(dB)')
    plt.ylabel('NMSE')
    if args.print:
        plt.savefig(outfoldername+'/HMSE.eps')
    plt.figure(2)
    barwidth=0.9*SNRstep/Nxcases
    for xcas in range(Nxcases):
        plt.bar(SNRdB+barwidth*xcas,np.mean(AllNpEst[:,:,xcas],1),width=barwidth)
    plt.legend(omplabels)
    plt.xlabel('SNR(dB)')
    plt.ylabel('No. Iterations ($|\hat{\mathcal{P}}|$)')
    if args.print:
        plt.savefig(outfoldername+'/Npaths.eps')
    plt.figure(3)
    plt.plot(SNRdB,10*np.log10(np.mean(AlltrueXMSE,1)))
    plt.plot(SNRdB,10*np.log10(np.mean(AllXMSE,1)))
    csilabels=omplabels.copy()
    csilabels.insert(0,'Perfect CSI')
    plt.legend(csilabels)
    plt.xlabel('SNR(dB)')
    plt.ylabel('Receiver MSE')
    if args.print:
        plt.savefig(outfoldername+'/XMSE.eps')
    plt.figure(4)
    plt.semilogy(SNRdB,np.mean(AlltrueSER,1))
    plt.semilogy(SNRdB,np.mean(AllSER,1))
    plt.legend(csilabels)
    plt.xlabel('SNR(dB)')
    plt.ylabel('Receiver SER')
    if args.print:
        plt.savefig(outfoldername+'/QPSKSER.eps')
    plt.figure(5)
    SEtrue=np.log2(1+1/(10**(-.1*SNRdB.reshape(Npoint,1))))
    SE=np.log2(1+(1-AllHMSE)/(10**(-.1*SNRdB.reshape(Npoint,1,1))+AllHMSE))
    plt.plot(SNRdB,np.mean(SEtrue,1))
    plt.plot(SNRdB,np.mean(SE,1))
    plt.legend(csilabels)
    plt.xlabel('SNR(dB)')
    plt.ylabel('Achievable Rate')
    if args.print:
        plt.savefig(outfoldername+'/Rate.eps')
    
    plt.figure(6)
    barwidth=0.9/Nxcases
#    barwidth=0.9*SNRstep/Nxcases
    for xcas in range(Nxcases):
        RTprecomp=np.mean(Allruntime[0,:,xcas]/AllNpEst[0,:,xcas],0)
        RTcached=np.mean(np.mean(Allruntime[1:,:,xcas]/AllNpEst[1:,:,xcas],0),0)
        plt.bar(np.arange(2)+(xcas-(Nxcases-1)/2)*barwidth,np.array((RTprecomp,RTcached)),width=barwidth)
#        plt.bar(SNRdB+xcas*barwidth,np.mean(Allruntime[:,:,xcas],1),width=barwidth)
    plt.legend(omplabels)
    plt.ylim([.002,100])
    plt.gca().set_xticks([0,1])
    plt.gca().set_xticklabels(['Compute Dic.','Cached Dic.'])
    plt.gca().set_yscale('log')
    plt.ylabel('Sim. Runtime / OMP Iter. (s)')
    if args.print:
        plt.savefig(outfoldername+'/Runtime.eps')
    
    if args.show:
        plt.show()

yt=np.zeros((D*V+2*(D-1),1),dtype=np.complex64)
