import numpy as np
import itertools
import matplotlib.pyplot as plt
import tf_util as tl
from mpl_toolkits.mplot3d import Axes3D
from smt.surrogate_models import KRG
from smt.surrogate_models import QP
from smt.surrogate_models import RBF
from rbf import RBF_MQ
from svr import svr
import copy
import random

##model_list=[RBF_MQ(delta=5),RBF(poly_degree=0,d0=0.34,print_global=False),\
##            RBF(poly_degree=0,d0=0.34,print_global=False),RBF(poly_degree=0,d0=0.34,print_global=False)]
model_list=[QP(print_global=False),KRG(poly='constant',print_global=False),\
            RBF_MQ(delta=5),RBF(poly_degree=0,d0=0.34,print_global=False),svr()]
##model_list=[QP(print_global=False),RBF_MQ(delta=5),\
##             RBF_MQ(delta=5),RBF(poly_degree=0,d0=0.34,print_global=False)]
class UES(object):
    def __init__(self,balance=1.0,lamda1=1,degree=5,maxiter=2):
        self.balance=balance
        self.degree=degree
        self.beta=0
        self.idlist=0
        self.lamda1=lamda1
        self.maxiter=maxiter
        self.name='UES'
    def set_training_values(self,xt,yt,name=None):
        xt = np.atleast_2d(xt.T).T
        self.xt=xt
        self.yt=np.atleast_2d(yt.T).T
        self.ptnum=np.shape(xt)[0]
        self.dimnum=np.shape(xt)[1]
    #x:n*d
    #y:n*1
    #err:n*1
    def get_cv(self,sm):
        ptnum=np.shape(self.xt)[0]
        x,y=self.xt,self.yt
        err=[]
        for j in range(ptnum):
            ids=list(range(ptnum))
            des=ids[j]
            del[ids[j]]
##            sm.set_training_values(x[ids], y[ids])
##            sm.train()
            xdd=x[des]
            if len(np.shape(xdd))==1:
                xdd=np.expand_dims(xdd,axis=0)
            ydes = sm.predict_values(xdd)
            #print(np.shape(ydes),np.shape(y[des]))
            err.append(np.abs(np.squeeze(ydes)-y[des]))
        err=np.array(err)
        #err=np.expand_dims(np.array(err),axis=-1)
        #print(np.shape(err))
        return err
    #x:m*d
    #errs:ns*n
    def get_local(self,x,errs):
        modelnum,ptnum=np.shape(errs)
        wik=np.zeros_like(errs)
        idx=np.concatenate([np.expand_dims(np.argmin(errs,axis=0),axis=-1),np.expand_dims(np.arange(ptnum),axis=-1)],axis=-1)
        idx=np.transpose(idx)
        idx=tuple(idx.tolist())
        wik[idx]=1.0
        #print(wik)
        dismat=np.expand_dims(x,axis=1)-np.expand_dims(self.xt,axis=0)#m*n*d
        dis=np.sum(np.square(dismat),axis=-1)#m*n
        discut=(dis<1e-8).astype(np.float)#m*n
        cut=np.sum(discut,axis=-1)
        for i in range(len(cut)):
            if cut[i]<1:
                discut[i]=1/dis[i]
        wii=np.sum(np.expand_dims(wik,axis=0)*np.expand_dims(discut,axis=1),axis=-1)#m*ns
        wi=wii/np.sum(wii,axis=-1,keepdims=True)#m*ns
        #print(np.shape(wi))
        return wi
    def get_global(self,errs):
        ei=np.sqrt(np.mean(np.square(errs),axis=-1))
        ee=np.mean(ei)
        wii=1/(ei+0.05*ee)
        wi=wii/np.sum(wii,axis=-1,keepdims=True)#ns
        return wi
    #errs:modelnum*n,wg:modelnum,wl:ptnum*modelnum
    def get_lambda(self,x):
        dismat=np.expand_dims(x,axis=1)-np.expand_dims(self.xt,axis=0)#m*n*d
        dis=np.sum(np.square(dismat),axis=-1)#m*n
        kdis=np.sort(dis,axis=1)[:,:2]#m*2
        lam=np.sin(np.pi/2*(kdis[:,0]/kdis[:,1])) #m*1
        lam=np.expand_dims(lam,axis=-1)
        return lam
        
    def train(self):
        model_num=len(model_list)
        modelparas=[]
        errs=[]
        for i in range(model_num):
            sm=copy.deepcopy(model_list[i])
            sm.set_training_values(self.xt,self.yt)
            sm.train()
            errs.append(self.get_cv(sm))
            modelparas.append(sm)
        errs=np.concatenate(errs,axis=-1)
        self.errs=np.transpose(errs)
        #print(np.shape(errs))
        self.modelparas=modelparas
        
    def predict_values(self,x):
        x = np.atleast_2d(x.T).T
        model_num=len(model_list)
        wl=self.get_local(x,self.errs)#m*ns
        wg=np.expand_dims(self.get_global(self.errs),axis=0)#1*ns
        lam=self.get_lambda(x)#m*1

        print(np.shape(lam),np.shape(wl))
        
        w=wg*lam+wl*(1-lam)#m*ns
        w=w/np.sum(w,axis=-1,keepdims=True)
        
        result=[]
        for i in range(model_num):
            sm=self.modelparas[i]
            yt=sm.predict_values(x)#m*1
            result.append(yt)
        result=np.concatenate(result,axis=-1)#m*ns
        #print(np.shape(result))
        result=np.sum(w*result,axis=-1,keepdims=True)
        
        return result
if __name__=='__main__':
    sm=UES()
    xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])
    sm.set_training_values(xt, yt)
    sm.train()

    num = 100
    x = np.linspace(0.0, 4.0, num)
    y = sm.predict_values(x)

    plt.plot(xt, yt, "o")
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["Training data", "Prediction"])
    plt.show()
