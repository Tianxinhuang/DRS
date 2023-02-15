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
class AHF(object):
    def __init__(self,balance=1.0,lamda1=1,degree=5,maxiter=2):
        self.balance=balance
        self.degree=degree
        self.beta=0
        self.idlist=0
        self.lamda1=lamda1
        self.maxiter=maxiter
        self.name='AHF'
    def set_training_values(self,xt,yt,name=None):
        xt = np.atleast_2d(xt.T).T
        self.xt=xt
        self.yt=np.atleast_2d(yt.T).T
        self.ptnum=np.shape(xt)[0]
        self.dimnum=np.shape(xt)[1]

    def _cal_dev(self,x,model):
        ptnum,dimnum=np.shape(x)
        result=[]
        for i in range(dimnum):
            result.append(model.predict_derivatives(x,i))
        result=np.concatenate(result,axis=-1)
        return result
    ##calculate 
    #x:n*2
    def cal_density(self,x):
        ptnum=np.shape(x)[0]
        dimnum=np.shape(x)[1]
        dismat=np.sum(np.square(np.expand_dims(x,axis=1)-np.expand_dims(x,axis=0)),axis=-1)#n*n
        densi=1/(np.sum(dismat,axis=-1))#n
        alpha=(np.max(densi)-densi)/(np.max(densi)-np.min(densi))#(n,)
        return alpha
    #xs:n*2
    #ys:(n,1)
    #alpha:(n,)
    def cal_limit(self,xs,ys,alpha,cal_base,solve_base):
        basepara=solve_base(xs,ys)
        fbase=cal_base(basepara,xs)#(n,1)
        d=(1+2*alpha)*np.max(np.abs(fbase-ys))#(n,)
        du=ys+np.expand_dims(d,axis=-1)#(n,1)
        dl=ys-np.expand_dims(d,axis=-1)#(n,1)
    ##    print(d)
        upara=solve_base(xs,du)
        lpara=solve_base(xs,dl)
        return basepara,upara,lpara
    def cal_delta(self,xt,upara,lpara,basepara,cal_base):
        yu=cal_base(upara,xt)#nt*1
        yl=cal_base(lpara,xt)
        ybase=cal_base(basepara,xt)
        u=(ybase-yl)/(yu-yl)
        delta1=u/np.sqrt(2*np.log(10))
        delta2=(1-u)/np.sqrt(2*np.log(10))#nt*1
        return u,delta1,delta2,yl,yu
    def err_kernel(self,y,u,delta):
        result=np.exp(-(y-u)**2/(1e-8+2*delta**2))
        return result
    def solve_model(self,xs,ys):
        sm=QP(print_global=False)
##        sm=KRG(poly='constant',print_global=False)
        sm.set_training_values(xs,ys)
        sm.train()
        return sm
    def cal_model(self,sm,x):
        result=sm.predict_values(x)
        return result
        
    def cal_shf(xs,ys,xt,mlist,cal_model,solve_model):
        result=0
        model_num=len(mlist)
        alpha=cal_density(xs)
        basepara,upara,lpara=cal_limit(xs,ys,alpha,cal_model,solve_model)
        u,delta1,delta2,yl,yu=cal_delta(xt,upara,lpara,basepara,cal_model)
    ##    print('u',u)
        err=[]
        results=[]
        for i in range(model_num):
            sm=mlist[i]
            sm.set_training_values(xs,ys)
            sm.train()
            yt=sm.predict_values(xt)#n*1
            ytt=(yt-yl)/(yu-yl)
            merr=err_kernel(ytt,u,delta1)*(ytt<=u)+err_kernel(ytt,u,delta2)*(ytt>u)#n*1
            err.append(merr)
            results.append(yt)
        err=np.concatenate(err,axis=-1)#n*m
        coe=err/(np.sum(err,axis=-1,keepdims=True))#n*m
        results=np.concatenate(results,axis=-1)#n*m
        result=np.sum(coe*results,axis=-1,keepdims=True)#n*1
    ##    print(coe)
        return result
    def train(self):
        model_num=len(model_list)
        alpha=self.cal_density(self.xt)
        self.basepara,self.upara,self.lpara=self.cal_limit(self.xt,self.yt,alpha,self.cal_model,self.solve_model)
        modelparas=[]
        for i in range(model_num):
            sm=copy.deepcopy(model_list[i])
            sm.set_training_values(self.xt,self.yt)
            sm.train()
            modelparas.append(sm)
        self.modelparas=modelparas
        
    def predict_values(self,x):
        u,delta1,delta2,yl,yu=self.cal_delta(x,self.upara,self.lpara,self.basepara,self.cal_model)
        results=[]
        err=[]
        model_num=len(model_list)
        for i in range(model_num):
            sm=self.modelparas[i]
            yt=sm.predict_values(x)#n*1
            ytt=(yt-yl)/(yu-yl)
            merr=self.err_kernel(ytt,u,delta1)*(ytt<=u)+self.err_kernel(ytt,u,delta2)*(ytt>u)#n*1
            err.append(merr)
            results.append(yt)
        err=np.concatenate(err,axis=-1)#n*m
        coe=err/(np.sum(err,axis=-1,keepdims=True))#n*m
        results=np.concatenate(results,axis=-1)#n*m
        result=np.sum(coe*results,axis=-1,keepdims=True)#n*1
        return result
if __name__=='__main__':
    sm=AHF()
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
