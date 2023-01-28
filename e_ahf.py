import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from smt.surrogate_models import KRG
from smt.surrogate_models import QP
from smt.surrogate_models import RBF
from rbf import RBF_MQ
from poly_rbf import Poly_RBF

import tf_util as tl

model_list=[QP(print_global=False),KRG(poly='constant',print_global=False),\
            RBF_MQ(delta=0.34),RBF(poly_degree=0,d0=0.34,print_global=False)]
model_list1=[QP(print_global=False),KRG(poly='constant',print_global=False),\
             RBF_MQ(delta=0.001),RBF(poly_degree=0,d0=0.001,print_global=False),Poly_RBF(delta=0.001,degree=2,\
            use_judgement=True,judge_type='a',judge_posi=True,topk=0.8,threshold=0.16,limit=6)]
##model_list=[QP,KRG,RBF_MQ,RBF]

##def fresh_models

def choose_models(x,y,beta=0.5):
    model_num=len(model_list)
    x=np.atleast_2d(x.T).T
    ptnum=np.shape(x)[0]
    dimnum=np.shape(x)[1]
    cv_list=[]
    for i in range(model_num):
        err=[]
        sm=model_list[i]
##        sm=model()
        for j in range(ptnum):
            ids=list(range(ptnum))
            des=ids[j]
            del[ids[j]]
##            print(np.shape(y),ids)
            sm.set_training_values(x[ids], y[ids])
            sm.train()
            xdd=x[des]
            if len(np.shape(xdd))==1:
                xdd=np.expand_dims(xdd,axis=0)
##            print(np.shape(xdd))
            ydes = sm.predict_values(xdd)
            err.append(np.square(ydes-y[des]))
        cv_list.append(np.sum(err))
    cv_list=np.array(cv_list)
##    print(cv_list)
    cv_list=(cv_list-min(cv_list))/(max(cv_list)-min(cv_list))
##    cv_list=np.array([1,0,0.79,0.87])
##    print(cv_list)
##    minid=np.argmin(cv_list)
    ids=np.array(range(model_num))
    okids=ids[cv_list<beta]
    minid=np.argmin(cv_list[okids])
    return minid,okids
def cal_gaussian(ys,yis,theta):
    n,m=np.shape(ys)
    s=np.shape(yis)[0]
    theta1=np.corrcoef(ys)#n*n
    theta2=np.corrcoef(ys,yis)[:n,n:]#n*s
    result=[]
    for i in range(s):
        p1=np.dot(theta2[:,i].T,theta1)
        p1=1-np.dot(p1,theta2[:,i])
        p2=np.dot(np.ones((1,n)),sc.linalg.pinv(theta1))
        p2=1-np.dot(p2,theta2[:,i])
        p2=p2/(np.dot(np.dot(np.ones((1,n)),sc.linalg.pinv(theta1)),np.ones((n,1))))
        p2=np.squeeze(p2)
        result.append(theta*(p1+p2))
    result=np.array(result)
    return result
    
    
def cal_coefficient(data,x,y,minid,okids,theta=1):
    model_num=len(okids)
    ptnum=np.shape(data)[0]
    ys=[]
    yis=[]
    for i in range(model_num):
        sm=model_list[okids[i]]
##        sm=model()
        sm.set_training_values(x,y)
        sm.train()
        ys.append(sm.predict_values(x))
        yis.append(sm.predict_values(data))
##    print(np.shape(ys[2]))
    ys=np.hstack(ys)#n*model_num
    yis=np.hstack(yis)#ptnum*model_num
    ybase=np.expand_dims(yis[:,minid],axis=-1)#ptnum*1
    s=cal_gaussian(ys,yis,theta)#ptnum
##    print(np.square(yis-ybase).max(),s.min())
    p=np.exp(-np.square(yis-ybase)/(1e-5+2*np.expand_dims(s,axis=-1)))#ptnum*model_num
##    print(p)
##    minp=np.min(p,axis=1,keepdims=True)
##    maxp=np.max(p,axis=1,keepdims=True)
##    p=(p-minp)/(maxp-minp)
    p=p/(1e-5+np.sum(p,axis=-1,keepdims=True))
    result=np.sum(yis*p,axis=1,keepdims=True)
    return p,result
def R_square(y,y1):
##    print(np.shape(y),np.shape(y1))
##    assert True
    yvar=np.mean(np.square(y1-np.mean(y1)))
    yy=np.mean(np.square(np.expand_dims(y,axis=-1)-y1))
##    print(np.shape(y),np.shape(y1))
##    result=1-yy/yvar
    return np.sqrt(yvar),np.sqrt(yy)
def cal_models_R(down,up,num1,num2,func,test_time=10):
    model_num=len(model_list1)
    errs=[]
    errslist=[]
    for j in range(test_time):
##        if test_time>1:
        xt,yt,x,y=get_data(down,up,num1,num2,func)
        errs=[]
##        ids,okids=choose_models(xt,yt,1.0) 
##        _,result=cal_coefficient(x,xt,yt,ids,okids)
##        errs.append(R_square(y,result)[-1])
        for i in range(model_num):
            sm=model_list1[i]
    ##        sm=model()
##            print(np.shape(yt))
            sm.set_training_values(xt,np.expand_dims(yt,axis=-1))
            sm.train()
            err=R_square(y,sm.predict_values(x))[-1]
            errs.append(err)
        errslist.append(errs)
    ee=np.mean(np.array(errslist),axis=0)
    vv=np.var(np.array(errslist),axis=0)
##    print('E-AHF ',ee[0],vv[0])
    for i in range(model_num):
        print(model_list1[i].name,ee[i],vv[i])
    print(' ')
def get_data(down,up,num1,num2,func):
    xt=tl.latian1_sampling(down,up,num1)
##    xt=np.array([0.039,0.260,0.590,0.750,0.990])
    x=tl.latian1_sampling(down,up,num2)
##    x=np.atleast_2d(x.T).T
##    xt=np.atleast_2d(xt.T).T
##    print(np.shape(xt),np.shape(x))
    yt=func(xt)
    y=func(x)
##    print(np.shape(yt),np.shape(x))
    return xt,yt,x,y
    

if __name__=='__main__':
##    xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
##    x=np.array([1.5,2.5])
##    yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])
##    xt=tl.latian1_sampling([0],[1],12).T
##    xt=np.array([0.039,0.260,0.590,0.750,0.990])
##    print(np.shape(xt))
##    yt=tl.test_func1(xt)
##    print(np.shape(yt))
##    x=tl.latian1_sampling([0],[1],500).T
##    y=tl.test_func1(x)
    down=[-100,-100]
    up=[100,100]
    num1=36
    num2=50
    func=tl.GN_func
##    down=[0]
##    up=[1]
##    num1=5
##    num2=50
##    func=tl.test_func1
    
##    xt,yt,x,y=get_data(down,up,num1,num2,func)
    
##    ids,okids=choose_models(xt,yt,0.8)
##    
##    p,result=cal_coefficient(x,xt,yt,ids,okids)
    
    cal_models_R(down,up,num1,num2,func)
##    print(R_square(y,result))
    
