import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from smt.surrogate_models import KRG
from smt.surrogate_models import QP
from smt.surrogate_models import RBF
from rbf import RBF_MQ
from residual_hybrid import resi_hybrid
from svr import svr
from ahf import AHF
from es_hgl import ES_HGL
from ues import UES
from ann import ANN

import tf_util as tl
import random
import copy
import os
import time
#choose this when compare DRS performances
model_list=[resi_hybrid(),AHF(),ES_HGL(),UES(),QP(print_global=False),KRG(poly='constant',print_global=False),\
            RBF_MQ(delta=5),RBF(poly_degree=0,d0=0.34,print_global=False),ANN()]

#choose this when compare validation
##model_list=[QP(print_global=False),KRG(poly='constant',print_global=False),\
##            RBF_MQ(delta=5),RBF(poly_degree=0,d0=0.34,print_global=False),svr()]

##calculate 
#x:n*2
def cal_density(x):
    ptnum=np.shape(x)[0]
    dimnum=np.shape(x)[1]
    dismat=np.sum(np.square(np.expand_dims(x,axis=1)-np.expand_dims(x,axis=0)),axis=-1)#n*n
    densi=1/(np.sum(dismat,axis=-1))#n
    alpha=(np.max(densi)-densi)/(np.max(densi)-np.min(densi))#(n,)
    return alpha
#xs:n*2
#ys:(n,1)
#alpha:(n,)
def cal_limit(xs,ys,alpha,cal_base,solve_base):
    basepara=solve_base(xs,ys)
    fbase=cal_base(basepara,xs)#(n,1)
    d=(1+2*alpha)*np.max(np.abs(fbase-ys))#(n,)
    du=ys+np.expand_dims(d,axis=-1)#(n,1)
    dl=ys-np.expand_dims(d,axis=-1)#(n,1)
##    print(d)
    upara=solve_base(xs,du)
    lpara=solve_base(xs,dl)
    return basepara,upara,lpara
def cal_delta(xt,upara,lpara,basepara,cal_base):
    yu=cal_base(upara,xt)#nt*1
    yl=cal_base(lpara,xt)
    ybase=cal_base(basepara,xt)
    u=(ybase-yl)/(yu-yl)
    delta1=u/np.sqrt(2*np.log(10))
    delta2=(1-u)/np.sqrt(2*np.log(10))#nt*1
    return u,delta1,delta2,yl,yu
def err_kernel(y,u,delta):
    result=np.exp(-(y-u)**2/(2*delta**2))
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
def solve_model(xs,ys):
    sm=QP(print_global=False)
##    sm=KRG(poly='constant',print_global=False)
    sm.set_training_values(xs,ys)
    sm.train()
    return sm
def cal_model(sm,x):
    result=sm.predict_values(x)
    return result
def show2dcurve(func,downval,upval,datax,datay):
    x=np.arange(downval[0],upval[0],step=(upval[0]-downval[0])/10)
    y=np.arange(downval[1],upval[1],step=(upval[1]-downval[1])/10)
    coornum=np.shape(x)[0]
    
    xx,yy=np.meshgrid(x,y)
    coors=np.concatenate([np.expand_dims(xx,axis=-1),np.expand_dims(yy,axis=-1)],axis=-1)
    coors=np.reshape(coors,[-1,2])
    result=func(coors)
    result=np.reshape(result,[coornum,coornum])
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    p1=ax.plot_surface(xx,yy,result,color='r',label='raw surface')
    p2=ax.scatter(datax[:,0],datax[:,1],datay,c='black',marker='o',label='traing data')
##    ax.legend()
    plt.show()
def show1dcurve(func,downval,upval,datax,datay):
    num = 100
    x = np.linspace(downval,upval, num)
    y = func(x)

##    plt.plot(datax, datay, "o")
    plt.plot(x, y,c='b')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(datax,datay,c='black',marker='o')
    plt.legend(['raw curve','training data'])
    plt.show()

def R_square(y,y1):
    y=np.reshape(y,[-1,1])
    y1=np.reshape(y1,[-1,1])
##    yvar=np.mean(np.square(y1-np.mean(y1)))
    yy=np.mean(np.square(y-y1))
##    print(np.shape(y),np.shape(y1))
##    result=1-yy/yvar
    return np.sqrt(yy)
def NRMSE(y,y1):
    y=np.reshape(y,[-1,1])
    y1=np.reshape(y1,[-1,1])
    yu=np.sum(np.square(y-y1))
    yd=np.sum(np.square(y))
    result=np.sqrt(yu/yd)
    return result
def cal_models_R(down,up,num1,num2,func,ws,funcid,test_time=1):
    model_num=len(model_list)
##    errs=[]
##    errslist=[]
    predsarray=np.zeros((test_time,model_num))
    npredsarray=np.zeros((test_time,model_num))
    timesarray=np.zeros((test_time,model_num))
    for j in range(test_time):
        xt,yt,x,y=get_data(down,up,num1,num2,func)
        errs=[]
        yt=np.reshape(yt,[-1,1])
##        result=cal_shf(xt,yt,x,model_list,cal_model,solve_model)
##        errs.append(R_square(y,result))
        for i in range(model_num):
            sm=copy.deepcopy(model_list[i])
            if i==0:
                sm.set_training_values(func,down,up,xt,yt)
            else:
                sm.set_training_values(xt,yt)
            sm.train()
##            show2dcurve(func,down,up,xt,yt)
            time_start=time.clock()
            ydes=sm.predict_values(x)
            time_end=time.clock()
            timesarray[j,i]=time_end-time_start
            predsarray[j,i]=R_square(y,ydes)
            npredsarray[j,i]=NRMSE(y,ydes)
    for i in range(model_num):
        print(model_list[i].name,np.mean(predsarray[:,i],axis=0),np.var(predsarray[:,i],axis=0))
        ws.write(funcid,1+4*i,np.mean(npredsarray[:,i],axis=0))
        ws.write(funcid,2+4*i,np.mean(predsarray[:,i],axis=0))
        ws.write(funcid,3+4*i,np.var(predsarray[:,i],axis=0))
        ws.write(funcid,4+4*i,np.mean(timesarray[:,i],axis=0))
        
    ws.write(funcid,1+4*model_num,model_list[np.argmin(np.min(predsarray[:,:],axis=0))].name)
    ws.write(funcid,2+4*model_num,model_list[np.argmin(np.mean(predsarray[:,:],axis=0))].name)
    ws.write(funcid,3+4*model_num,model_list[np.argmin(np.var(predsarray[:,:],axis=0))].name)
    ws.write(funcid,4+4*model_num,np.min(np.min(predsarray[:,:],axis=0)))
    ws.write(funcid,5+4*model_num,np.min(np.mean(predsarray[:,:],axis=0)))
    ws.write(funcid,6+4*model_num,np.min(np.var(predsarray[:,:],axis=0)))
    print(' ')

def eva_err(dev2list,devlist,x,y,model,knum=8,kf=6,testnum=20,use_type='c'):
    ptnum=np.shape(x)[0]
    errlist=[]
    nerrlist=[]
    #crosscheck
    if use_type=='c':
        fnum=int((ptnum+1)/kf)
        for i in range(testnum):
            idlist=list(range(ptnum))
            random.shuffle(idlist)
            for j in range(kf):
                target_id=idlist[j*fnum:j*fnum+fnum]
                train_id=idlist[:j*fnum]
                if len(train_id)==0:
                    train_id=idlist[j*fnum+fnum:]
                else:
                    train_id.extend(idlist[j*fnum+fnum:])
                traindata=x[train_id]
                trainlabel=y[train_id]
                testdata=x[target_id]
                testlabel=y[target_id]
                model.set_training_values(traindata,trainlabel)
                model.train()
                y_pred=model.predict_values(testdata)
                
                nerr=np.sum(np.square(y_pred-testlabel))/np.sum(np.square(testlabel))
                err=np.mean(np.square(y_pred-testlabel))
                errlist.append(err)
                nerrlist.append(nerr)
                
        errs=np.mean(errlist)
        nerrs=np.mean(nerrlist)
    #leave one
    elif use_type=='l':
        fnum=1
        kf=ptnum
        for i in range(testnum):
            idlist=list(range(ptnum))
            random.shuffle(idlist)
            for j in range(kf):
                target_id=idlist[j*fnum:j*fnum+fnum]
                train_id=idlist[:j*fnum]
                if len(train_id)==0:
                    train_id=idlist[j*fnum+fnum:]
                else:
                    train_id.extend(idlist[j*fnum+fnum:])
                traindata=x[train_id]
                trainlabel=y[train_id]
                testdata=x[target_id]
                testlabel=y[target_id]
                model.set_training_values(traindata,trainlabel)
                model.train()
                y_pred=model.predict_values(testdata)
                nerr=np.sum(np.square(y_pred-testlabel))/np.sum(np.square(testlabel))
                err=np.mean(np.square(y_pred-testlabel))
                errlist.append(err)
                nerrlist.append(nerr)
        nerrs=np.mean(nerrlist)
        errs=np.mean(errlist)
    #random
    elif use_type=='r':
        fnum=int((ptnum+1)/kf)
        idlist=list(range(ptnum))
        random.shuffle(idlist)
        target_id=idlist[:fnum]
        train_id=idlist[fnum:]
        traindata=x[train_id]
        trainlabel=y[train_id]
        testdata=x[target_id]
        testlabel=y[target_id]
        model.set_training_values(traindata,trainlabel)
        model.train()
        y_pred=model.predict_values(testdata)
        nerrs=np.sum(np.square(y_pred-testlabel))/np.sum(np.square(testlabel))
        errs=np.mean(np.square(y_pred-testlabel))
    #dev
    elif use_type=='d':
        fnum=ptnum-int((ptnum+1)/kf)
        idlist=list(range(ptnum))
        train_id=np.argpartition(-np.max(np.mean(np.abs(devlist),axis=-1),axis=-1),fnum)[:fnum]
        traindata=x[train_id]
        trainlabel=y[train_id]
        testdata=x
        testlabel=y
        model.set_training_values(traindata,trainlabel)
        model.train()
        y_pred=model.predict_values(testdata)
##        errs=np.sum(np.square(y_pred-testlabel))
        nerrs=np.sum(np.square(y_pred-testlabel))/np.sum(np.square(testlabel))
        errs=np.mean(np.square(y_pred-testlabel))
    elif use_type=='2d':
        fnum=ptnum-int((ptnum+1)/kf)
        idlist=list(range(ptnum))
        train_id=np.argpartition(-np.max(np.mean(np.abs(dev2list),axis=-1),axis=-1),fnum)[:fnum]
        traindata=x[train_id]
        trainlabel=y[train_id]
        testdata=x
        testlabel=y
        model.set_training_values(traindata,trainlabel)
        model.train()
        y_pred=model.predict_values(testdata)
##        errs=np.sum(np.square(y_pred-testlabel))
        nerrs=np.sum(np.square(y_pred-testlabel))/np.sum(np.square(testlabel))
        errs=np.mean(np.square(y_pred-testlabel))
    return errs,nerrs
#data:ptnum*dimnum,devlist:ptnum*dimnum*dimnum
def choose_pts(data,y0,dev2list,devlist,choice_num=18,use_type='2d'):
    if use_type=='2d':
        idx=np.argpartition(-np.max(np.mean(np.abs(dev2list),axis=-1),axis=-1),choice_num-1)[:choice_num]
    elif use_type=='1d':
##        print(np.shape(dev2list))
        idx=np.argpartition(-np.max(np.mean(np.abs(devlist),axis=-1),axis=-1),choice_num-1)[:choice_num]
    else:
        idx=list(range(np.shape(data)[0]))
        random.shuffle(idx)
        idx=idx[:choice_num]
    return data[idx],y0[idx]
def cal_dev(x,func):
    ptnum,dimnum=np.shape(x)
    tiny_dis=1e-8
    movemat=np.eye(dimnum)*tiny_dis#dimnum*dimnum
    alldata=np.expand_dims(movemat,axis=0)+np.expand_dims(x,axis=1)#n*dimnum*dimnum
    alldata=np.reshape(alldata,[-1,dimnum])
    yy=func(alldata)
    yy=np.reshape(yy,[ptnum,dimnum])
    y=np.reshape(func(x),[-1,1])#n*1
    result=(yy-y)/tiny_dis#n*dimnum
    return result
def cal_dev2(x,func):
    dev2,dev=tl.cal_dev2(x,func)
##    print(np.shape(dev2),np.shape(dev))
    return dev2,dev
#remove data based on func
def remain_data(func,xt,yt,fp,ft):
    ptnum=np.shape(xt)[0]
    plist=fp
    tlist=ft
    datalist=[]
    labellist=[]
    dev2,dev=cal_dev2(xt,lambda x:cal_dev(x,func))
##    print(np.shape(dev),np.shape(dev2))
    
    for i in range(len(plist)):
        chnum=int(plist[i]*ptnum)
        data=[]
        label=[]
        for j in range(len(tlist)):
            ut=tlist[j]
            newdata,newlabel=choose_pts(xt,yt,dev2,dev,chnum,ut)
            data.append(newdata)
            label.append(newlabel)
        datalist.append(data)
        labellist.append(label)
    return datalist,labellist

def compare_sampling(down,up,num1,num2,func,ws,funcid,test_time=1):
    model_num=len(model_list)
##    tlist=['c','l','r','d','2d']
    tlist=['c','r','d']
##    fp=[0.25,0.5,0.75,6/7]
    fp=[6/7]
##    ft=['1d','r','2d']
    
    tnum=len(tlist)
    errs=[]
    errslist=[]
    ws,ws1,ws2,ws3=ws
    ws1,ws11=ws1
##    errsarray=np.zeros((test_time,len(fp),len(ft),model_num))#len(ptlist),len(tlist)
##    predsarray2=np.zeros((test_time,len(fp),len(ft),model_num))
    predsarray=np.zeros((test_time,tnum,model_num))
    npredsarray=np.zeros((test_time,tnum,model_num))
    timesarray=np.zeros((test_time,tnum,model_num))
    
    valids=np.zeros((test_time,tnum+1))
    nvalids=np.zeros((test_time,tnum+1))
    
    valerrs=np.zeros((test_time,tnum+1,model_num))
    nvalerrs=np.zeros((test_time,tnum+1,model_num))
    
    for j in range(test_time):
        xt,yt,x,y=get_data(down,up,num1,num2,func)
        yt=np.reshape(yt,[-1,1])
        
        for i in range(model_num):
            sm=copy.deepcopy(model_list[i])
            sm.set_training_values(xt,yt)
            sm.train()
            dev2,dev=cal_dev2(xt, lambda x:cal_dev(x,sm.predict_values))
            valerrs[j,tnum,i]=R_square(y,sm.predict_values(x))
            nvalerrs[j,tnum,i]=NRMSE(y,sm.predict_values(x))
            for t in range(tnum):
                time_start=time.clock()
                valerrs[j,t,i],nvalerrs[j,t,i]=np.sqrt(eva_err(dev2,dev,xt,yt,sm,knum=8,kf=3,testnum=1,use_type=tlist[t]))
                time_end=time.clock()
                timesarray[j,t,i]=time_end-time_start
        for t in range(tnum):
            valids[j,t]=int(np.argmin(valerrs[j,t,:],axis=-1))
            nvalids[j,t]=int(np.argmin(nvalerrs[j,t,:],axis=-1))
        
        valids[j,tnum]=int(np.argmin(valerrs[j,tnum,:],axis=-1))
        nvalids[j,tnum]=int(np.argmin(nvalerrs[j,tnum,:],axis=-1))
        
    predsarray=np.abs(valerrs[:,:tnum,:]-valerrs[:,tnum:,:])
    npredsarray=np.abs(nvalerrs[:,:tnum,:]-nvalerrs[:,tnum:,:])#30*3*5
    print(np.shape(predsarray))
    for i in range(test_time):
        for t in range(tnum):
            if funcid==1:
                ws3.write(0,2*t+2*i*tnum,'test_'+str(i)+'_'+tlist[t]+'_RMSE')
                ws3.write(0,1+2*t+2*i*tnum,'test_'+str(i)+'_'+tlist[t]+'_NRMSE')
            ws3.write(funcid,2*t+2*i*tnum,np.mean(predsarray,axis=-1)[i,t])
            ws3.write(funcid,1+2*t+2*i*tnum,np.mean(npredsarray,axis=-1)[i,t])
                        
    for i in range(test_time):
        for t in range(tnum):
            if funcid==1:
                ws2.write(0,3*t+3*i*tnum,'test_'+str(i)+'_'+tlist[t]+'_RMSE')
                ws2.write(0,1+3*t+3*i*tnum,'test_'+str(i)+'_'+tlist[t]+'_time')
                ws2.write(0,2+3*t+3*i*tnum,'test_'+str(i)+'_'+tlist[t]+'_NRMSE')
            ws2.write(funcid,3*t+3*i*tnum,valerrs[i,tnum,int(valids[i,t])])
            ws2.write(funcid,1+3*t+3*i*tnum,np.mean(timesarray[i,t]))
            ws2.write(funcid,2+3*t+3*i*tnum,nvalerrs[i,tnum,int(nvalids[i,t])])
        
  
def get_data(down,up,num1,num2,func):
    xt=tl.latian1_sampling(down,up,num1)
    x=tl.latian1_sampling(down,up,num2)
    yt=func(xt)
    y=func(x)
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
    num2=100
    func=tl.GN_func
##    down=[0]
##    up=[1]
##    num1=5
##    num2=100
##    func=tl.test_func1
##    
##    xt,yt,x,y=get_data(down,up,num1,num2,func)
    
##    ids,okids=choose_models(xt,yt,0.8)
##    
##    p,result=cal_coefficient(x,xt,yt,ids,okids)
    
    cal_models_R(down,up,num1,num2,func)
##    print(R_square(y,result))
    
