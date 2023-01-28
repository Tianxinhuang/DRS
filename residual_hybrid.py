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
class resi_hybrid(object):
    def __init__(self,balance=1.0,lamda1=1,degree=5,maxiter=2):
        self.balance=balance
        self.degree=degree
        self.beta=0
        self.idlist=0
        self.lamda1=lamda1
        self.maxiter=maxiter
        self.name='residual_hybrid'
    def set_training_values(self,func,down,up,xt,yt,name=None):
        self.func=func
        xt = np.atleast_2d(xt.T).T
        self.xt=xt
        self.yt=np.atleast_2d(yt.T).T
        self.down=down
        self.up=up

        self.ptnum=np.shape(xt)[0]
        self.dimnum=np.shape(xt)[1]

    def _cal_dev(self,x,model):
        ptnum,dimnum=np.shape(x)
        tiny_dis=1e-8
        movemat=np.eye(dimnum)*tiny_dis#dimnum*dimnum
        alldata=np.expand_dims(movemat,axis=0)+np.expand_dims(x,axis=1)#n*dimnum*dimnum
        alldata=np.reshape(alldata,[-1,dimnum])
        yy=model.predict_values(alldata)
        yy=np.reshape(yy,[ptnum,dimnum])
        y=np.reshape(model.predict_values(x),[-1,1])#n*1
        result=(yy-y)/tiny_dis#n*dimnum
        
##        ptnum,dimnum=np.shape(x)
##        result=[]
##        for i in range(dimnum):
##            result.append(model.predict_derivatives(x,i))
##        result=np.concatenate(result,axis=-1)
        
        return result
    def _eva_err(self,x,y,model,knum=8,kf=6,testnum=20):
        ptnum=np.shape(x)[0]
        errlist=[]
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
                err=np.sum(np.square(y_pred-testlabel))
                errlist.append(err)
        errs=np.mean(err)
##        for i in range(testnum):
##            idlist=list(range(ptnum))
##            random.shuffle(idlist)
##            kk=int(knum)
##            if kk<3:
##                kk=3
##            idlist=idlist[:kk]
##            dt=x[idlist]
##            ydt=y[idlist]
##            model.set_training_values(dt,ydt)
##            model.train()
##            y_pred=model.predict_values(x)
####            print('y_pred',y_pred)
####            print('y',y)
##            err=np.sum(np.square(y_pred-y))
##            errlist.append(err)
##        errs=np.mean(errlist)
        
        return errs
    #data:ptnum*dimnum,devlist:ptnum*dimnum*dimnum
    def _choose_pts(self,data,y0,devlist,choice_num=18):
        idx=np.argpartition(-np.max(np.mean(np.abs(devlist),axis=-1),axis=-1),choice_num)[:choice_num]
        uidx=np.argpartition(-np.max(np.mean(np.abs(devlist),axis=-1),axis=-1),choice_num)[choice_num:]
##        idx=list(range(np.shape(data)[0]))
##        random.shuffle(idx)
##        idx=idx[:choice_num]
        result=data[idx]
        return result,y0[idx],data[uidx],y0[uidx]
    def show2dcurve(self,func1,func2,func3,downval,upval,datax,datay,cx,cy):
        x=np.arange(downval[0],upval[0],step=(upval[0]-downval[0])/20)
        y=np.arange(downval[1],upval[1],step=(upval[1]-downval[1])/20)
        coornum=np.shape(x)[0]
        
        xx,yy=np.meshgrid(x,y)
        coors=np.concatenate([np.expand_dims(xx,axis=-1),np.expand_dims(yy,axis=-1)],axis=-1)
        coors=np.reshape(coors,[-1,2])
        
        result1=func1(coors)
        result1=np.reshape(result1,[coornum,coornum])

        result2=func2(coors)
        result2=np.reshape(result2,[coornum,coornum])

        result3=func3(coors)
        result3=np.reshape(result3,[coornum,coornum])
        
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        alpha=0.9
        rs=20.0
        ax.plot_surface(xx,yy,result1,color='lightblue',label='predict surface',alpha=alpha)
        ax.plot_surface(xx,yy,result2,color='lightgreen',label='points removed surface',alpha=alpha)
        ax.plot_surface(xx,yy,result3,color='lightyellow',label='raw surface',alpha=alpha)
        p2=ax.scatter(datax[:,0],datax[:,1],datay,c='red',s=rs,marker='X',label='training data',alpha=alpha)
        p3=ax.scatter(cx[:,0],cx[:,1],cy,c='black',s=rs,marker='o',label='chosen data',alpha=alpha)
##        plt.legend()
##        ax.legend()
        plt.show()
    def show1dcurve(self,func1,func2,func3,downval,upval,datax,datay,cx,cy):
        num = 100
        x = np.linspace(downval,upval, num)
        y1 = func1(x)
        ax=plt.axes()
        ax.plot(x, y1,c='b')

        y2 = func2(x)
        ax.plot(x, y2,c='g')

        y3 = func3(x)
        ax.plot(x, y3,c='y')

        
        plt.xlabel("x")
        plt.ylabel("y")
        ax.scatter(datax,datay,c='red',marker='o')
        ax.scatter(cx,cy,c='black',marker='o')
        plt.legend(['points removed curve','trained curve','raw curve'],fontsize=20)
        plt.show()
    def showcurve(self,plt,ax,func,downval,upval,dimnum=1,bal=1):
        if dimnum==1:
            num = 100
            x = np.linspace(downval,upval, num)
            y=func(x)/bal
##            ax=plt.axes()
            ax.plot(x, y,c='b')
            plt.xlabel("x")
            plt.ylabel("y")
##            plt.legend(['points removed curve','trained curve','raw curve'],fontsize=20)
##            plt.show()
        elif dimnum==2:
            x=np.arange(downval[0],upval[0],step=(upval[0]-downval[0])/20)
            y=np.arange(downval[1],upval[1],step=(upval[1]-downval[1])/20)
            coornum=np.shape(x)[0]
            
            xx,yy=np.meshgrid(x,y)
            coors=np.concatenate([np.expand_dims(xx,axis=-1),np.expand_dims(yy,axis=-1)],axis=-1)
            coors=np.reshape(coors,[-1,2])
            
            result=func(coors)/bal
            result=np.reshape(result,[coornum,coornum])
##            fig = plt.figure()
##            ax = plt.axes(projection='3d')
            alpha=0.9
            rs=20.0
##            ax.set_xticks([])
##            ax.set_yticks([])
##            ax.set_zticks([])
##            frame = plt.gca()
##            frame.axes.get_yaxis().set_visible(False)
##            frame.axes.get_xaxis().set_visible(False)
##            frame.axes.get_zaxis().set_visible(False)
            ax.plot_surface(xx,yy,result,cmap='rainbow',label='predict surface',alpha=alpha)
        else:
            print('')
    def draw_curve(self,func,smlist,downval,upval,dimnum=1,fignum=3):
        funclist=[]
        funclist.append(func)
        for i in range(len(smlist)):
            print(smlist[i].name)
            funclist.append(smlist[i].predict_values)
        locnum=100+fignum*10+1
        for i in range(fignum):
            if dimnum==1:
                ax=plt.subplot(locnum+i)
            else:
                ax=plt.subplot(locnum+i,projection='3d')
            if i<fignum-1:
                bal=fignum-1
            else:
                bal=1.0
            self.showcurve(plt,ax,funclist[i],downval,upval,dimnum=dimnum,bal=bal)
        plt.show()
        
    def _get_idx(self,errlist,devlist,dev2list,pool_err=2,pool_dev=3):
        id2=np.argmin(errlist)
        return id2
    def train(self):
        modelnum=len(model_list)
        modelid=[]
        self.modelpara=[]
        y0=self.yt
        lastidx=None
        down=np.min(self.xt,axis=0)
        up=np.max(self.xt,axis=0)
        
        num=1000
        montelist=[]
        montenum=10
##        for i in range(montenum):
##            montelist.append(tl.latian1_sampling(down,up,num))
##        monte_data=tl.latian1_sampling(self.down,self.up,num)
        monte_data=self.xt
        for i in range(self.maxiter):
            ylist=[]
            err_list=[]
            dev_list=[]
            dev2_list=[]
            modelparas=[]
            trainerrlist=[]
            for j in range(modelnum):
                sm=copy.deepcopy(model_list[j])
                sm2=copy.deepcopy(model_list[j])

                err=self._eva_err(self.xt,y0,sm,knum=5*np.shape(self.xt)[0]/6,kf=7,testnum=1)
                
                sm.set_training_values(self.xt,y0)
                sm.train()
                sm2.set_training_values(self.xt,y0)
                sm2.train()
                dev2,dev,yy,yyy=tl.monte_integral(monte_data,lambda x:self._cal_dev(x,sm))
##                if i<self.maxiter-1:
##                    newdata,newlabel=self._choose_pts(monte_data,y0,dev2,choice_num=int(5*np.shape(self.xt)[0]/6))
##                else:
##                    newdata,newlabel=self.xt,y0
##                newdata,newlabel=self.xt,y0
                monte_y=y0
##                monte_y=self.func(monte_data)
                newdata,newlabel,udata,ulabel=self._choose_pts(monte_data,monte_y,dev,choice_num=np.shape(monte_data)[0]-int((1+np.shape(monte_data)[0])/7))
##                self.show1dcurve(sm.predict_values,down,up,udata,ulabel,newdata,newlabel)
##                assert False
                sm.set_training_values(newdata,newlabel)
                sm.train()
##                if np.shape(self.xt)[-1]==1:
##                    self.show1dcurve(sm.predict_values,sm2.predict_values,self.func,self.down,self.up,udata,ulabel,newdata,newlabel)
##                else:
##                    self.show2dcurve(sm.predict_values,sm2.predict_values,self.func,self.down,self.up,udata,ulabel,newdata,newlabel)
                trainerr=np.sum(np.square(y0-sm.predict_values(self.xt)))
##                trainerr2=np.sum(np.square(y0-sm.predict_values(self.xt)))-np.sum(np.square(y0-sm2.predict_values(self.xt)))
##                if trainerr2<1e-5:
##                    print('**********')
##                trainerr=1*np.sum(np.square(y0-sm2.predict_values(self.xt)))+0.65*np.abs(np.sum(np.square(y0-sm.predict_values(self.xt)))-np.sum(np.square(y0-sm2.predict_values(self.xt))))
##                trainerr=np.sum(np.square(ulabel-sm.predict_values(udata)))
                trainerrlist.append(trainerr)
##                if i==self.maxiter-1:
##                    sm.set_training_values(self.xt,y0)
##                    sm.train()
                modelparas.append(sm)
                y=sm.predict_values(self.xt)
                ylist.append(y)
                err_list.append(err)
                dev_list.append(yy)
                dev2_list.append(yyy)
            errs=np.array(err_list)
            devs=np.array(dev_list)
            devs2=np.array(dev2_list)
            trainerrs=np.array(trainerrlist)
            if i<self.maxiter-1:
##                idx=self._get_idx(trainerrs,devs,dev2_list)
                idx=np.argmin(trainerrs)
                modelparas[idx].set_training_values(self.xt,y0)
                modelparas[idx].train()
                y0=y0-modelparas[idx].predict_values(self.xt)/self.maxiter
            else:
                idx=np.argmin(trainerrs)
                modelparas[idx].set_training_values(self.xt,y0)
                modelparas[idx].train()
##            print(idx)
                
##            print(idx,trainerrs,errs)
##            y0=y0-ylist[idx]
            self.modelpara.append(modelparas[idx])

##        print(errs,trainerrlist,devs,devs2)
##        idlist=self._get_idx(errs,devs,dev2_list)

        self.modelid=modelid
##        self.modelpara=modelpara
    def predict_values(self,x):
        result=0
        for i in range(self.maxiter):
##            self.draw_curve(self.func,self.modelpara,self.down,self.up,dimnum=np.shape(x)[-1],fignum=self.maxiter+1)
            if i<self.maxiter-1:
                result=result+self.modelpara[i].predict_values(x)/self.maxiter
            else:
                result=result+self.modelpara[i].predict_values(x)
        return result
if __name__=='__main__':
    sm=resi_hybrid()
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
