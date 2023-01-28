##from abaqus import *
##from odbAccess import *
from numpy import *
import os
##import load
import random
import itertools
import numpy as np
##from jobMessage import ANY_JOB, ANY_MESSAGE_TYPE
loadList=['Load-bt1','Load-bt2','Load-bt3']
base_dir=os.getcwd()

def change_of_load(pathName,modelName,jobName,bias):
    mdb=openMdb(pathName=pathName)
    myModel=mdb.models[modelName]
    for i in range(len(loadList)):
        myModel.loads[loadList[i]].setValuesInStep(stepName='Step-2',magnitude=float(bias[i]))
    myJob=mdb.Job(name=jobName,model=modelName,numCpus=8,numDomains=8)
    print('start job')
    myJob.submit()
    myJob.waitForCompletion()
    print('finish job')
    
def extract_result(odbPath,outPath,option):
    odb=openOdb(path=odbPath)
    stepkeys=odb.steps.keys()

    oface=odb.rootAssembly.nodeSets['SET-OUTPUT']
    S=odb.steps[stepkeys[-1]].frames[-1].fieldOutputs[option]
    oS=S.getSubset(region=oface)

    labels, xyz = [], []
    for node in oface.nodes[0]:
        labels.append(node.label)
        xyz.append(node.coordinates)
    cc = dict(zip(labels, xyz))

    ovalues=oS.values
    
    cpFile=file(outPath,'w')
    for i in range(len(ovalues)):
        coord=cc[ovalues[i].nodeLabel]
        cpFile.write(str(ovalues[i].nodeLabel)+' '+str(coord[0])+' '+str(coord[1])+' '+str(ovalues[i].data[-1])+'\n')
    cpFile.close()
    odb.close()
def getloss(filePath):
    fp=open(filePath)
    cpress=[]
    for line in fp.readlines():
        values=line.strip().split(' ')
        cpress.append(float(values[3]))
    varcpress=var(cpress)
    return varcpress
def cal_modelval(inputval,modelpath,modelname):
    change_of_load(modelpath,modelname,'test',inputval)
    extract_result(base_dir+'test.odb',base_dir+'/test.txt','U')
    loss=getloss(base_dir+'/test.txt')
    return loss
#初始采样
def all_sampling(mindata,maxdata,ptnum):
    length=len(mindata)
    ptlist=[]
    for i in range(ptnum):
        for j in range(ptnum):
            a=[i,j]
            smallist=[]
            for m in range(len(a)):
                downval=mindata[m]+a[m]*(maxdata[m]-mindata[m])/ptnum
                upval=mindata[m]+(a[m]+1)*(maxdata[m]-mindata[m])/ptnum
                smallist.append(random.uniform(downval,upval))
            ptlist.append(smallist)
    return transpose(array(ptlist))
#均匀采样                
def uni_sampling(mindata,maxdata,ptnum):
    length=len(mindata)
    num=int(sqrt(ptnum))
    ptlist=[]
    for i in range(num):
        for j in range(num):
            a=[i,j]
            smallist=[]
            for m in range(len(a)):
                downval=mindata[m]+a[m]*(maxdata[m]-mindata[m])/num
                upval=mindata[m]+(a[m]+1)*(maxdata[m]-mindata[m])/num
                smallist.append((downval+upval)/2)
            ptlist.append(smallist)
    return transpose(array(ptlist))
#拉丁超立方采样,返回长度*点数
def latian1_sampling(mindata,maxdata,ptnum,trans=True):
    length=len(mindata)
    ptlist=[]
    for i in range(length):
        smalist=[]
        for j in range(ptnum):
            downval=mindata[i]+j*(maxdata[i]-mindata[i])/ptnum
            upval=mindata[i]+(j+1)*(maxdata[i]-mindata[i])/ptnum
            smalist.append(random.uniform(downval,upval))
        random.shuffle(smalist)
        ptlist.append(array(smalist))
    pts=vstack(ptlist)
    if trans:
        pts=pts.T
    return pts
#最远区域采样.data:2*n
def fps_sampling(data,ptnum):
    dimsize,allnum=shape(data)
    newids=[0]
    data1=expand_dims(data,axis=1)
    data2=expand_dims(data,axis=2)
    dismat=sqrt(sum(square(data1-data2),axis=0))#n*n
    for i in range(ptnum-1):
        farmat=dismat[newids]
        newid=argmax(amin(farmat,axis=0))
        newids.append(newid)
##    newdata=data[newids]
    return newids
#修改版的超立方采样,可以去除中间区域
def latian2_sampling(mindata,maxdata,minv,maxv,ptnum):
    length=len(mindata)
    ptlist=[]
    ptnum1=ptnum+1
    for i in range(length):
        smalist=[]
        valrange=(maxdata[i]-mindata[i])-(maxv[i]-minv[i])
        sym=0
        index=0
        for j in range(ptnum1):
            if sym==0:
                downval=mindata[i]+j*valrange/ptnum1
                upval=mindata[i]+(j+1)*valrange/ptnum1
            else:
                downval=maxdata[i]-(ptnum-index)*valrange/ptnum1
                upval=downval+valrange/ptnum1

            if upval>minv[i] and upval<maxv[i]:
                sym=1
                index=j+1
                continue

            
            smalist.append(random.uniform(downval,upval))
        random.shuffle(smalist)
        ptlist.append(array(smalist))
    pts=vstack(ptlist)
    return pts
def cal_funcval(inputval,func):
    return func(inputval)
def SC_func(pts):
    pts=pts.T
    pt12=pts[0]*pts[0]
    pt14=pt12*pt12
    pt16=pt14*pt12
    pt22=pts[1]*pts[1]
    pt24=pt22*pt22
    outval=4*pt12-2.1*pt14+pt16/3+pts[0]*pts[1]-4*pt22+4*pt24
    return outval
def GN_func(pts):
    pts=pts.T
    p1=pts[0]*pts[0]/200+pts[1]*pts[1]/200
    p2=cos(pts[0])*cos(pts[1]/sqrt(2))
    return p1-p2+1
def goldstein_price(X):
    X=X.T
    x=X[0]
    y=X[1]
    result1=1+(19-14*x+13*x**2-14*y+6*x*y+3*y**2)*(x+y+1)**2
    result2=30+(18-32*x+12*x**2-48*y-36*x*y+27*y**2)*(2*x-3*y)**2
    return result1*result2
def branin_hoo(X):
    x = X[0]
    y = X[1]
##    X1 = 15*x-5
##    X2 = 15*y
    a = 1
    b = 5.1/(4*pi**2)
    c = 5/pi
    d = 6
    e = 10
    ff = 1/(8*pi)
    return a*( y - b*x**2 + c*x - d )**2 + e*(1-ff)*cos(x) + e
def test_func1(X):
    x=X
    result=sin(2*(6*x-2))*(6*x-2)**2
    return result
def test_func2(X):
    X=X.T
    x = X[0]
    y = X[1]
    result=(30+x*sin(x))*(4+exp(-y**2))
    return result
#3*n
def hartmann_func3(X):
    X=X.T
    if len(shape(X))<2:
        X=expand_dims(X,axis=-1)
    n=shape(X)[-1]
    A=array([[3.0,10,30],[0.1,10,35],[3.0,10,30],[0.1,10,35]])
    P=array([[0.3689,0.1170,0.2673],[0.4699,0.4387,0.7470],[0.1091,0.8732,0.5547],[0.03815,0.5743,0.8828]])
    c=array([1,1.2,3,3.2])
    cc=-sum(square(expand_dims(X.T,axis=1)-P)*A,axis=-1)
    result=sum(c*cc,axis=-1)
    return result
#6*n    
def hartmann_func6(X):
    X=X.T
    if len(shape(X))<2:
        X=expand_dims(X,axis=-1)
    n=shape(X)[-1]
    A=array([[10.0,3.0,17.0,3.5,1.7,8.0],[0.05,10.0,17.0,0.1,8.0,14.0],\
             [3.0,3.5,1.7,10.0,17.0,8.0],[17.0,8.0,0.05,10.0,0.1,14.0]])
    P=array([[0.1312,0.1696,0.5569,0.0124,0.8283,0.5886],\
             [0.2329,0.4135,0.8307,0.3736,0.1004,0.9991],\
             [0.2348,0.1451,0.3522,0.2883,0.3047,0.6650],\
             [0.4047,0.8828,0.8732,0.5743,0.1091,0.0381]])
    c=array([1,1.2,3,3.2])
    cc=-sum(square(expand_dims(X.T,axis=1)-P)*A,axis=-1)
    result=sum(c*cc,axis=-1)
    return result
#2*n
def self_func4(X):
    x = X[0]
    y = X[1]
    result=x**4+y**3+x**2+y**1+(x**3)*y
    return result
#2*n
def self_func5(X):
    x = X[0]
    y = X[1]
    result=x**5+y**4+x**3+y**2+(x**4)*y
    return result
#2*n
def self_func6(X):
    x = X[0]
    y = X[1]
    result=x**6+y**5+x**4+y**3+(x**5)*y
    return result
#2*n
def self_func7(X):
    x = X[0]
    y = X[1]
    result=x**7+y**6+x**5+y**4+(x**6)*y
    return result
#2*n
def self_func8(X):
    x = X[0]
    y = X[1]
    result=x**8+y**7+x**6+y**5+(x**7)*y
    return result

from Touchstone_master.touchstone.models.ackley import Ackley
#(-8.192,-8.192),(8.192,8.192)
def ackley2(X):
    model=Ackley(n_dim=2,bounds='small')
    result=model.evaluate(X.T)
    result=reshape(result,[-1,1])
    return result
def ackley3(X):
    model=Ackley(n_dim=3,bounds='small')
    result=model.evaluate(X.T)
    result=reshape(result,[-1,1])
    return result
from Touchstone_master.touchstone.models.booth import Booth
#(-10,-10),(10,10)
def booth(X):
    model=Booth()
    result=model.evaluate(X.T)
    result=reshape(result,[-1,1])
    return result
from Touchstone_master.touchstone.models.bukin import Bukin
#(-15,-3),(-5,3)
def bukin(X):
    model=Bukin()
    result=model.evaluate(X.T)
    result=reshape(result,[-1,1])
    return result
from Touchstone_master.touchstone.models.crossintray import CrossInTray
#(-10,-10),(10,10)
def crossintray(X):
    model=CrossInTray()
    result=model.evaluate(X.T)
    result=reshape(result,[-1,1])
    return result
from Touchstone_master.touchstone.models.dropwave import DropWave
#(-5.12,-5.12),(5.12,5.12)
def dropwave(X):
    model=DropWave()
    result=model.evaluate(X.T)
    result=reshape(result,[-1,1])
    return result
from Touchstone_master.touchstone.models.eggholder import Eggholder
#(-512,-512),(512,512)
def eggholder(X):
    model=Eggholder()
    result=model.evaluate(X.T)
    result=reshape(result,[-1,1])
    return result
from Touchstone_master.touchstone.models.gramacylee import GramacyLee
#(0.5,2.5)
def gramacylee(X):
    model=GramacyLee()
    result=model.evaluate(X.T)
    result=reshape(result,[-1,1])
    return result
from Touchstone_master.touchstone.models.holdertable import HolderTable
#(-10,-10),(10,10)
def holdertable(X):
    model=HolderTable()
    result=model.evaluate(X.T)
    result=reshape(result,[-1,1])
    return result
from Touchstone_master.touchstone.models.langermann import Langermann
#(0,0),(10,10)
def langermann(X):
    model=Langermann()
    result=model.evaluate(X.T)
    result=reshape(result,[-1,1])
    return result
#(-10,10),(n*4)
def test_4dim1(X):
    result=100*((X[:,0])**2-X[:,1])**2+(X[:,0]-1)**2+90*(X[:,2]**2-X[:,3])**2
    result=result+10.1*((X[:,1]-1)**2+(X[:,3]-1)**2)+19.8*(X[:,1]-1)*(X[:,3]-1)
    result=reshape(result,[-1,1])
    return result
#(-10,10),(n*4)
def test_4dim2(X):
    result=100*(X[:,0]-X[:,1])**2+(X[:,0]-1)**2+90*(X[:,2]**2-X[:,3])**2+(X[:,2]-1)**2
    result=result+10.1*((X[:,1]-1)**2+(X[:,3]-1)**2)+19.8*(X[:,1]-1)*(X[:,3]-1)
    result=reshape(result,[-1,1])
    return result
#(0,1),(n*4)
def test_4dim3(X):
    result=10*sin(2*(X[:,0]-0.6*pi))+X[:,1]+X[:,2]+X[:,3]+X[:,0]*X[:,1]+X[:,1]*X[:,2]+X[:,0]**3+X[:,3]**3
    result=reshape(result,[-1,1])
    return result
#(0,10^5),(n*6)
def test_6dim1(X):
    x1,x2,x3,x4,x5,x6=X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5]
    result=0.0204*x4*x1**2+x2*x3+0.01870*x1*x2*x3+1.57*x2*x4+0.0607*x1*x4*(x1+x2+x3)*x5**2
    result=result+0.0437*x2*x3*(x1+1.57*x2+x4)*x6**2
    result=reshape(result,[-1,1])
    return result
#(-1,1),(n*8)
def test_8dim1(X):
    result=0.3+sin(16*X/15-1)+sin(16*X/15-1)**2
    result=np.sum(result,axis=-1)
    result=reshape(result,[-1,1])
    return result
#(-1,1),(n*10)
def test_10dim1(X):
    result=0.3+sin(16*X/15-1)+sin(16*X/15-1)**2
    result=np.sum(result,axis=-1)
    result=reshape(result,[-1,1])
    return result
#(0,1),(n*10)
def test_10dim2(X):
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10=X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5],X[:,6],X[:,7],X[:,8],X[:,9]
    result=x1**2+x2**2+x1*x2-14*x1-16*x2+((x3-10)**2)*((x4-5)**2)+(x5-3)**2+2*(x6-1)**2+5*x7**2+2*(x9-10)**2
    result=result+2*(x10-7)**2+45
    result=reshape(result,[-1,1])
    return result
#(-10,10),(n*15)
def test_15dim1(X):
    result=(X[:,0]-1)**2
    for i in range(14):
        result=result+(i+2)*(2*X[:,i+1]**2-X[:,i])**2
    result=reshape(result,[-1,1])
    return result
#(-5,10),(n*15)
def test_15dim2(X):
    result=0
    for i in range(14):
        result=result+100*(X[:,i+1]-X[:,i]**2)**2+(X[:,i]-1)**2
    result=reshape(result,[-1,1])
    return result
#(-5,5),(n*16)
def test_16dim1(X):
    result=(X[:,0]-1)**2
    for i in range(15):
        result=result+(i+2)*(2*X[:,i+1]**2-X[:,i])**2
    result=reshape(result,[-1,1])
    return result

def judge_same(l1,l2):
    re=[x for x in l1 if x not in l2]
    
    return len(re)>0
def list_same(l1,mat2):
    err=sum(np.min(square(array(l1)-array(mat2)),axis=0))
    if err<0.01:
        return False
    else:
        return True
def elem_same(l1,l2):
    sl1=sorted(list(l1))
    sl2=sorted(list(l2))
    if sl1==sl2:
        return True
    else:
        return False
def elem_exist(l1,mat):
    al=array(l1)
    err=np.min(np.sum(np.square(al-array(mat)),axis=-1))
    if err<0.001:
        return False
    else:
        return True
def err_evaluate(data,pts,b,degree=2,limit=4,\
                 getdegree=True,idin=None,use_idlist=False,k=0.5,threshold=0.3,eva_type='k',delta=5):
    ptnum,dimnum=np.shape(pts)
    dis=np.sum(np.square(expand_dims(pts,axis=0)-expand_dims(pts,axis=1)),axis=-1)#ptnum*ptnum
    maxdis=np.max(dis)
    evadis=np.sum(np.square(expand_dims(pts,axis=1)-expand_dims(data,axis=0)),axis=-1)#ptnum*evanum
    judge=np.sum(np.where(evadis<threshold*maxdis,1,0),axis=-1)#ptnum*1
##    print(judge[:10])
##    idx=[i for i in range(ptnum)]
    if eva_type=='k':
        itertime=1
        errlist=[]
        idlists=[]
        if getdegree:
            itertime=limit-1
            degree=1
        for i in range(itertime):
            idx=argsort(-judge)
            topk=int(k*ptnum)
            traindata=pts[idx[topk:]]
            trainb=b[idx[topk:]]
            testdata=pts[idx[:topk]]
            testb=b[idx[:topk]]
            beta,idlist=solve_polysrbf(traindata,trainb,degree,idin,use_idlist,delta=delta)
            result=cal_polysrbf(beta,testdata,traindata,idlist,degree,delta=delta)#k*1
            degree+=1
            err=np.sum(np.square(result-testb))
            idlists.append(idlist)
            errlist.append(err)
        bestid=argmin(errlist)
        bestdegree=bestid+1
        bestidlist=idlists[bestid]
    else:
        if eva_type=='w':
            weights=judge/np.sum(judge)
        else:
            weights=ones((ptnum,1))
        err=[]
        errlist=[]
        idlists=[]
        if getdegree:
            itertime=limit-1
            degree=1
        for i in range(itertime):
            err=[]
            idlist=[]
            for j in range(ptnum):
                ids=list(range(ptnum))
                des=ids[j]
                del[ids[j]]
                beta,idlist=solve_polysrbf(pts[ids],b[ids],degree,idin,use_idlist,delta=delta)
                result=cal_polysrbf(beta,expand_dims(pts[des],0),pts[ids],idlist,degree,delta=delta)#k*1
                err.append(np.square(result-b[des]))
            idlists.append(idlist)
            errlist.append(sum(err)*weights[i])
            degree+=1
##        print(idlists)
        bestid=argmin(errlist)
        bestdegree=bestid+1
        bestidlist=idlists[bestid]
##    print(bestdegree,errlist)
    return bestdegree,bestidlist
        
        
            
        
#X:多项式矩阵ptnum*dimnum
#F:RBF矩阵ptnum*ptnum
def err_judge(X,F,b):
    ptnum,dimnum=shape(X)
##    errF=dot(eye(ptnum)-dot(transpose(F),F),\
##             (eye(ptnum)-dot(linalg.pinv(dot(transpose(F),F)),transpose(F))))
    errF=eye(ptnum)-dot(dot(F,linalg.pinv(dot(transpose(F),F))),transpose(F))
    errX=dot(dot(X,linalg.pinv(dot(transpose(X),X))),transpose(X))-eye(ptnum)
    err=dot(dot(errF,errX),b)
    error=sum(square(err))
    return error
#data:n*2
def polys(data,degree,idin=None,use_idlist=False):
    ptnum,dimnum=shape(data)
    idlist=[]
    result=[]
    data=hstack((ones((ptnum,1)),data))
    if use_idlist and idin is not None:
        idlist=idin
    else:
        #对每个多项式组分分别计算
        for item in itertools.product(* [[i for i in range(dimnum+1)]]*degree):
            symbol_same=False
    ##        thislist=prod(data[:,list(item)],axis=-1)
            sorted_id=sorted(list(item))
            #if len(result)==0 or list_same(thislist,result):
            if len(idlist)==0 or elem_exist(sorted_id,idlist):
                idlist.append(sorted_id)
    ##            result.append(thislist)
            
    ##        for l1 in reidx:
    ##            if ~judge_same(thislist,l1):
    ##                symbol_same=True
    ##                break
    ##        if symbol_same:
    ##            continue
    ##        reidx.append(thislist)
    ##        result.append(prod(data[thislist],axis=0))
    ##    print(reidx)
    result=array(np.prod(data[:,idlist],axis=-1))
##    result=transpose(array(result))#n*m
    return result,array(idlist)
def RBF(data,delta):
    a=sum(square(expand_dims(data,axis=2)-expand_dims(data,axis=1)),axis=0)+delta+1e-5
    b=1/sqrt(a)
    return b
def get_degree(data,b,limit,delta):
    errs=[]
    F=RBF(transpose(data),delta)
    idlist=[]
    for i in range(1,limit+1):
        X,thislist=polys(data,i)
        idlist.append(thislist)
        errs.append(err_judge(X,F,b))
    idx=argmin(errs)
    minvalue=min(errs)
    idresult=idlist[idx]
    print(errs)
    return idx,idresult
#暂时是RBF
#data:dnum*dim
#pts:pnum*dim
#beta:pnum*1
def cal_rbfgradient(data,pts,beta,delta=5):
    a=expand_dims(data,axis=1)-expand_dims(pts,axis=0)
    dis=sum(square(a),axis=-1)+delta
    base=1/((dis*sqrt(dis))+1e-5)#dnum*pnum
    base=expand_dims(base*transpose(beta),axis=-1)#dnum*pnum*1
    vec=a#dnum*pnum*dim
    result=sum(vec*base,axis=1)
    return result

#pts:pnum*dim
#y:pnum*1
def solve_RBF(pts,y,delta=5):
    length=shape(pts)[1]
    ptnum=shape(pts)[0]
    a=expand_dims(pts,axis=1)-expand_dims(pts,axis=0)
    dis=sum(square(a),axis=-1)+delta
    A=1/sqrt(dis)
    A=array(A)
    result=reshape(dot(linalg.pinv(A),y),[ptnum,1])
    return result
#base:ptnum*1
#pts:ptnum*dimnum
#data:ptnum2*dimnum

def cal_RBF(base,data,pts,delta=5):
    a=np.expand_dims(data,axis=1)-np.expand_dims(pts,axis=0)
    dis=np.sum(np.square(a),axis=-1)+delta
    A=1/np.sqrt(dis)
    A=np.array(A)
    result=np.sum(A*np.transpose(base),axis=-1,keepdims=True)#ptnum*1
    return result

#pts:pnum*dim
#y:pnum*1
def solve_polys(pts,y,idin=None,use_idlist=False,degree=2):
    A,idlist=polys(pts,degree,idin,use_idlist)#pnum*polynum
    result=reshape(dot(linalg.pinv(A),y),[shape(A)[1],1])
    return result,A,idlist
def solve_polysrbf(pts,y,degree,idin=None,use_idlist=False,delta=5):
    beta1,A,idlist=solve_polys(pts,y,idin,use_idlist,degree)
    result1=dot(A,beta1)
##    print(y-result1)
    beta2=solve_RBF(pts,y-result1,delta)
    beta=vstack((beta1,beta2))
    return beta,idlist
def cal_polysrbf(beta,data,pts,idlist,degree,delta=5):
##    data=np.atleast_2d(data.T)
##    pts=np.atleast_2d(pts.T).T
##    print(shape(data),shape(pts))
    ptnum,dimnum=shape(pts)
    bnum=shape(beta)[0]
    beta1,beta2=beta[:bnum-ptnum],beta[bnum-ptnum:]
    poly_result=cal_polys(beta1,data,idlist)
    rbf_result=cal_RBF(beta2,data,pts,delta)
##    print(rbf_result)
    result=poly_result+rbf_result
##    print(result)
    return result
    
def cal_gradient(data,pts,degree,beta,idlist,delta=5):
    ptnum,dimnum=shape(pts)
    bnum=shape(beta)[0]
    beta1,beta2=beta[:bnum-ptnum],beta[bnum-ptnum:]
    poly_gra=cal_polys_gradient(beta1,data,degree,idlist)
    rbf_gra=cal_rbfgradient(data,pts,beta2,delta)
    result=poly_gra+rbf_gra
    return result

#base:polynum*1
#data:n*dim
#degree
#idlist:polynum*(dim+1)
#output:n*dim
def cal_polys_gradient(base,data,degree,idlist):
    ptnum,dimnum=shape(data)
    polynum=shape(idlist)[0]
    data=hstack((ones((ptnum,1)),data))#(n+1)*dim
    result=[]
##    print(idlist)
    for i in range(dimnum):
        sub_array=(i+1)*ones((polynum,1))
        idx,idy=where(idlist==i+1)
        gnum=sum(where(idlist==i+1,1,0),axis=1)
        gnum=expand_dims(gnum,axis=0)
        id=unique(idx)
        chglist=[]
        new_idlist=idlist.copy()
        for item in id:
            ids=where(idx==item)[0][0]
            new_idlist[item,idy[ids]]=0
        reslist=sum(prod(data[:,new_idlist],axis=-1)*gnum*transpose(base),axis=-1,keepdims=True)#36*1
        result.append(reslist)
    result=-hstack(result)
    return result
def cal_polys(base,data,idlist):
    ptnum,dimnum=shape(data)
    polynum=shape(idlist)[0]
    data=hstack((ones((ptnum,1)),data))#(n+1)*dim
    reslist=sum(prod(data[:,idlist],axis=-1)*transpose(base),axis=-1,keepdims=True)#36*1
    return reslist
#data:n*2,y:n*1
def search_peak(downval,upval,data,y,learning_rate=0.1,peaknum=10,itertime=50,delta=0.1,surro='r'):
    #if surro=='r':
    delta=delta
    itertime=itertime
    lr=learning_rate
    beta=solve_RBF(data,y,delta=delta)
    ptnum,dimnum=np.shape(data)
    pnum=peaknum
    x=latian1_sampling(downval,upval,peaknum)#pnum*2
##    print(downval)
    for i in range(itertime):
        gradients=cal_rbfgradient(x,data,beta,delta=delta)#pnum*2
##        if i==0:
##            print(gradients)
        x=x-lr*gradients
##    print(x)
##    print(gradients)
    dismat=np.sqrt(np.sum(np.square(np.expand_dims(data,axis=1)-np.expand_dims(x,axis=0)),axis=-1))#ptnum*pnum
    peakid=np.argmin(dismat,axis=1)#(ptnum,)
    rowid=list(range(ptnum))
##    peakidx=np.concatenate([np.expand_dims(rowid,axis=-1),np.expand_dims(peakid,axis=-1)],axis=-1)
##    print(np.shape(dismat),peakidx)
    choicemat=dismat
    for i in range(ptnum):
        choicemat[i,peakid[i]]=10000
##    choicemat[peakidx]=10000
    region_r=np.min(choicemat,axis=0)#(pnum,)
    local_pts=[]
    local_ys=[]
    srange=0.01*(upval-downval)
    for i in range(pnum):
        local_ptnum=sum(peakid==i)
        pts_data=data[peakid==i]
        ys_data=y[peakid==i]
        if local_ptnum<10:
            xins=latian1_sampling(x[i]-srange,x[i]+srange,10-local_ptnum)
            yins=cal_RBF(beta,xins,data,delta=delta)
            if local_ptnum>0:
                pts_data=np.concatenate([pts_data,xins],axis=0)
                ys_data=np.concatenate([ys_data,yins],axis=0)
            else:
                pts_data=xins
                ys_data=yins
        local_pts.append(pts_data)
        local_ys.append(ys_data)
    return region_r,x,local_ys,local_pts
#data:n*dimnum
#output:1
def monte_integral(data,func):
##    data=latian1_sampling(down,up,num)
    dev2,dev=cal_dev2(data,func)#n*dimnum
##    result=np.exp(-5*np.min(np.abs(dev2),axis=1))
    result=np.max(np.max(np.abs(dev2),axis=1),axis=1)
    result=np.sum(result)
##    print(dev,dev2)
##    assert False
    result2=np.exp(-5*np.min(np.min(np.abs(dev2),axis=1),axis=1))
##    result2=np.sum(np.abs(dev),axis=-1)
    result2=np.sum(result2)
    return dev2,dev,result,result2
    
#data:n*dimnum
#output:n*dimnum
def cal_dev2(data,func):
    ptnum,dimnum=np.shape(data)
    tiny_dis=1e-5
    movemat=np.eye(dimnum)*tiny_dis
    alldata=np.expand_dims(movemat,axis=0)+np.expand_dims(data,axis=1)#n*dimnum*dimnum
    alldata=np.reshape(alldata,[-1,dimnum])
    yy=func(alldata)#(n*dimnum)*1
##    print(np.shape(alldata))
    yy=np.reshape(yy,[ptnum,dimnum,-1])
    y=np.expand_dims(func(data),axis=1)#n*1*dimnum
##    print(y)
##    print(yy)
##    print(data,alldata)
    dev2=(yy-y)/tiny_dis#n*dimnum*dimnum
##    print(np.shape(yy),np.shape(y),ptnum,dimnum)
##    print(y)
##    dev2=np.reshape(dev2,[ptnum,-1])
##    dev=np.reshape(np.array(y),[ptnum,-1])
    return dev2,np.array(y)
    
    
if __name__=='__main__':
##    x1=array([1.0,2.0,3.0,4.0,5.0,6.0]).T
##    x2=array([[1.0,2.0,3.0,4.0,5.0,6.0],[1.5,2.5,3.5,4.0,5.0,6.0]]).T
##    print(hartmann_func6(x1),hartmann_func6(x2))
##    pts=latian1_sampling([0,0,0,0,0,0],[1,1,1,1,1,1],36)
    pts=latian1_sampling([-100,-100],[100,100],36)
    pts2=latian1_sampling([-100,-100],[100,100],50)
##    b=goldstein_price(pts)
    b=expand_dims(GN_func(pts),axis=1)
    y=expand_dims(GN_func(pts2),axis=1)
    pts=transpose(pts)
    pts2=pts2.T
##    polys(pts,3)
##    b=hartmann_func6(pts)
##    b=branin_hoo(pts)
##    degree,idlist=get_degree(pts,b,4,delta=5)
##    degree,idlist=err_evaluate(pts2,pts,b,degree=2,limit=4,getdegree=True,idin=None,use_idlist=False,k=0.5,threshold=0.3,eva_type='w')
##    print(degree,idlist)
    beta,idlist=solve_polysrbf(pts,b,2,None,False,delta=5)
    result=cal_polysrbf(beta,pts2,pts,idlist,2,delta=5)
    print(shape(result))
    err=sqrt(mean(square(result-y)))
    print(err)
##    beta,idlist=solve_polysrbf(pts,b,degree,delta=5)
##    gra=cal_gradient(pts2,pts,degree,beta,idlist,delta=5)
##    print(gra)
    
    
##    print(fps_sampling(pts,6))
##    sampts=pts[:,fps_sampling(pts,6)]
##    print(sampts)
    
            
            
