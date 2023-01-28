import matplotlib.pyplot as plt
import matplotlib as mpl
import xlrd
import numpy as np

import sys
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
from matplotlib.font_manager import FontProperties

import random

##font_set = FontProperties(fname = r"***path***/mpl-data/fonts/simfang.ttf")
def xls2sheet(path,sheet_index):
    file=xlrd.open_workbook(path)
    val_sheet=file.sheets()[sheet_index]
##    rownum=val_sheet.max_row()
##    colnum=val_sheet.max_column()
    return val_sheet
def get_fdv_data(sheet,modelnum=5,typenum=3):
    rownum=sheet.nrows
    colnum=sheet.ncols
    funcnum=rownum-1
    recordnum=int(colnum/(2*typenum))
    data=np.zeros((funcnum,recordnum,2,typenum))
    
    for i in range(funcnum):
        for j in range(recordnum):
            for k in range(typenum):
                data[i,j,0,k]=sheet.cell(i+1,2*j*typenum+2*k).value
                data[i,j,1,k]=sheet.cell(i+1,2*j*typenum+2*k+1).value
    return data
def get_drs_data(sheet,typenum=6):
    rownum=sheet.nrows
    colnum=sheet.ncols
##    print(rownum,colnum)
    funcnum=rownum-1
    recordnum=int((colnum-7)/typenum)
    data=np.zeros((funcnum,recordnum,typenum))#27*4*6
##    timedata=np.zeros((funcnum,recordnum,typenum))
    for i in range(funcnum):
        for j in range(recordnum):
            for k in range(typenum):
                data[i,j,k]=sheet.cell(i+1,j+k*recordnum+1).value
    timedata=data[:,-1,:]
    return data,timedata
def get_val_data(sheet,typenum=3):
    rownum=sheet.nrows
    colnum=sheet.ncols
    funcnum=rownum-1
    timenum=int(colnum/(3*typenum))
    data=np.zeros((funcnum,timenum,2,typenum))
    timedata=np.zeros((funcnum,timenum,typenum))
    for i in range(funcnum):
        for j in range(timenum):
            for k in range(typenum):
                data[i,j,0,k]=sheet.cell(i+1,3*j*typenum+3*k).value
                timedata[i,j,k]=sheet.cell(i+1,3*j*typenum+3*k+1).value
                data[i,j,1,k]=sheet.cell(i+1,3*j*typenum+3*k+2).value
    return data,timedata
def get_multidrs_data(spath_list):
    datalist=[]
    timelist=[]
    for i in range(len(spath_list)):
        sheet=xls2sheet(spath_list[i],0)
        data,timedata=get_drs_data(sheet,typenum=9)
        datalist.append(np.expand_dims(data,axis=0))
        timelist.append(np.expand_dims(timedata,axis=0))
    datalist=np.concatenate(datalist,axis=0)
    timelist=np.concatenate(timelist,axis=0)#4*27*4*6
    return datalist,timelist

def get_multival_data(spath_list):
##    spath_list=['./results/result_all.xls','./results/result_nokrg.xls','./result_nokrg_norbfmq.xls','./results/result_qp_svr.xls']
    datalist=[]
    timelist=[]
    for i in range(len(spath_list)):
        sheet=xls2sheet(spath_list[i],3)
        data,timedata=get_val_data(sheet)
        datalist.append(np.expand_dims(data,axis=0))
        timelist.append(np.expand_dims(timedata,axis=0))

    datalist=np.concatenate(datalist,axis=0)
    timelist=np.concatenate(timelist,axis=0)
    return datalist,timelist

##def get_eva_bar(sheet,typenum=5):
#data:funcnum*typenum
def multi_bars(plt,ax,data,xname,yname):
    bar_width = 0.2
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    color_list=['b','m','r']
    name_list=['c','r','2d']
    typenum=np.shape(data)[1]
    funcnum=np.shape(data)[0]
    index=np.arange(funcnum)
##    print(np.squeeze(data[:,0]))
    for i in range(typenum):
        ax.bar(index+bar_width*i,np.squeeze(data[:,i]),bar_width,alpha=opacity,color=color_list[i],error_kw=error_config,label=name_list[i])
    ax.set_xticks(index+typenum*bar_width/typenum)
    ax.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    
def funcs_multibars(plt,ax,data,xname,yname):
    bar_width = 0.2
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    color_list=['#9999ff','#E599E5','#FF9999','#99CC99']
    name_list=['Kriging+RBF_MQ+RBF+QP+SVR','RBF_MQ+RBF+QP+SVR','RBF+QP+SVR','QP+SVR']
    funcnum=np.shape(data)[1]
    barnum=np.shape(data)[0]
    typenum=np.shape(data)[-1]
    index=np.arange(typenum)
    btlist=[]
##    print(np.squeeze(data[:,0]))
##    print(np.squeeze(np.mean(data,axis=1)[0]))
    for i in range(barnum):
        bt=ax.boxplot(x = data[i], # 指定绘图数据

            positions=index+bar_width*i,

            widths=bar_width,

            autorange=True,
 
            patch_artist=True, # 要求用自定义颜色填充盒形图，默认白色填充
 
            showmeans=False, # 以点的形式显示均值

            showfliers=False,

            meanline=False,

            labels=['K-fold','Simple','FDV'],
 
            boxprops = {'color':'white','facecolor':color_list[i]}, # 设置箱体属性，填充色和边框色
 
            flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # 设置异常值属性，点的形状、填充色和边框色
 
            meanprops = {'marker':'D','markerfacecolor':'indianred'}, # 设置均值点的属性，点的形状、填充色
 
            medianprops = {'linestyle':'--','color':'orange'}) # 设置中位数线的属性，线的类型和颜色
        btlist.append(bt['boxes'][0])
    ax.set_xticks(index+typenum*bar_width/typenum)
    plt.tick_params(labelsize=10)
    ax.legend(btlist,name_list,loc='upper left',fontsize=10)
##    plt.xlabel(xname)
    plt.ylabel(yname)
##    fig.tight_layout()
def drs_multibars(plt,ax,data,xname,yname):
    bar_width = 0.2
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    color_list=['#9999ff','#E599E5','#FF9999','#99CC99']
    name_list=['Kriging+RBF_MQ+RBF+QP+SVR','RBF_MQ+RBF+QP+SVR','RBF+QP+SVR','QP+SVR']
    funcnum=np.shape(data)[1]
    barnum=np.shape(data)[0]
    typenum=np.shape(data)[-1]
    index=np.arange(typenum)
##    print(np.shape(data))
##    assert False
    btlist=[]
    for i in range(barnum):
        bt=ax.boxplot(x = data[i], # 指定绘图数据

            positions=index+bar_width*i,

            widths=bar_width,

            autorange=True,
 
            patch_artist=True, # 要求用自定义颜色填充盒形图，默认白色填充
 
            showmeans=False, # 以点的形式显示均值

            showfliers=False,

            meanline=False,

            labels=['DRS','AHF','ES_HGL','UES','QP','Kriging','RBF_MQ','RBF','SVR'],
 
            boxprops = {'color':'white','facecolor':color_list[i]}, # 设置箱体属性，填充色和边框色
 
            flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # 设置异常值属性，点的形状、填充色和边框色
 
            meanprops = {'marker':'D','markerfacecolor':'indianred'}, # 设置均值点的属性，点的形状、填充色
 
            medianprops = {'linestyle':'--','color':'orange'}) # 设置中位数线的属性，线的类型和颜色
        btlist.append(bt['boxes'][0])
##        bt=ax.bar(index+bar_width*i,data[i],bar_width,alpha=opacity,color=color_list[i],error_kw=error_config,label=name_list[i])
##        for b in bt:
##            h=b.get_height()
##            ax.text(b.get_x()+b.get_width()/2,h,format(h,'.4e'),ha='center',va='bottom')
    ax.set_xticks(index+typenum*bar_width/typenum)
##            plt.xticks(index,['DRS','AHF','ES_HGL','UES','QP','Kriging','RBF_MQ','RBF','SVR'])
##    ax.set_xticks(index+typenum*bar_width/typenum)
##    ax.legend(loc='upper left',fontsize=8)
    plt.tick_params(labelsize=10)
    ax.legend(btlist,name_list,loc='upper left',fontsize=10)
##    plt.xlabel(xname)
    plt.ylabel(yname)
##    fig.tight_layout()
def single_bar(plt,ax,data,xname,yname):
    bar_width = 0.6
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    color_list=['b','m','r']
    name_list=['c','r','2d']
    typenum=np.shape(data)[1]
    funcnum=np.shape(data)[0]
    index=np.arange(typenum)
    print(np.shape(data))
    bt=ax.boxplot(x = data, # 指定绘图数据

            autorange=True,
 
            patch_artist=True, # 要求用自定义颜色填充盒形图，默认白色填充
 
            showmeans=False, # 以点的形式显示均值

            showfliers=False,

            meanline=False,

            labels=['K-fold','Simple','FDV'],
 
            boxprops = {'color':'black','facecolor':'#9999ff'}, # 设置箱体属性，填充色和边框色
 
            flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # 设置异常值属性，点的形状、填充色和边框色
 
            meanprops = {'marker':'D','markerfacecolor':'indianred'}, # 设置均值点的属性，点的形状、填充色
 
            medianprops = {'linestyle':'--','color':'orange'}) # 设置中位数线的属性，线的类型和颜色
##    bt=ax.bar(index,np.squeeze(np.mean(data,axis=0)),bar_width,alpha=opacity,color=color_list[0])
##    for b in bt:
##        h=b.get_height()
##        ax.text(b.get_x()+b.get_width()/2,h,format(h,'.4e'),ha='center',va='bottom')
##    ax.set_xticks(index+typenum*bar_width/typenum)
##    plt.xticks(index,['K-fold','Simple','FDV'])
##    plt.xticks(index,['SDV','Random','FDV'])
    plt.tick_params(labelsize=10)
##    ax.legend()
##    plt.xlabel(xname)
    plt.ylabel(yname)
def single_funcbar(plt,ax,data,xname,yname):
    bar_width = 0.6
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    color_list=['b','m','r','g','y','p']
    typenum=len(data)
##    print(np.shape(data))
    index=np.arange(typenum)
    data=np.transpose(data).tolist()
    print(np.shape(data))
    print(np.mean(data,axis=-1))
    #assert False
    #bt=ax.bar(index,data,bar_width,alpha=opacity,color=color_list[0])
    
    bt=ax.boxplot(x = data, # 指定绘图数据

            autorange=True,
 
            patch_artist=True, # 要求用自定义颜色填充盒形图，默认白色填充
 
            showmeans=False, # 以点的形式显示均值

            showfliers=False,

            meanline=False,

            labels=['DRS','AHF','ES-HGL','UES','QP','Kriging','RBF_MQ','RBF','SVR'],
 
            boxprops = {'color':'black','facecolor':'#9999ff'}, # 设置箱体属性，填充色和边框色
 
            flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # 设置异常值属性，点的形状、填充色和边框色
 
            meanprops = {'marker':'D','markerfacecolor':'indianred'}, # 设置均值点的属性，点的形状、填充色
 
            medianprops = {'linestyle':'--','color':'orange'}) # 设置中位数线的属性，线的类型和颜色

##    for b in bt:
##        h=b.get_height()
##        ax.text(b.get_x()+b.get_width()/2,h,format(h,'.4e'),ha='center',va='bottom')
##    plt.xticks(index,['DRS','AHF','ES_HGL','UES','QP','Kriging','RBF_MQ','RBF','SVR'])
##    ax.legend()
##    plt.xlabel(xname)
##    mpl.rcParams['font.size'] = 20.0
    plt.ylabel(yname)
##    plt.tick_params(labelsize=20)
##def draw_basebar(data):

def draw_fdvbar(data):
    errmean=np.mean(data,axis=0)
##    errmean=np.mean(errmean,axis=0)
    print('RMSE: ',np.mean(errmean[:,0],axis=0))
    print('NRMSE: ',np.mean(errmean[:,1],axis=0))

    ax=plt.subplot(121)
    single_bar(plt,ax,errmean[:,0],'operation','RMSE')
    ax=plt.subplot(122)
    single_bar(plt,ax,errmean[:,1],'operation',' NRMSE')
    plt.show()
    
def draw_valbar(data,timedata):
    funcnum,timenum,num,typenum=np.shape(data)
    errmean=np.mean(data[:,:,0],axis=1)#funcnum*typenum
    
    nerrmean=np.mean(data[:,:,1],axis=1)#funcnum*typenum

##    print(np.shape(data))
    
    errvar=np.var(data[:,:,0],axis=1)#funcnum*typenum
    timemean=np.mean(timedata,axis=1)

    print('Time: ',np.mean(timemean,axis=0))
    print('RMSE: ',np.mean(errmean,axis=0))
    print('NRMSE: ',np.mean(nerrmean,axis=0))
    print('RMSE var: ',np.mean(errvar,axis=0))

    hitmat=np.abs(data[:,:,:typenum]-np.min(data[:,:,:typenum],axis=-1,keepdims=True))/data[:,:,:typenum]<0.0001#funcnum*testnum*typenum
    hitmat=np.mean(hitmat,axis=1)#funcnum*typenum
    
##    plt.figure(figsize=(10, 10), dpi=80)
    ax=plt.subplot(221)
##    multi_bars(plt,ax,hitmat,'operation','mean error')
    single_bar(plt,ax,timemean,'operation','Prediction Time Cost')
    ax=plt.subplot(222)
    single_bar(plt,ax,errmean,'operation','RMSE')
    ax=plt.subplot(223)
    single_bar(plt,ax,nerrmean,'operation','NRMSE')
    ax=plt.subplot(224)
    single_bar(plt,ax,errvar,'operation','RMSE Variance')
    plt.show()
def draw_funcbar(data,timedata):
##    print(np.shape(data))
    barnum,funcnum,timenum,num,typenum=np.shape(data)
    typenum=typenum-1

    ndata=data[:,:,:,1,:]
    rdata=data[:,:,:,0,:]
    
    errmean=np.mean(rdata,axis=2)#barnum*funcnum*typenum
    errvar=np.var(rdata,axis=2)#barnum*funcnum*typenum
    nerrmean=np.mean(ndata[:,:,:],axis=2)#barnum*funcnum*typenum
    timemean=np.mean(timedata,axis=2)
    
##    hitmat=np.abs(data[:,:,:,:typenum]-np.min(data[:,:,:,:typenum],axis=-1,keepdims=True))/data[:,:,:,:typenum]<0.0001#barnum*funcnum*testnum*typenum
##    hitmat=np.mean(hitmat,axis=2)#barnum*funcnum*typenum

    ax=plt.subplot(221)
    funcs_multibars(plt,ax,timemean,'operation','Prediction Time Cost')
    ax=plt.subplot(222)
    funcs_multibars(plt,ax,errmean,'operation','RMSE')
    ax=plt.subplot(223)
    funcs_multibars(plt,ax,nerrmean,'operation','NRMSE')
    ax=plt.subplot(224)
    funcs_multibars(plt,ax,errvar,'operation','RMSE Variance')
    plt.show()
def draw_drs(data,timedata):
    funcnum,recordnum,typenum=np.shape(data)
    #errmean=np.mean(data,axis=0)#4*6
    errmean=data
##    hitmat=np.abs(data[:,:,:typenum]-np.min(data[:,:,:typenum],axis=-1,keepdims=True))/data[:,:,:typenum]<0.0001#funcnum*recordnum*typenum
##    hitmat=np.mean(hitmat[:,1,:],axis=0)
    errvar=errmean[:,2,:]
##    errvar=np.sqrt(errmean[:,2,:])
    timemean=errmean[:,3,:]
    mpl.rcParams['font.size'] = 20.0
    ax=plt.subplot(221)
    single_funcbar(plt,ax,timemean,'operation','Prediction Time Cost')
    ax=plt.subplot(222)
    single_funcbar(plt,ax,errmean[:,0,:],'operation','NRMSE')
    ax=plt.subplot(223)
    single_funcbar(plt,ax,errmean[:,1,:],'operation','RMSE')
    ax=plt.subplot(224)
    single_funcbar(plt,ax,errvar,'operation','RMSE Variance')
    plt.show()
def draw_func_drs(data,timedata):
    barnum,funcnum,timenum,typenum=np.shape(data)#4*27*4*6
    typenum=typenum-1
    errmean=data
##    print(np.shape(data))
##    assert False
##    errmean=np.mean(data,axis=1)#barnum*timenum*typenum

    ax=plt.subplot(221)
    drs_multibars(plt,ax,errmean[:,:,3,:],'operation','Prediction Time Cost')
    ax=plt.subplot(222)
    drs_multibars(plt,ax,errmean[:,:,0,:],'operation','NRMSE')
    ax=plt.subplot(223)
    drs_multibars(plt,ax,errmean[:,:,1,:],'operation','RMSE')
    ax=plt.subplot(224)
    drs_multibars(plt,ax,errmean[:,:,2,:],'operation','RMSE Variance')
    plt.show()
def draw_func_drs_hitmat(data,timedata):
    hitmat=np.abs(data-np.min(data,axis=-1,keepdims=True))/data<0.0001#4*27*4*6
##    hitmat=np.mean(hitmat,axis=1)#4*4*6
    ax=plt.subplot(221)
    drs_multibars(plt,ax,hitmat[:,:,3,:],'operation','Prediction Time Cost Hit Rate')
    ax=plt.subplot(222)
    drs_multibars(plt,ax,hitmat[:,:,0,:],'operation','NRMSE Hit Rate')
    ax=plt.subplot(223)
    drs_multibars(plt,ax,hitmat[:,:,1,:],'operation','RMSE Hit Rate')
    ax=plt.subplot(224)
    drs_multibars(plt,ax,hitmat[:,:,2,:],'operation','RMSE Variance Hit Rate')
    plt.show()
def draw_drs_hitmat(data,timedata):
    hitmat=np.abs(data-np.min(data,axis=-1,keepdims=True))/data<0.0001#funcnum*recordnum*typenum
    #hitmat=np.mean(hitmat,axis=0)#4*6
    
    ax=plt.subplot(221)
    single_funcbar(plt,ax,hitmat[:,3,:],'operation','Prediction Time Cost Hit Rate')
    ax=plt.subplot(222)
    single_funcbar(plt,ax,hitmat[:,0,:],'operation','NRMSE Hit Rate')
    ax=plt.subplot(223)
    single_funcbar(plt,ax,hitmat[:,1,:],'operation','RMSE Hit Rate')
    ax=plt.subplot(224)
    single_funcbar(plt,ax,hitmat[:,2,:],'operation','RMSE Variance Hit Rate')
    plt.show()
if __name__=='__main__':
##    sheet=xls2sheet('./results/fdv/result_all.xls',4)
##    sheet=xls2sheet('./results/rule/result.xls',2)
##    sheet=xls2sheet('./results/drs/result_drs_all.xls',0)
##    sheet=xls2sheet('./result.xls',0)
##    data,timedata=get_drs_data(sheet,typenum=9)

##    draw_drs(data,timedata)
##    data,timedata=get_multidrs_data(['./results/drs/result_drs_all.xls','./results/drs/result_drs_nokrg.xls','./results/drs/result_drs_nokrg_norbfmq.xls','./results/drs/result_drs_qp_svr.xls'])
##    print(np.shape(data),np.shape(timedata))
##    data,timedata=draw_func_drs(data[:,:10,:,:],timedata)
##    data,timedata=draw_func_drs_hitmat(data[:,:10,:,:],timedata)
    sheet=xls2sheet('result.xls',3)
    data,timedata=get_val_data(sheet)
    num0=0
    num=10
##    print(np.shape(data))
##    assert False
##    random.shuffle(data)
##    print(np.shape(data[num0:num]))
##    draw_drs(data[num0:num],timedata[num0:num])
##    draw_drs_hitmat(data[num0:num],timedata[num0:num])
    draw_valbar(data[num0:num],timedata[num0:num])

##    datalist,timelist=get_multival_data(['./results/fdv/result_all.xls','./results/fdv/result_nokrg.xls','./results/fdv/result_nokrg_norbfmq.xls','./results/fdv/result_qp_svr.xls'])
##    draw_funcbar(datalist[:,0:10,:,:],timelist[:,0:10,:,:])

##    sheet=xls2sheet('result.xls',4)
##    data=get_fdv_data(sheet,modelnum=5)
##    draw_fdvbar(data[0:10])

    
