from smt.problems import *
from tf_util import *
from s_ahf import *
import xlwt
import xlrd
import xlutils.copy
import copy
def get_list(word,dimnum):
    result=[]
    words=word.split(' ')
    for i in range(dimnum):
        if len(words)<dimnum and len(words)==1:
            result.append(float(words[0]))
        else:
            result.append(float(words[i]))
    return result
def read_funcs(path):
    model_list=[]
    down_list=[]
    up_list=[]
    sample_list=[]
    words=[]
    name_list=[]
    f=open(path,'r')
    for line in f:
        line=line[:-1]
        if line[0]=='#':
            continue
        word=line.split(',')
        name_list.append(word[0])
        model_list.append(eval(word[0]))
        dimnum=int(word[-2])
        sample_num=int(word[-1])
        down_word=word[1]
        up_word=word[2]
        down_list.append(get_list(down_word,dimnum))
        up_list.append(get_list(up_word,dimnum))
        sample_list.append(sample_num)
    f.close()   
    return name_list,model_list,down_list,up_list,sample_list

def err_calculate(names,models,downs,ups,samplenum,testnum=100):
    modelnum=len(models)
    errs=[]
    wb,wss=openexcel('result.xls')
    ws,ws1,ws2,ws3=wss
    for i in range(modelnum):
        print('func',i,'name:',names[i])
        ws.write(1+i,0,names[i])
        model=models[i]
        dimnum=len(downs[i])
        if type(model)==type:
            sm=model(ndim=dimnum)
        else:
            sm=model
##        cal_models_R(downs[i],ups[i],samplenum[i],testnum,sm,ws,i+1,test_time=30)#choose when compare DRS
        compare_sampling(downs[i],ups[i],samplenum[i],testnum,sm,wss,i+1,test_time=20)#choose when compare validation
##        data1=latian1_sampling(downs[i],ups[i],samplenum[i])
##        label1=sm(data1)
##        data2=latian1_sampling(downs[i],ups[i],testnum)
##        label2=sm(data2)
    wb.save('result.xls')
    return errs
    
##
##def write_results()
def openexcel(path):
    if os.path.exists(path):
        os.remove(path)
    wb=xlwt.Workbook(encoding='ascii')
    ws=wb.add_sheet('result')
    ws1=wb.add_sheet('func_sample_compare_train')
    ws11=wb.add_sheet('func_sample_compare_pred')
    ws2=wb.add_sheet('model_val_err')
    ws3=wb.add_sheet('model_val_id')
    length=len(model_list)
    ws.write(0,0,'funcs')
    for i in range(length):
        ws.write(0,1+4*i,model_list[i].name+'_min')
        ws.write(0,2+4*i,model_list[i].name+'_mean')
        ws.write(0,3+4*i,model_list[i].name+'_var')
        ws.write(0,4+4*i,model_list[i].name+'_time')
    ws.write(0,1+4*length,'minerr')
    ws.write(0,2+4*length,'meanerr')
    ws.write(0,3+4*length,'minvar')
    ws.write(0,4+4*length,'minerrval')
    ws.write(0,5+4*length,'meanerrval')
    ws.write(0,6+4*length,'minvarval')
    return wb,[ws,[ws1,ws11],ws2,ws3]
if __name__=='__main__':
    names,models,down,up,sample=read_funcs('functions2.txt')
    err_calculate(names,models,down,up,sample,testnum=30)
