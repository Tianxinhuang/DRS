import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

class ANN(object):
    def __init__(self,delta=5):
        self.delta=np.square(delta)
        self.name='ANN'
    def set_training_values(self,xt,yt,name=None):
        xt = np.atleast_2d(xt.T).T
##        self.xt=xt
        
##        self.ynorm=np.linalg.norm(yt)
        self.cen=(np.max(xt,axis=0)+np.min(xt,axis=0))/2
        self.len=(np.max(xt,axis=0)-np.min(xt,axis=0))/2
        self.xt=(xt-self.cen)/self.len
        
        self.minv=np.min(yt)
        self.maxv=np.max(yt)
        self.yt=yt
        self.yt=(yt-self.minv)/(self.maxv-self.minv)
##        print(self.yt,(self.maxv-self.minv))
##        print(yt,self.yt)
        self.yt=np.atleast_2d(yt.T).T
        self.ptnum=np.shape(xt)[0]
        self.dimnum=np.shape(xt)[1]
    def train(self):
        minvalue=np.min(self.xt,axis=0,keepdims=True)
        maxvalue=np.max(self.xt,axis=0,keepdims=True)
        values=np.concatenate([minvalue,maxvalue],axis=0)
        net = nl.net.newff(values.T,[64,64,1])
##        net = nl.net.newff(values.T,[64,64,64,64,64,64,64,1])
        net.trainf = nl.train.train_gd
        net.train(self.xt,self.yt,epochs=500,show=600,goal=0.01,lr=0.001)
        self.net=net
##    def predict_derivatives(self,x,kx):
##        a=np.expand_dims(x,axis=1)-np.expand_dims(self.xt,axis=0)
##        dis=np.sum(np.square(a),axis=-1)+self.delta
##        base=1/((dis*np.sqrt(dis))+1e-5)#dnum*pnum
##        base=np.expand_dims(base*np.transpose(self.beta),axis=-1)#dnum*pnum*1
##        vec=a#dnum*pnum*dim
##        result=np.sum(vec*base,axis=1)
##        result=np.expand_dims(result[:,kx],axis=-1)
##        return result
    def predict_values(self,x):
        x=np.atleast_2d(x.T).T
        x=(x-self.cen)/self.len
        result=self.net.sim(x)
##        print((self.maxv-self.minv)*self.net.sim(self.xt)+self.minv)
        result=(self.maxv-self.minv)*result+self.minv
##        result=self.len*result+self.cen
##        result=np.squeeze(result,-1)
        result=np.reshape(result,[-1,1])
        return result
if __name__=='__main__':
    sm=ANN(delta=1)
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
