import numpy as np
import matplotlib.pyplot as plt

class RBF_MQ(object):
    def __init__(self,delta=5):
        self.delta=np.square(delta)
        self.name='RBF_MQ'
    def set_training_values(self,xt,yt,name=None):
        xt = np.atleast_2d(xt.T).T
        self.xt=xt
        self.yt=yt
        self.ptnum=np.shape(xt)[0]
        self.dimnum=np.shape(xt)[1]
    def train(self):
        a=np.expand_dims(self.xt,axis=1)-np.expand_dims(self.xt,axis=0)
        dis=np.sum(np.square(a),axis=-1)+self.delta
        A=1/np.sqrt(dis)
        A=np.array(A)
        self.beta=np.reshape(np.dot(np.linalg.pinv(A),self.yt),[self.ptnum,1])
    def predict_derivatives(self,x,kx):
        a=np.expand_dims(x,axis=1)-np.expand_dims(self.xt,axis=0)
        dis=np.sum(np.square(a),axis=-1)+self.delta
        base=1/((dis*np.sqrt(dis))+1e-5)#dnum*pnum
        base=np.expand_dims(base*np.transpose(self.beta),axis=-1)#dnum*pnum*1
        vec=a#dnum*pnum*dim
        result=np.sum(vec*base,axis=1)
        result=np.expand_dims(result[:,kx],axis=-1)
        return result
    def predict_values(self,x):
        x=np.atleast_2d(x.T).T
        a=np.expand_dims(x,axis=1)-np.expand_dims(self.xt,axis=0)
        dis=np.sum(np.square(a),axis=-1)+self.delta
        A=1/np.sqrt(dis)
        A=np.array(A)
        result=np.sum(A*np.transpose(self.beta),axis=-1,keepdims=True)#ptnum
        return result
if __name__=='__main__':
    sm=RBF_MQ(delta=1)
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
