import numpy as np
from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

class svr(object):
    def __init__(self,kernel='rbf'):
        self.sm=SVR(kernel=kernel)
##        self.sm=GridSearchCV(SVR(kernel=kernel, gamma=0.1), cv=5,\
##                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],\
##                               "gamma": np.logspace(-2, 2, 5)})
        self.name='SVR'
    def set_training_values(self,xt,yt,name=None):
        xt = np.atleast_2d(xt.T).T
        self.xt=xt
        self.yt=np.reshape(yt,[-1])
        self.ptnum=np.shape(xt)[0]
        self.dimnum=np.shape(xt)[1]
    def train(self):
        self.sm.fit(self.xt,self.yt)
    def predict_derivatives(self,x,kx):
        tiny_dis=0.0001
        vec=np.zeros_like(x)
        vec[:,kx]=tiny_dis
        newx=x+vec
        result=(self.sm.predict(newx)-self.sm.predict(x))/tiny_dis
        result=np.reshape(result,[-1,1])
        return result
    def predict_values(self,x):
        x=np.atleast_2d(x.T).T
        result=self.sm.predict(x)#ptnum
        result=np.reshape(result,[-1,1])
##        print(np.shape(result))
        return result
if __name__=='__main__':
    sm=svr(kernel='rbf')
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
