from .model import Model
from numpy import array, exp, pi, cos,expand_dims
import numpy as np

class Langermann(Model):
    def __init__(self):
        """
        Langermann Function m=5
        https://arxiv.org/pdf/1308.4008v1.pdf
        """
        super().__init__(name='Langermann')
        self._n_dim = 2
        self._x_opt = [2.00299219, 1.006096]
        self._f_opt = -5.1621259
        self._bounds = array([[0.0, 10.0], [0.0, 10.0]])

        self.a = expand_dims(array([3, 5, 2, 1, 7]),axis=0)
        self.b = expand_dims(array([5, 2, 1, 4, 9]),axis=0)
        self.c = expand_dims(array([1, 2, 5, 2, 3]),axis=0)#1*5

    def evaluate(self, X):
        x1=expand_dims(X[0],axis=-1)
        x2=expand_dims(X[1],axis=-1)#n*1
        part1=exp((-1/pi)*((x1-self.a)**2+(x2-self.b)**2))#n*5
        part2=cos(pi*((x1-self.a)**2+(x2-self.b)**2))#n*5
        result=np.sum(self.c*part1*part2,axis=-1,keepdims=True)
        return result
##        return -sum(self.c*exp(-(1/pi)*(pow(X[0] - self.a, 2.) + pow(X[1] - self.b, 2.))) *
##                    cos(pi*(pow(X[0] - self.a, 2.) + pow(X[1] - self.b, 2.))))
