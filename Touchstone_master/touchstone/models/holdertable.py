from .model import Model
from numpy import array, sin, exp, fabs, sqrt, pi, cos


class HolderTable(Model):
    def __init__(self):
        """
        Holder Table Function
        """
        super().__init__(name='Holder Table')
        self._n_dim = 2
        self._x_opt = [[8.05502, 9.66459], [8.05502, -9.66459], [-8.05502, 9.66459], [-8.05502, -9.66459]]
        self._f_opt = -19.2085
        self._bounds = array([[-10., 10.], [-10., 10.]])

    def evaluate(self, X):
        f1 = sin(X[0])*cos(X[1])
        f2 = exp(fabs(1. - sqrt(pow(X[0], 2.) + pow(X[1], 2.))/pi))
        return -fabs(f1*f2)


