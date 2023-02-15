from .model import Model
from numpy import array, sin, sqrt, fabs


class Eggholder(Model):
    def __init__(self):
        """
        Egg Holder function
        """
        super().__init__(name='Egg Holder')
        self._n_dim = 2
        self._x_opt = [512.0, 404.2319]
        self._f_opt = -959.6407
        self._bounds = array([[-512., 512.], [-512., 512.]])

    def evaluate(self, X):
        t1 = -(X[1] + 47.)*sin(sqrt(fabs(X[1]+X[0]/2. + 47.)))
        t2 = -X[0]*sin(sqrt(fabs(X[0]-(X[1] + 47.))))
        return t1 + t2
