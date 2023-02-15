
from .model import Model
from numpy import array


class Booth(Model):
    def __init__(self):
        super().__init__(name='Booth')
        self._n_dim = 2
        self._x_opt = [1.0, 3.0]
        self._f_opt = 0.0
        self._bounds = array([(-10.0, 10.0), (-10.0, 10.0)])

    def evaluate(self, X):
        y = (X[0] + 2*X[1] - 7)**2 + (2*X[0] + X[1] - 5)**2
        return y
