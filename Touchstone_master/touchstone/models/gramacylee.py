
from .model import Model
from numpy import array, sin, pi, power


class GramacyLee(Model):
    def __init__(self):
        super().__init__(name='Gramacy Lee')
        self._n_dim = 1
        self._x_opt = [[0.54856344]]
        self._f_opt = -0.86901113
        self._bounds = array([[0.5, 2.5]])

    def evaluate(self, X):
        y = sin(10*pi*X) / (2*X) + power(X - 1., 4.)
        return y
