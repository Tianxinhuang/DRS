from .model import Model
from numpy import array, cos, sqrt

class DropWave(Model):
    def __init__(self):
        """
        Bukin Function N. 6
        https://www.sfu.ca/~ssurjano/bukin6.html

        Global Optimization Test Functions Index. Retrieved June 2013,
        from http://infinity77.net/global_optimization/test_functions.html#test-functions-index
        """
        super().__init__(name='Drop Wave')
        self._n_dim = 2
        self._x_opt = [[0.0, 0.0]]
        self._f_opt = -1.0
        self._bounds = array([[-5.12, 5.12], [-5.12, 5.12]])

    def evaluate(self, X):
        f1 = 1. + cos(12. * sqrt(pow(X[0], 2.) + pow(X[1], 2.)))
        f2 = 0.5*(pow(X[0], 2.) + pow(X[1], 2.)) + 2.
        return -f1/f2
