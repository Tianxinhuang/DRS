

from .model import Model
from numpy import sqrt, array, fabs


class Bukin(Model):
    def __init__(self):
        """
        Bukin Function N. 6
        https://www.sfu.ca/~ssurjano/bukin6.html

        Global Optimization Test Functions Index. Retrieved June 2013,
        from http://infinity77.net/global_optimization/test_functions.html#test-functions-index
        """
        super().__init__(name='Bukin')
        self._n_dim = 2
        self._x_opt = [-10., 1.]
        self._f_opt = 0.0
        self._bounds = array([[-15., -5.], [-3., 3.]])

    def evaluate(self, X):
        t1 = 100.*sqrt(fabs(X[1] - 0.01*pow(X[0], 2.)))
        t2 = 0.01*fabs(X[0] + 10.0)
        return t1 + t2

