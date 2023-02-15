from .model import Model
from numpy import array, sin, exp, fabs, sqrt, pi


class CrossInTray(Model):
    def __init__(self):
        """
        Cross-In-Tray function
        """
        super().__init__(name='Cross In Tray')
        self._n_dim = 2
        self._x_opt = [[1.3491, -1.3491], [1.3491, 1.3491], [-1.3491, 1.3491], [-1.3491, -1.3491]]
        self._f_opt = -2.06261
        self._bounds = array([[-10., 10.], [-10., 10.]])

    def evaluate(self, X):
        f1 = sin(X[0])*sin(X[1])
        f2 = exp(fabs(100. - sqrt(pow(X[0], 2.) + pow(X[1], 2.))/pi))
        return -0.0001*pow(fabs(f1 + f2) + 1., 0.1)



