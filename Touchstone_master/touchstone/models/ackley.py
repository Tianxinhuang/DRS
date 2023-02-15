from .model import Model
from numpy import pi, cos, exp, sqrt, array


class Ackley(Model):
    def __init__(self, n_dim=2, bounds='small'):
        super().__init__(name='Ackley')
        self._n_dim = n_dim
        self._x_opt = [0.0]*n_dim
        self._f_opt = 0.0

        if bounds == 'small':
            self._bounds = array([(-8.192, 8.192)]*n_dim)
        else:
            self._bounds = array([(-32.768, 32.768)]*n_dim)

    def evaluate(self, X):
        a, b, c = 20, 0.2, 2*pi
        sum1, sum2 = 0.0, 0.0

        for i in range(self._n_dim):
            xi = X[i]
            sum1 += xi**2
            sum2 += cos(c*xi)

        term1 = -a*exp(-b*sqrt(sum1/self.n_dim))
        term2 = -exp(sum2/self.n_dim)

        return term1 + term2 + a + exp(1)
