
from touchstone import Model
from numpy import array


class Rosenbrock(Model):
    def __init__(self, n_dim=2, bounds='small'):
        super().__init__(name='Rosenbrock')
        self._n_dim = n_dim
        self._x_opt = [1.0]*n_dim
        self._f_opt = 0.0

        if bounds == 'small':
            self._bounds = array([(-2.048, 2.048)]*n_dim)
        else:
            self._bounds = array([(-5.0, 10.0)]*n_dim)

    def evaluate(self, X):
        y = 0
        for i in range(self.n_dim - 1):
            xi = X[i]
            x_next = X[i+1]
            y += 100.0*(x_next - xi**2)**2 + (xi - 1)**2

        return y



