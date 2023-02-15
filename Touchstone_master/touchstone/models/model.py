
from abc import ABC, abstractmethod

import numpy as np
from sklearn.utils import check_array
from numpy import apply_along_axis


class Model(ABC):
    def __init__(self, name=" "):
        self._name = name
        self._bounds = None
        self._n_dim = None
        self._x_opt = None
        self._f_opt = None

    def __call__(self, X):
        try:
            x_ = check_array(X, ensure_min_features=self._n_dim)
        except ValueError as ex:
            # Could be a single value
            x = np.array(X)
            x = x.reshape(1, x.shape[0])
            x_ = check_array(x, ensure_min_features=self._n_dim, ensure_2d=True)

            # Make sure the dimensions match up
            if x_.shape[1] != self._n_dim:
                raise ValueError('Input shape ' + str(x_.shape[1]) + ' != to model dimensions ' + str(self._n_dim))

        # Check the bounds
        for point in x_:
            for value, bound in zip(point, self._bounds):
                if value < bound[0]:
                    raise ValueError('value ' + str(value) + ' is less than lower bound of ' + str(bound[0]))

                if value > bound[1]:
                    raise ValueError('value ' + str(value) + ' is greater than upper bound of ' + str(bound[1]))

        f = apply_along_axis(self.evaluate, 1, x_)
        return f

    @abstractmethod
    def evaluate(self, X):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def bounds(self):
        return self._bounds

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def x_opt(self):
        return self._x_opt

    @property
    def f_opt(self):
        return self._f_opt




