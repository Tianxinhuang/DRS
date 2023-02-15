
from touchstone import Model
from numpy import array
from sklearn.utils.testing import assert_equal, assert_array_almost_equal, assert_raises


class MockModel(Model):
    def __init__(self):
        super().__init__(name='Mock')
        self._n_dim = 2
        self._x_opt = [1.0, 1.0]
        self._f_opt = 0.0
        self._bounds = array([[0.0, 1.0], [1.0, 5.0]])

    def evaluate(self, X):
        return X[0] + X[1]


def test_get_name():
    model = MockModel()
    assert_equal(model.name, 'Mock')


def test_evaluate_model_single_point():
    X = [1.0, 2.0]

    model = MockModel()
    y = model.evaluate(X)
    assert_equal(y, 3.0)


def test_evaluate_model_multiple_points():
    X = [[1.0, 2.0], [0.5, 4.0]]

    model = MockModel()
    y = model(X)
    assert_array_almost_equal(y, [3.0, 4.5])


def test_check_invalid_input():
    def raises_value_error():
        X = [1.0, 2.0, 3.0]

        model = MockModel()
        y = model(X)

    assert_raises(ValueError, raises_value_error)


def test_check_out_of_bounds():
    def lower_bound_violation():
        X = [[-1.0, 1.0]]
        model = MockModel()
        y = model(X)

    def upper_bound_violation():
        X = [[1.0, 6.0]]
        model = MockModel()
        y = model(X)

    assert_raises(ValueError, lower_bound_violation)
    assert_raises(ValueError, upper_bound_violation)









