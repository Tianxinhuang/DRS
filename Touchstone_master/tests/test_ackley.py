
from touchstone.models import Ackley
from sklearn.utils.testing import assert_array_almost_equal, assert_almost_equal


def test_ackley():
    X = [[-1.890599, 8.100709,1.998216],
         [7.573548, 3.594374, -3.194529],
         [-0.345594, 3.899347, 6.406367],
         [-7.458480, 2.093705, -5.130343],
         [1.463660, -7.304065, 6.515623]]

    y_true = [12.908202, 14.989536, 13.492035, 14.677363, 15.875441]

    model = Ackley(n_dim=3, bounds='small')
    y = model(X)

    assert_array_almost_equal(y_true, y, 4)


def test_ackley_optimum():
    model = Ackley(n_dim=3, bounds='small')
    f_opt = model(model.x_opt)

    assert_almost_equal(model.f_opt, f_opt)
