
from touchstone.models import Eggholder
from sklearn.utils.testing import assert_array_almost_equal, assert_almost_equal


def test_eggholder():
    X = [[-180.80862657, -256.07971953],
         [473.84817756, 131.00858577],
         [-423.68204422, 68.74104648],
         [-264.99057055, 196.69088878],
         [182.60743218, 455.50598237]]
    y_true = [-357.75400105, 294.59679077, -357.17976791, 77.97098250, 497.82607156]

    model = Eggholder()
    y = model(X)
    assert_array_almost_equal(y, y_true, 5)


def test_eggholder_optimum():
    model = Eggholder()
    f_opt = model(model.x_opt)
    assert_almost_equal(model.f_opt, f_opt, 4)
