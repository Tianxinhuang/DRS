
from touchstone.models import Bukin
from sklearn.utils.testing import assert_array_almost_equal, assert_almost_equal


def test_bukin():
    X = [[-11.715702, -0.475302],
         [-9.521474, -2.811872],
         [-13.429017, -1.963331],
         [-8.279647, -2.024629],
         [-13.868720, 1.891353]]

    y_true = [135.953854, 192.837790, 194.114574, 164.642683, 17.944285]

    model = Bukin()
    y = model(X)
    assert_array_almost_equal(y_true, y, 4)


def test_bukin_optimum():
    model = Bukin()
    f_opt = model(model.x_opt)
    assert_almost_equal(model.f_opt, f_opt)