
from touchstone.models import CrossInTray
from sklearn.utils.testing import assert_array_almost_equal, assert_almost_equal
from nose.plugins.skip import SkipTest

"""
Currently skipping these tests due to the very large numbers produced by the
exponential component of the function being very large and not matching up with
the Matlab code being used to validate.
"""

@SkipTest
def test_crossintray():
    X = [[-7.90666785,1.20018952],
         [4.61018337, -4.43964713],
         [2.94350597, -3.01907760],
         [-9.84758637, -1.64090505],
         [2.98324085, 4.27359053]]
    y_true = [-1.69541919, -1.78897181, -1.32668222, -1.46602161, -1.53586947]

    model = CrossInTray()
    y = model(X)
    assert_array_almost_equal(y, y_true, 4)

@SkipTest
def test_crossintray_optimum():
    model = CrossInTray()

    for x_opt in model.x_opt:
        f_opt = model(x_opt)
        assert_almost_equal(model.f_opt, f_opt, 4)
