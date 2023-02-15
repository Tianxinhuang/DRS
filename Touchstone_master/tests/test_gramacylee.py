
from touchstone.models import GramacyLee
from sklearn.utils.testing import assert_array_almost_equal, assert_almost_equal


def test_gramacylee():
    X = [[2.00940742], [1.48521277], [0.5057841] , [1.98950074], [2.14005096]]
    y_true = [[1.11063789], [0.20625639], [-0.11899009], [0.87725852], [1.46694269]]

    model = GramacyLee()
    y = model(X)
    assert_array_almost_equal(y, y_true, 5)


def test_gramacylee_optimum():
    model = GramacyLee()
    f_opt = model(model.x_opt)
    assert_almost_equal(model.f_opt, f_opt, 4)