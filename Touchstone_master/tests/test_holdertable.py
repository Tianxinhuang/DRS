from touchstone.models import HolderTable
from sklearn.utils.testing import assert_array_almost_equal, assert_almost_equal


def test_holdertable():
    X = [[9.61116027,2.84274116],
         [7.95862793, -1.15520798],
         [7.72684959, -6.73169297],
         [7.08365278, 3.08092919],
         [-9.77008520, -5.76036984]]
    y_true = [-1.58289043, -1.91048917, -8.58286036, -3.08102732, -3.98901641]

    model = HolderTable()
    y = model(X)
    assert_array_almost_equal(y, y_true, 5)


def test_holdertable_optimum():
    model = HolderTable()
    f_opt = model(model.x_opt)
    assert_almost_equal(model.f_opt, f_opt, 5)
