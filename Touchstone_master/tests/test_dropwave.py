
from touchstone.models import DropWave
from sklearn.utils.testing import assert_array_almost_equal, assert_almost_equal


def test_dropwave():
    X = [[3.82204597,-5.05954049],
         [1.54699996, 2.44769036],
         [-2.34032737, -0.55004107],
         [2.63638253, 4.40141853],
         [5.01036187, -3.03397548]]
    y_true = [-0.08006066, -0.00289002, -0.03286582, -0.08583045, -0.07241506]

    model = DropWave()
    y = model(X)
    assert_array_almost_equal(y, y_true, 5)


def test_drop_optimum():
    model = DropWave()
    f_opt = model(model.x_opt)
    assert_almost_equal(model.f_opt, f_opt, 5)
