from touchstone.models import Langermann
from sklearn.utils.testing import assert_almost_equal


def test_langermann_optimum():
    model = Langermann()
    f_opt = model(model.x_opt)
    assert_almost_equal(model.f_opt, f_opt, 5)