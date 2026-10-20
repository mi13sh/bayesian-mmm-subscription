import numpy as np
import pytest

from mmm.transforms import geometric_adstock, hill_saturation


def test_adstock_preserves_total_effect():
    x = np.zeros(30)
    x[0] = 100
    out = geometric_adstock(x, decay=0.6, max_lag=12)
    assert out.sum() == pytest.approx(100)


def test_adstock_zero_decay_is_identity():
    x = np.array([1.0, 5.0, 2.0])
    assert np.allclose(geometric_adstock(x, decay=0.0), x)


def test_adstock_carries_spend_forward():
    x = np.array([10.0, 0, 0, 0])
    out = geometric_adstock(x, decay=0.5)
    assert out[1] > 0 and out[1] < out[0]


def test_adstock_rejects_bad_decay():
    with pytest.raises(ValueError):
        geometric_adstock(np.ones(3), decay=1.0)


def test_hill_is_half_at_half_sat():
    assert hill_saturation(np.array([8.0]), half_sat=8.0, slope=1.7)[0] == pytest.approx(0.5)


def test_hill_is_bounded_and_increasing():
    y = hill_saturation(np.linspace(0, 100, 50), half_sat=10, slope=1.5)
    assert y[0] == 0 and y.max() < 1
    assert np.all(np.diff(y) > 0)
