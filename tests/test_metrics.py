import numpy as np
from src.metrics import mae, bias, rmsse


def test_perfect_forecast_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0
    assert rmsse(y, y, np.arange(30)) == 0.0


def test_bias_sign():
    assert bias([1, 1], [2, 2]) > 0
