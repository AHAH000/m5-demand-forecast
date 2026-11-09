"""Forecast accuracy metrics (M5 style)."""
import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def bias(y_true, y_pred):
    """Mean error. Positive means over-forecasting."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def rmsse(y_true, y_pred, y_train):
    """Root mean squared scaled error, as used in M5.

    Scale = mean squared one-step naive error on the training data, starting from the
    first non-zero sale. Below 1 beats the one-step naive forecast.
    """
    y_train = np.asarray(y_train, dtype=float)
    nz = np.flatnonzero(y_train)
    if len(nz) == 0:
        return float("nan")
    y_train = y_train[nz[0]:]
    scale = np.mean(np.diff(y_train) ** 2) if len(y_train) > 1 else 0.0
    if scale == 0:
        return float("nan")
    err = np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)
    return float(np.sqrt(err / scale))
