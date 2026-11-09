"""Rolling-origin (expanding window) backtesting over days."""


def rolling_origin_splits(n_days: int, initial: int, horizon: int = 28, step: int = 28):
    """Yield (train_end, test_start, test_end) as day positions (end exclusive).

    n_days: total number of days
    initial: size of the first training window
    horizon: forecast length (M5 uses 28)
    step: how far the origin moves each iteration
    """
    origin = initial
    while origin + horizon <= n_days:
        yield origin, origin, origin + horizon
        origin += step
