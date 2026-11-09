from src.backtest import rolling_origin_splits


def test_splits_do_not_leak_and_have_horizon():
    splits = list(rolling_origin_splits(n_days=400, initial=300, horizon=28, step=28))
    assert len(splits) > 0
    for train_end, test_start, test_end in splits:
        assert train_end <= test_start
        assert test_end - test_start == 28
        assert test_end <= 400
