import numpy as np
import pandas as pd

from src.data_loader import sales_to_long, add_prices
from src.features import add_calendar_features, add_lag_features


def make_data(n_days=60):
    dates = pd.date_range("2020-01-01", periods=n_days)
    cal = pd.DataFrame({
        "d": [f"d_{i + 1}" for i in range(n_days)],
        "date": dates,
        "wm_yr_wk": [11000 + i // 7 for i in range(n_days)],
        "weekday": dates.day_name(),
        "event_name_1": [None] * n_days,
        "event_type_1": [None] * n_days,
        "snap_CA": 0, "snap_TX": 0, "snap_WI": 0,
    })
    rows = {
        "id": ["A_CA_1", "B_CA_1"],
        "item_id": ["A", "B"],
        "dept_id": ["D1", "D1"],
        "cat_id": ["C1", "C1"],
        "store_id": ["CA_1", "CA_1"],
        "state_id": ["CA", "CA"],
    }
    for i in range(n_days):
        rows[f"d_{i + 1}"] = [i, 2 * i]
    sales = pd.DataFrame(rows)
    prices = pd.DataFrame({
        "store_id": ["CA_1"] * 2 * 9,
        "item_id": ["A"] * 9 + ["B"] * 9,
        "wm_yr_wk": list(range(11000, 11009)) * 2,
        "sell_price": [1.0] * 9 + [2.0] * 9,
    })
    return sales, cal, prices


def test_long_format_and_prices():
    sales, cal, prices = make_data()
    long = sales_to_long(sales, cal)
    assert len(long) == 2 * 60
    long = add_prices(long, prices)
    assert long["sell_price"].notna().all()


def test_lag_features_match_history():
    sales, cal, prices = make_data()
    long = add_calendar_features(sales_to_long(sales, cal))
    feat = add_lag_features(long)
    a = feat[feat["id"] == "A_CA_1"].reset_index(drop=True)
    # sales for item A on day i equals i, so lag_28 at row 40 should be 12
    assert a.loc[40, "lag_28"] == 12
    assert np.isnan(a.loc[10, "lag_28"])
