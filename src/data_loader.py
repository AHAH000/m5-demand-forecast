"""Load the M5 files and build a long-format modeling table for a subset."""
import pandas as pd

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CAL_COLS = [
    "d", "date", "wm_yr_wk", "weekday",
    "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI",
]


def load_calendar(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_prices(path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_sales_wide(path, store_ids=None, cat_ids=None) -> pd.DataFrame:
    """Read the wide sales file and keep only the requested stores / categories."""
    df = pd.read_csv(path)
    if store_ids:
        df = df[df["store_id"].isin(store_ids)]
    if cat_ids:
        df = df[df["cat_id"].isin(cat_ids)]
    return df.reset_index(drop=True)


def sales_to_long(sales_wide: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Melt d_1..d_N columns to rows and attach calendar info. Sorted by id, date."""
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    id_cols = [c for c in ID_COLS if c in sales_wide.columns]
    long = sales_wide.melt(
        id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales"
    )
    cal = calendar[[c for c in CAL_COLS if c in calendar.columns]]
    long = long.merge(cal, on="d", how="left")
    return long.sort_values(["id", "date"]).reset_index(drop=True)


def add_prices(long: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach weekly sell prices. Filters the prices table to the subset first (memory)."""
    prices = prices[
        prices["store_id"].isin(long["store_id"].unique())
        & prices["item_id"].isin(long["item_id"].unique())
    ]
    return long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
