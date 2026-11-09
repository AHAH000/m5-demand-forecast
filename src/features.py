"""Feature engineering for a global model (one model, many series).

Direct 28-day forecasting: every lag / rolling feature must be at least 28 days old
at prediction time, otherwise the training features leak information the model
will not have when forecasting.
"""
import pandas as pd

HORIZON = 28


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["dom"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["has_event"] = df["event_name_1"].notna().astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    target: str = "sales",
    lags=(28, 35, 42),
    windows=(7, 28),
) -> pd.DataFrame:
    """Expects df sorted by id, date. Lags are >= HORIZON to avoid leakage."""
    df = df.copy()
    g = df.groupby("id")[target]
    for k in lags:
        assert k >= HORIZON, "lag must be >= forecast horizon"
        df[f"lag_{k}"] = g.shift(k)
    for w in windows:
        df[f"roll_mean_{w}_lag{HORIZON}"] = g.transform(
            lambda s, w=w: s.shift(HORIZON).rolling(w).mean()
        )
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_rel"] = df["sell_price"] / df.groupby("id")["sell_price"].transform("mean")
    return df
