"""Stage `featurize`: lag, rolling and calendar features, then a time-based split.

The model forecasts HORIZON days ahead, so a feature for day d may only use
sales up to day d - HORIZON. The same rule is applied by the API through
`features_from_history`, which keeps training and serving consistent.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.config import CLEAN_CSV, TARGET, TEST_CSV, TRAIN_CSV, load_params

HORIZON = 7
BASE_FEATURES = ["store", "item", "dayofweek", "month", "dayofyear"]


def feature_columns(lags: Sequence[int], windows: Sequence[int]) -> list[str]:
    return BASE_FEATURES + [f"lag_{k}" for k in lags] + [f"rmean_{w}" for w in windows]


def _check(lags: Sequence[int]) -> None:
    if min(lags) < HORIZON:
        raise ValueError(f"lags must be >= forecast horizon ({HORIZON}), got {list(lags)}")


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"])
    df["dayofweek"] = dates.dt.dayofweek
    df["month"] = dates.dt.month
    df["dayofyear"] = dates.dt.dayofyear
    return df


def add_features(df: pd.DataFrame, lags: Sequence[int], windows: Sequence[int]) -> pd.DataFrame:
    """Add all model features; rows without enough history are dropped."""
    _check(lags)
    df = df.sort_values(["store", "item", "date"]).copy()
    series = df.groupby(["store", "item"])[TARGET]
    for k in lags:
        df[f"lag_{k}"] = series.shift(k)
    for w in windows:
        df[f"rmean_{w}"] = series.transform(lambda s, w=w: s.shift(HORIZON).rolling(w).mean())
    df = add_calendar(df)
    return df.dropna(subset=feature_columns(lags, windows)).reset_index(drop=True)


def history_length(lags: Sequence[int], windows: Sequence[int]) -> int:
    """Number of past daily sales the API needs to build the features."""
    return max(max(lags) - HORIZON + 1, max(windows))


def features_from_history(
    history: Sequence[float], lags: Sequence[int], windows: Sequence[int]
) -> dict[str, float]:
    """Lag and rolling features from the most recent daily sales.

    `history` is ordered oldest to newest and ends HORIZON days before the
    date being forecast (i.e. it ends today when forecasting next week).
    """
    _check(lags)
    need = history_length(lags, windows)
    if len(history) < need:
        raise ValueError(f"history needs at least {need} values, got {len(history)}")
    out = {f"lag_{k}": float(history[-(k - HORIZON + 1)]) for k in lags}
    for w in windows:
        out[f"rmean_{w}"] = float(sum(history[-w:]) / w)
    return out


def split_by_time(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last `test_days` days: no shuffling for a forecasting task."""
    dates = pd.to_datetime(df["date"])
    cutoff = dates.max() - pd.Timedelta(days=test_days)
    return df[dates <= cutoff], df[dates > cutoff]


def main() -> None:
    p = load_params()["features"]
    df = add_features(pd.read_csv(CLEAN_CSV), p["lags"], p["rolling_windows"])
    train, test = split_by_time(df, p["test_days"])
    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False, lineterminator="\n")
    test.to_csv(TEST_CSV, index=False, lineterminator="\n")
    print(f"featurize: train={len(train)} rows, test={len(test)} rows")


if __name__ == "__main__":
    main()
