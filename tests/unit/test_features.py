import pandas as pd
import pytest

from src.features import add_features, feature_columns, split_by_time

LAGS, WINDOWS = [7, 14, 28], [7, 28]


@pytest.fixture
def series() -> pd.DataFrame:
    n = 80
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
            "store": 1,
            "item": 1,
            "sales": range(n),  # sales on day index d is d: easy to check lags
        }
    )


def test_features_only_use_sales_older_than_the_horizon(series):
    out = add_features(series, LAGS, WINDOWS)
    row = out.iloc[-1]  # day index 79
    assert row["lag_7"] == 72 and row["lag_14"] == 65 and row["lag_28"] == 51
    assert row["rmean_7"] == pytest.approx(sum(range(66, 73)) / 7)
    assert not out[feature_columns(LAGS, WINDOWS)].isna().any().any()


def test_series_do_not_leak_into_each_other(series):
    other = series.assign(item=2, sales=1000)
    out = add_features(pd.concat([series, other]), LAGS, WINDOWS)
    assert (out.loc[out["item"] == 1, "lag_7"] < 1000).all()


def test_lag_shorter_than_horizon_is_rejected(series):
    with pytest.raises(ValueError, match="horizon"):
        add_features(series, [1], WINDOWS)


def test_split_is_chronological(series):
    train, test = split_by_time(series, test_days=10)
    assert train["date"].max() < test["date"].min()
    assert test["date"].nunique() == 10
