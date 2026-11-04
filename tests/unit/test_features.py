import pandas as pd
import pytest

from src.features import (
    HORIZON,
    add_features,
    feature_columns,
    features_from_history,
    history_length,
    split_by_time,
)

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


def test_api_features_match_training_features(series):
    """Training/serving consistency: same history -> same features."""
    trained = add_features(series, LAGS, WINDOWS).iloc[-1]
    history = series["sales"].tolist()[: 80 - HORIZON]  # ends 7 days before day 79
    served = features_from_history(history, LAGS, WINDOWS)
    for name, value in served.items():
        assert value == pytest.approx(trained[name]), name


def test_history_too_short_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        features_from_history([1.0] * (history_length(LAGS, WINDOWS) - 1), LAGS, WINDOWS)


def test_lag_shorter_than_horizon_is_rejected(series):
    with pytest.raises(ValueError, match="horizon"):
        add_features(series, [1], WINDOWS)


def test_split_is_chronological(series):
    train, test = split_by_time(series, test_days=10)
    assert train["date"].max() < test["date"].min()
    assert test["date"].nunique() == 10
