import numpy as np
import pandas as pd
import pytest

from src.data.prepare import clean
from src.data.sample import generate

SMALL = {"seed": 1, "n_stores": 2, "n_items": 3, "start_date": "2024-01-01", "n_days": 120}


def test_generate_is_deterministic():
    pd.testing.assert_frame_equal(generate(SMALL), generate(SMALL))


def test_clean_leaves_one_valid_row_per_series_and_day():
    df = clean(generate(SMALL))
    assert len(df) == 2 * 3 * 120
    assert not df.duplicated(["store", "item", "date"]).any()
    assert df["sales"].notna().all() and (df["sales"] >= 0).all()
    assert df["sales"].dtype.kind == "i"


def test_clean_imputes_missing_and_negative_sales_from_previous_day():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "store": 1,
            "item": 1,
            "sales": [10.0, np.nan, -3.0, 12.0],
        }
    )
    assert clean(raw)["sales"].tolist() == [10, 10, 10, 12]


def test_clean_rejects_a_file_without_the_expected_columns():
    with pytest.raises(ValueError, match="missing columns"):
        clean(pd.DataFrame({"date": ["2024-01-01"], "sales": [1]}))
