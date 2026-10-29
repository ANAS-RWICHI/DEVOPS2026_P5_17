"""Stage `prepare`: clean the raw Kaggle sales file (date, store, item, sales)."""

from __future__ import annotations

import pandas as pd

from src.config import CLEAN_CSV, RAW_CSV

KEYS = ["store", "item", "date"]
REQUIRED = ["date", "store", "item", "sales"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return one valid row per (store, item, date), sorted.

    Missing or negative sales are imputed from the previous day of the same
    series instead of being dropped, so that each series stays a gap-free
    daily sequence (the lag features rely on that).
    """
    missing_cols = set(REQUIRED) - set(df.columns)
    if missing_cols:
        raise ValueError(f"missing columns: {sorted(missing_cols)}")

    df = df[REQUIRED].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=KEYS).sort_values(KEYS).reset_index(drop=True)

    df.loc[df["sales"] < 0, "sales"] = float("nan")
    by_series = df.groupby(["store", "item"])["sales"]
    df["sales"] = by_series.ffill().fillna(by_series.bfill())
    df = df.dropna(subset=["sales"])

    df["sales"] = df["sales"].round().astype(int)
    return df.reset_index(drop=True)


def main() -> None:
    raw = pd.read_csv(RAW_CSV)
    df = clean(raw)
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False, lineterminator="\n")
    print(f"prepare: {len(raw)} raw rows -> {len(df)} clean rows")


if __name__ == "__main__":
    main()
