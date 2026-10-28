"""Synthetic stand-in with the same columns as the Kaggle file.

Used by the unit tests, and to try the pipeline before the real data is
available:  python -m src.data.sample
It is NOT the project dataset; the real one is data/raw/train.csv from Kaggle
("Store Item Demand Forecasting Challenge"), tracked by DVC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RAW_CSV

DEFAULT = {"seed": 42, "n_stores": 5, "n_items": 20, "start_date": "2013-01-01", "n_days": 730}


def generate(p: dict) -> pd.DataFrame:
    rng = np.random.default_rng(p["seed"])
    n_days = p["n_days"]
    dates = pd.date_range(p["start_date"], periods=n_days, freq="D")

    store_factor = rng.uniform(0.7, 1.3, p["n_stores"])
    item_base = rng.uniform(15, 60, p["n_items"])

    dow = dates.dayofweek.to_numpy()
    weekly = 1 + 0.25 * np.isin(dow, [5, 6]) - 0.05 * (dow == 0)
    yearly = 1 + 0.2 * np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 80) / 365.25)
    trend = 1 + 0.0003 * np.arange(n_days)

    frames = []
    for s in range(p["n_stores"]):
        for i in range(p["n_items"]):
            mean = store_factor[s] * item_base[i] * weekly * yearly * trend
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates.strftime("%Y-%m-%d"),
                        "store": s + 1,
                        "item": i + 1,
                        "sales": rng.poisson(mean).astype(float),
                    }
                )
            )
    df = pd.concat(frames, ignore_index=True)

    # Defects the `prepare` stage must handle: missing values and duplicates.
    missing = rng.choice(len(df), size=len(df) // 1000, replace=False)
    df.loc[missing, "sales"] = np.nan
    duplicates = df.sample(frac=0.002, random_state=p["seed"])
    return pd.concat([df, duplicates], ignore_index=True)


def main() -> None:
    if RAW_CSV.exists():
        raise SystemExit(f"{RAW_CSV} already exists; refusing to overwrite it with sample data")
    df = generate(DEFAULT)
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False, lineterminator="\n")
    print(f"sample: {len(df)} synthetic rows -> {RAW_CSV}")


if __name__ == "__main__":
    main()
