"""Stage 2: feature engineering and a time-based train/test split.

Writes the train and test sets, plus a Feast offline source with an entity key
(transaction_id) and event timestamp for every transaction.
"""
import numpy as np
import pandas as pd

from fraud.config import (FEAST_SOURCE_PATH, PCA_COLUMNS, RAW_PATH, TARGET, TEST_PATH,
                          TRAIN_PATH, load_params)

FEATURE_COLUMNS = [*PCA_COLUMNS, "log_amount", "hour_of_day", "is_night"]
DATASET_START = pd.Timestamp("2013-09-01", tz="UTC")  # the ULB data covers two days in September 2013


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pure function used by training, serving and tests, so features never drift apart."""
    out = df[PCA_COLUMNS].astype("float64").copy()
    out["log_amount"] = np.log1p(df["Amount"].astype("float64"))
    hour = (df["Time"].astype("float64") // 3600) % 24
    out["hour_of_day"] = hour
    out["is_night"] = hour.between(0, 5).astype("float64")
    return out[FEATURE_COLUMNS]


def time_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier transactions and test on later ones, as in production."""
    df = df.sort_values("Time").reset_index(drop=True)
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut], df.iloc[cut:]


def main() -> None:
    params = load_params()
    raw = pd.read_parquet(RAW_PATH)

    feats = build_features(raw)
    feats[TARGET] = raw[TARGET].values
    feats["Time"] = raw["Time"].values
    feats["transaction_id"] = np.arange(len(raw), dtype="int64")
    feats["event_timestamp"] = DATASET_START + pd.to_timedelta(raw["Time"], unit="s")

    train, test = time_split(feats, params["split"]["test_fraction"])
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)

    FEAST_SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats[["transaction_id", "event_timestamp", *FEATURE_COLUMNS]].to_parquet(FEAST_SOURCE_PATH, index=False)

    print(f"Train: {len(train):,} rows, {int(train[TARGET].sum())} fraud | "
          f"Test: {len(test):,} rows, {int(test[TARGET].sum())} fraud")


if __name__ == "__main__":
    main()
