"""Stage 1: load raw transactions.

Real data: the ULB credit card fraud dataset (Kaggle "creditcardfraud"), fetched from
its OpenML mirror. 284,807 European card transactions over two days in 2013, 492 frauds.
Features V1 to V28 are PCA components, so the raw card details stay private.
"""
import argparse

import numpy as np
import pandas as pd

from fraud.config import PCA_COLUMNS, RAW_COLUMNS, RAW_PATH, TARGET, load_params

OPENML_ID = 1597


def load_openml() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto")
    df = bunch.frame.copy()
    df[TARGET] = df[TARGET].astype(int)
    return df[RAW_COLUMNS]


def make_synthetic(rows: int, fraud_rate: float, seed: int) -> pd.DataFrame:
    """Generate data with the real schema. Fraud rows are shifted on a few PCA
    components and have different amounts, so models have a signal to learn."""
    rng = np.random.default_rng(seed)
    n_fraud = max(int(rows * fraud_rate), 10)
    y = np.zeros(rows, dtype=int)
    y[rng.choice(rows, n_fraud, replace=False)] = 1

    X = rng.normal(0, 1, size=(rows, len(PCA_COLUMNS)))
    shift = np.zeros(len(PCA_COLUMNS))
    shift[[2, 9, 11, 13, 16]] = [-3.0, -2.5, -3.0, -4.0, -3.5]
    X[y == 1] += shift + rng.normal(0, 1.0, size=(n_fraud, len(PCA_COLUMNS)))

    amount = np.round(rng.lognormal(3.5, 1.2, rows), 2)
    amount[y == 1] = np.round(rng.lognormal(4.5, 1.5, n_fraud), 2)
    time = np.sort(rng.uniform(0, 172_800, rows))  # two days, in seconds

    df = pd.DataFrame(X, columns=PCA_COLUMNS)
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df[TARGET] = y
    return df[RAW_COLUMNS]


def main(source: str | None = None) -> pd.DataFrame:
    params = load_params()["data"]
    source = source or params["source"]
    if source == "openml":
        df = load_openml()
    elif source == "synthetic":
        df = make_synthetic(params["synthetic_rows"], params["synthetic_fraud_rate"], params["seed"])
    else:
        raise ValueError(f"Unknown data source: {source}")

    df = df.sort_values("Time").reset_index(drop=True)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_PATH, index=False)
    print(f"Saved {len(df):,} transactions ({df[TARGET].sum():,} fraud) to {RAW_PATH}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["openml", "synthetic"])
    main(parser.parse_args().source)
