import numpy as np
import pandas as pd

from fraud.data import make_synthetic
from fraud.features import FEATURE_COLUMNS, build_features, time_split


def test_synthetic_schema_and_fraud_rate():
    df = make_synthetic(rows=5000, fraud_rate=0.01, seed=1)
    assert list(df.columns) == ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
    assert df["Class"].sum() == 50


def test_build_features_columns_and_values():
    raw = pd.DataFrame({"Time": [0.0, 3600 * 3, 3600 * 30], "Amount": [0.0, 99.0, 10.0],
                        **{f"V{i}": [0.1, 0.2, 0.3] for i in range(1, 29)}})
    X = build_features(raw)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X["log_amount"].iloc[1] == np.log1p(99.0)
    assert X["hour_of_day"].tolist() == [0.0, 3.0, 6.0]  # 30h wraps to 6am
    assert X["is_night"].tolist() == [1.0, 1.0, 0.0]


def test_time_split_keeps_order_and_no_overlap():
    df = make_synthetic(rows=1000, fraud_rate=0.02, seed=2)
    train, test = time_split(df, 0.2)
    assert len(train) == 800 and len(test) == 200
    assert train["Time"].max() <= test["Time"].min()
