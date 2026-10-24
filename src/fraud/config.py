"""Shared paths and parameter loading."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "transactions.parquet"
TRAIN_PATH = DATA_DIR / "features" / "train.parquet"
TEST_PATH = DATA_DIR / "features" / "test.parquet"
FEAST_SOURCE_PATH = ROOT / "feature_repo" / "data" / "transaction_features.parquet"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_DIR = ROOT / "metrics"
REPORTS_DIR = ROOT / "reports"

TARGET = "Class"
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_COLUMNS = ["Time", *PCA_COLUMNS, "Amount", TARGET]


def load_params(path: Path = ROOT / "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
