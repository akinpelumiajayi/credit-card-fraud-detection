"""Model bundle helpers shared by training, evaluation and the BentoML service."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud.config import MODEL_PATH
from fraud.features import FEATURE_COLUMNS, build_features


@dataclass
class ModelBundle:
    name: str
    model: object
    threshold: float
    feature_columns: list

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Higher score means more likely fraud, for both model types."""
        X = X[self.feature_columns]
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return -self.model.score_samples(X)  # IsolationForest: low score_samples = anomaly

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score(X) >= self.threshold).astype(int)


def save_bundle(bundle: ModelBundle, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path = MODEL_PATH) -> ModelBundle:
    return joblib.load(path)


def score_transactions(bundle: ModelBundle, transactions: pd.DataFrame) -> pd.DataFrame:
    """Score raw transactions (Time, V1..V28, Amount) and return score and fraud flag."""
    X = build_features(transactions)
    scores = bundle.score(X)
    return pd.DataFrame({
        "fraud_score": scores,
        "is_fraud": (scores >= bundle.threshold).astype(int),
    })


__all__ = ["ModelBundle", "save_bundle", "load_bundle", "score_transactions", "FEATURE_COLUMNS"]
