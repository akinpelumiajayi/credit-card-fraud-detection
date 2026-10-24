"""BentoML service: score card transactions over HTTP.

Run locally:   bentoml serve service.service:FraudDetector   (from the repo root)
Build image:   bentoml build && bentoml containerize fraud_detector:latest
"""
from pathlib import Path

import bentoml
import pandas as pd
from pydantic import BaseModel, Field

from fraud.predict import load_bundle, score_transactions

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds since the first transaction in the dataset")
    Amount: float = Field(..., ge=0)
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0


class Prediction(BaseModel):
    fraud_score: float
    is_fraud: bool
    model: str
    threshold: float


@bentoml.service(name="fraud_detector", resources={"cpu": "1"}, traffic={"timeout": 10})
class FraudDetector:
    def __init__(self) -> None:
        self.bundle = load_bundle(MODEL_PATH)

    @bentoml.api
    def predict(self, transactions: list[Transaction]) -> list[Prediction]:
        df = pd.DataFrame([t.model_dump() for t in transactions])
        scored = score_transactions(self.bundle, df)
        return [
            Prediction(fraud_score=float(row.fraud_score), is_fraud=bool(row.is_fraud),
                       model=self.bundle.name, threshold=self.bundle.threshold)
            for row in scored.itertuples()
        ]

    @bentoml.api
    def health(self) -> dict:
        return {"status": "ok", "model": self.bundle.name}
