"""Read features from the Feast feature store.

Training uses the offline store (point-in-time correct joins on event_timestamp).
Serving uses the online store after `feast materialize`.
"""
import pandas as pd
from feast import FeatureStore

from fraud.config import ROOT
from fraud.features import FEATURE_COLUMNS

FEATURE_VIEW = "transaction_features"
FEATURE_REFS = [f"{FEATURE_VIEW}:{c}" for c in FEATURE_COLUMNS]


def store() -> FeatureStore:
    return FeatureStore(repo_path=str(ROOT / "feature_repo"))


def get_training_features(entity_df: pd.DataFrame) -> pd.DataFrame:
    """entity_df needs transaction_id and event_timestamp columns."""
    return store().get_historical_features(entity_df=entity_df, features=FEATURE_REFS).to_df()


def get_online_features(transaction_ids: list[int]) -> pd.DataFrame:
    rows = [{"transaction_id": int(t)} for t in transaction_ids]
    return store().get_online_features(features=FEATURE_REFS, entity_rows=rows).to_df()
