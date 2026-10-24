"""Feast definitions: one entity (a transaction) and one feature view.

The parquet source is written by the DVC `features` stage.
Apply with:  cd feature_repo && feast apply
Load online: feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64

transaction = Entity(name="transaction", join_keys=["transaction_id"], value_type=ValueType.INT64,
                     description="A single card transaction")

source = FileSource(
    name="transaction_features_source",
    path="data/transaction_features.parquet",
    timestamp_field="event_timestamp",
)

FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour_of_day", "is_night"]

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=36500),  # 2013 data: keep it all valid
    schema=[Field(name=f, dtype=Float64) for f in FEATURES],
    source=source,
    online=True,
    tags={"team": "fraud", "owner": "akinpelumi-ajayi"},
)
