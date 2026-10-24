"""Stage 5: data drift report with Evidently.

Compares the training period (reference) with the test period (current) feature by
feature. In production, run this on each new batch of live transactions and retrain
when the share of drifted features passes a limit.
"""
import json

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from fraud.config import METRICS_DIR, REPORTS_DIR, TEST_PATH, TRAIN_PATH
from fraud.features import FEATURE_COLUMNS

DRIFT_LIMIT = 0.3  # retrain if more than 30% of features drift


def main() -> dict:
    reference = pd.read_parquet(TRAIN_PATH, columns=FEATURE_COLUMNS)
    current = pd.read_parquet(TEST_PATH, columns=FEATURE_COLUMNS)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    REPORTS_DIR.mkdir(exist_ok=True)
    report.save_html(str(REPORTS_DIR / "drift.html"))

    result = report.as_dict()["metrics"][0]["result"]
    share = float(result["share_of_drifted_columns"])
    summary = {
        "drifted_features": int(result["number_of_drifted_columns"]),
        "total_features": int(result["number_of_columns"]),
        "share_drifted": share,
        "retrain_recommended": share > DRIFT_LIMIT,
    }
    METRICS_DIR.mkdir(exist_ok=True)
    (METRICS_DIR / "drift.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
