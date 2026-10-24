"""End-to-end run on synthetic data in a temp copy of the project paths."""
import json

import pandas as pd
import pytest

import fraud.config as cfg


@pytest.fixture()
def tmp_project(tmp_path, monkeypatch):
    paths = {
        "RAW_PATH": tmp_path / "data/raw/transactions.parquet",
        "TRAIN_PATH": tmp_path / "data/features/train.parquet",
        "TEST_PATH": tmp_path / "data/features/test.parquet",
        "FEAST_SOURCE_PATH": tmp_path / "feature_repo/data/transaction_features.parquet",
        "MODEL_PATH": tmp_path / "models/model.joblib",
        "METRICS_DIR": tmp_path / "metrics",
        "REPORTS_DIR": tmp_path / "reports",
    }
    import fraud.data, fraud.drift, fraud.evaluate, fraud.features, fraud.predict, fraud.train  # noqa: E401
    for mod in (cfg, fraud.data, fraud.features, fraud.train, fraud.evaluate, fraud.drift, fraud.predict):
        for name, value in paths.items():
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, value)
    monkeypatch.setattr(fraud.train, "ROOT", tmp_path)
    # save_bundle/load_bundle use default args bound at import time
    monkeypatch.setattr(fraud.train, "save_bundle", lambda b: fraud.predict.save_bundle(b, paths["MODEL_PATH"]))
    monkeypatch.setattr(fraud.evaluate, "load_bundle", lambda: fraud.predict.load_bundle(paths["MODEL_PATH"]))
    return paths


def test_full_pipeline_on_synthetic_data(tmp_project):
    from fraud import data, drift, evaluate, features, train

    data.main("synthetic")
    features.main()
    summary = train.main()
    report = evaluate.main()
    drift_summary = drift.main()

    assert summary["selected_model"] in {"isolation_forest", "gradient_boosting"}
    assert report["pr_auc"] > 0.5  # synthetic fraud is separable
    assert tmp_project["MODEL_PATH"].exists()
    assert (tmp_project["REPORTS_DIR"] / "drift.html").exists()
    assert 0.0 <= drift_summary["share_drifted"] <= 1.0
    assert json.loads((tmp_project["METRICS_DIR"] / "eval.json").read_text())["model"] == summary["selected_model"]


def test_scoring_raw_transactions(tmp_project):
    from fraud import data, features, train
    from fraud.predict import load_bundle, score_transactions

    raw = data.main("synthetic")
    features.main()
    train.main()
    bundle = load_bundle(tmp_project["MODEL_PATH"])
    out = score_transactions(bundle, raw.drop(columns="Class").head(5))
    assert list(out.columns) == ["fraud_score", "is_fraud"]
    assert len(out) == 5 and set(out["is_fraud"].unique()) <= {0, 1}
