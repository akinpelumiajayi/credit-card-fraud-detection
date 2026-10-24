"""Stage 3: train two models, track both in MLflow, keep the better one.

1. IsolationForest: unsupervised anomaly detection, needs no fraud labels.
2. HistGradientBoostingClassifier: supervised, with balanced class weights.

The decision threshold is chosen on a time-based validation slice (the last 20% of the
training period), never on the test set.
"""
import argparse
import json

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from fraud.config import METRICS_DIR, ROOT, TARGET, TRAIN_PATH, load_params
from fraud.features import FEATURE_COLUMNS, time_split
from fraud.metrics import classification_report, pick_threshold
from fraud.predict import ModelBundle, save_bundle


def load_training_frame(use_feast: bool) -> pd.DataFrame:
    train = pd.read_parquet(TRAIN_PATH)
    if not use_feast:
        return train
    from fraud.feast_features import get_training_features

    feats = get_training_features(train[["transaction_id", "event_timestamp"]])
    return feats.merge(train[["transaction_id", TARGET, "Time"]], on="transaction_id")


def build_models(params: dict, seed: int) -> dict:
    iso, gb = params["isolation_forest"], params["gradient_boosting"]
    return {
        "isolation_forest": IsolationForest(n_estimators=iso["n_estimators"], max_samples=iso["max_samples"],
                                            random_state=seed, n_jobs=-1),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=gb["max_iter"], learning_rate=gb["learning_rate"],
                                                            max_leaf_nodes=gb["max_leaf_nodes"], class_weight="balanced",
                                                            random_state=seed),
    }


def main(use_feast: bool = False) -> dict:
    params = load_params()
    seed = params["data"]["seed"]
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    name = params["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(name)

    df = load_training_frame(use_feast)
    fit_df, val_df = time_split(df, 0.2)
    X_fit, y_fit = fit_df[FEATURE_COLUMNS], fit_df[TARGET].to_numpy()
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET].to_numpy()

    results = {}
    for name, model in build_models(params["models"], seed).items():
        with mlflow.start_run(run_name=name):
            if name == "isolation_forest":
                model.fit(X_fit)
            else:
                model.fit(X_fit, y_fit)
            bundle = ModelBundle(name, model, 0.0, FEATURE_COLUMNS)
            scores = bundle.score(X_val)
            bundle.threshold = pick_threshold(y_val, scores, params["threshold"]["min_precision"])
            report = classification_report(y_val, scores, bundle.threshold)

            mlflow.log_params({"model": name, "use_feast": use_feast, **params["models"][name]})
            mlflow.log_metrics({f"val_{k}": v for k, v in report.items()})
            _log_model(model)
            results[name] = (bundle, report)
            print(f"{name:18s} val PR-AUC {report['pr_auc']:.3f}  recall {report['recall']:.3f}  "
                  f"precision {report['precision']:.3f}")

    best_name = max(results, key=lambda n: results[n][1]["pr_auc"])
    best_bundle, best_report = results[best_name]
    save_bundle(best_bundle)

    summary = {"selected_model": best_name,
               "validation": {n: r for n, (_, r) in results.items()}}
    METRICS_DIR.mkdir(exist_ok=True)
    (METRICS_DIR / "train.json").write_text(json.dumps(summary, indent=2))
    print(f"Selected {best_name} (val PR-AUC {best_report['pr_auc']:.3f})")
    return summary


def _log_model(model) -> None:
    """MLflow 3 renamed artifact_path to name."""
    if int(mlflow.__version__.split(".")[0]) >= 3:
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
    else:
        mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-feast", action="store_true", help="read training features from the Feast offline store")
    main(parser.parse_args().use_feast)
