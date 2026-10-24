"""Stage 4: score the held-out test period with the selected model."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from fraud.config import METRICS_DIR, RAW_PATH, REPORTS_DIR, TARGET, TEST_PATH  # noqa: E402
from fraud.metrics import classification_report  # noqa: E402
from fraud.predict import load_bundle  # noqa: E402


def main() -> dict:
    bundle = load_bundle()
    test = pd.read_parquet(TEST_PATH)
    raw = pd.read_parquet(RAW_PATH, columns=["Amount"])
    amounts = raw["Amount"].to_numpy()[test["transaction_id"].to_numpy()]

    y = test[TARGET].to_numpy()
    scores = bundle.score(test)
    report = {"model": bundle.name, **classification_report(y, scores, bundle.threshold, amounts)}

    METRICS_DIR.mkdir(exist_ok=True)
    (METRICS_DIR / "eval.json").write_text(json.dumps(report, indent=2))

    precision, recall, _ = precision_recall_curve(y, scores)
    REPORTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="#1F4E79")
    ax.scatter([report["recall"]], [report["precision"]], color="#C00000", zorder=3, label="chosen threshold")
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts which are fraud)")
    ax.set_title(f"Test set PR curve, {bundle.name} (AUC {report['pr_auc']:.3f})")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "pr_curve.png", dpi=120)

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
