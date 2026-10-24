# Credit Card Fraud Detection with an End-to-End MLOps Pipeline

Detects fraudulent card transactions with two models, an unsupervised **Isolation Forest** and a supervised **gradient boosting** classifier, and runs the full model lifecycle: versioned data, tracked experiments, a feature store, an HTTP scoring API, drift monitoring and orchestration on Kubernetes.

![CI](https://github.com/akinpelumiajayi/credit-card-fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)

| Stage | Tool |
|---|---|
| Data and pipeline versioning | DVC |
| Experiment tracking and model registry | MLflow |
| Feature store (offline and online) | Feast |
| Model serving | BentoML |
| Drift monitoring | Evidently |
| Orchestration | Argo Workflows on Kubernetes |
| CI | GitHub Actions |

## The problem

Fraud is rare. In the ULB credit card dataset only 492 of 284,807 transactions (0.17%) are fraud, so a model predicting "not fraud" every time scores 99.8% accuracy and catches nothing. This project measures what matters to a bank instead:

- **PR-AUC**: how well the model ranks fraud above genuine transactions
- **Recall**: the share of fraud caught
- **Precision**: the share of alerts which are fraud, so analysts are not flooded
- **Fraud value caught**: the share of fraudulent money stopped

The decision threshold is chosen for the highest recall at **80% precision or better**, on a validation slice taken from the training period. The test set is never used for tuning.

## Data

[ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): European card transactions over two days in September 2013. Features `V1` to `V28` are PCA components, which protects cardholder details, plus `Time` and `Amount`. The pipeline downloads it from its [OpenML mirror](https://www.openml.org/d/1597).

The split is **time-based**: the model trains on the first 80% of transactions and is tested on the last 20%, the way a production model meets future data.

A synthetic generator with the same schema (`source: synthetic` in `params.yaml`) lets CI and tests run in seconds without the download.

## Pipeline

```
prepare ──> features ──┬──> train ──> evaluate
                       └──> drift
```

| Stage | Command | Output |
|---|---|---|
| prepare | `python -m fraud.data` | `data/raw/transactions.parquet` |
| features | `python -m fraud.features` | train and test sets, Feast source |
| train | `python -m fraud.train` | `models/model.joblib`, MLflow runs |
| evaluate | `python -m fraud.evaluate` | `metrics/eval.json`, PR curve |
| drift | `python -m fraud.drift` | `reports/drift.html`, `metrics/drift.json` |

**Features**: the 28 PCA components, `log_amount`, `hour_of_day` and `is_night`. One function (`build_features`) builds them for training, the feature store and the API, so training and serving never disagree.

**Models**: both are logged to MLflow with parameters, validation metrics and the model artifact. The one with the higher validation PR-AUC is saved for serving.

## Results

Run `dvc repro` on the real data, then `dvc metrics show`. Paste your numbers here:

| Metric (test period) | Value |
|---|---|
| Selected model | |
| PR-AUC | |
| Recall | |
| Precision | |
| Fraud value caught | |

## Quick start

```bash
git clone https://github.com/akinpelumiajayi/credit-card-fraud-detection-mlops.git
cd credit-card-fraud-detection-mlops
python -m venv .venv && source .venv/bin/activate      # Python 3.10 to 3.12
pip install -r requirements.txt
export PYTHONPATH=src

dvc repro               # run all five stages on the real dataset
dvc metrics show        # compare metrics
mlflow ui --backend-store-uri sqlite:///mlflow.db       # browse experiments at localhost:5000
```

To run on synthetic data, set `source: synthetic` in `params.yaml`.

### Feature store

```bash
cd feature_repo
feast apply                                             # register entity and feature view
feast materialize 2013-08-31T00:00:00 2013-09-03T00:00:00   # load the online store
cd ..
python -m fraud.train --use-feast                       # train from point-in-time correct features
```

### Scoring API

```bash
PYTHONPATH=src:. bentoml serve service.service:FraudDetector
```

```bash
curl -X POST localhost:3000/predict -H "Content-Type: application/json" \
  -d '{"transactions": [{"Time": 7200, "Amount": 950, "V14": -5, "V17": -4}]}'
```

```json
[{"fraud_score": 0.985, "is_fraud": true, "model": "gradient_boosting", "threshold": 0.286}]
```

Build a container with `bentoml build && bentoml containerize fraud_detector:latest`.

### Kubernetes

```bash
docker build -t ghcr.io/akinpelumiajayi/fraud-pipeline:latest .
argo submit k8s/argo-workflow.yaml -n argo --watch      # pipeline as a DAG
kubectl apply -f k8s/serving.yaml                       # API with 2 to 6 autoscaled pods
```

## Drift monitoring

Evidently compares each feature in the test period with the training period. If more than 30% of features drift, `metrics/drift.json` sets `retrain_recommended: true`. In production this runs on each new batch of transactions.

## Tests

```bash
pytest -q
```

Seven tests cover the feature logic, the threshold rule, the metrics and a full pipeline run on synthetic data. GitHub Actions runs them on every push, then runs `dvc repro`, `feast apply` and Feast-based training.

## Project structure

```
├── src/fraud/          data, features, train, evaluate, drift, metrics, predict, feast_features
├── service/service.py  BentoML API
├── feature_repo/       Feast entity and feature view
├── k8s/                Argo Workflow, Deployment, Service, HPA
├── tests/              pytest suite
├── dvc.yaml            pipeline stages
├── params.yaml         all tunable settings
└── .github/workflows/  CI
```

## Author

Akinpelumi Michael Ajayi, MSc Artificial Intelligence (Distinction), Sheffield Hallam University
[LinkedIn](https://www.linkedin.com/in/akinpelumi-michael-ajayi-) | [GitHub](https://github.com/akinpelumiajayi)

## Licence

MIT. See [LICENSE](LICENSE). The dataset is released by the ULB Machine Learning Group under the Open Database License.
