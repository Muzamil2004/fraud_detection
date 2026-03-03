# Credit Card Fraud Detection System

End-to-end fraud detection project with engineered behavioral features, model benchmarking, threshold tuning, and a FastAPI inference service.

## Overview

This repository provides:
- A training pipeline for imbalanced fraud data.
- Feature engineering focused on temporal and behavioral risk signals.
- Automated model benchmarking with hyperparameter search.
- Threshold and cost analysis utilities for business-driven decision tuning.
- A modular FastAPI service for single and batch predictions.
- Unit tests for core preprocessing, training utilities, and API runtime behavior.

## Architecture

Core modules:
- `src/preprocessing.py`: data split, preprocessing, behavioral feature engineering.
- `src/train.py`: model search, stratified CV, benchmarking, artifact export.
- `src/evaluate.py`: offline evaluation at configurable thresholds.
- `src/threshold_tuning.py`: threshold optimization by F1 or custom cost.
- `src/cost_analysis.py`: threshold-vs-cost report generation.
- `src/api/schemas.py`: API request/response schemas.
- `src/api/runtime.py`: model artifact loading, runtime cache, probability inference.
- `src/api/routes.py`: API endpoint definitions.

Interface:
- `app.py`: API bootstrap and router wiring.

Tests:
- `tests/test_preprocessing.py`
- `tests/test_train.py`
- `tests/test_api_runtime.py`

## Technical Highlights

- Models:
  - Logistic Regression
  - Random Forest
  - HistGradientBoosting
  - MLPClassifier
  - Soft Voting Ensemble
  - XGBoost
  - Optional: LightGBM, CatBoost (auto-included if installed)
- Validation: Stratified K-Fold with `RandomizedSearchCV`
- Optimization metric: PR-AUC (`average_precision`)
- Imbalance handling: SMOTE + class-weighted learners
- Artifact output: `model.pkl` with selected model, metrics, and benchmark metadata

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python -m src.train --data data/credit_card_fraud_10k.csv --output model.pkl
```

Optional tuning controls:

```bash
python -m src.train --cv-folds 5 --search-iter 20
python -m src.train --no-smote
```

## Evaluation and Threshold Tuning

Evaluate trained model:

```bash
python -m src.evaluate --model model.pkl --data data/credit_card_fraud_10k.csv
```

Tune threshold for best F1:

```bash
python -m src.threshold_tuning --mode f1
```

Tune threshold for minimum business cost:

```bash
python -m src.threshold_tuning --mode cost --fp-cost 1 --fn-cost 15
```

Generate cost analysis table:

```bash
python -m src.cost_analysis --fp-cost 1 --fn-cost 15 --output data/cost_analysis.csv
```

## Run API

```bash
uvicorn app:app --reload
```

Endpoints:
- Swagger: `http://127.0.0.1:8000/docs`
- Frontend UI: `http://127.0.0.1:8000/ui`
- Health: `GET /health`
- Model info: `GET /model-info`
- Predict: `POST /predict`
- Batch predict: `POST /predict/batch`

Sample request payload:

```json
{
  "amount": 84.47,
  "transaction_hour": 22,
  "merchant_category": "Electronics",
  "foreign_transaction": 0,
  "location_mismatch": 0,
  "device_trust_score": 66,
  "velocity_last_24h": 3,
  "cardholder_age": 40
}
```

## Tests

```bash
pytest
```

