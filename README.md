# Credit Card Fraud Detection System

Production-focused fraud detection project with feature engineering, model benchmarking, threshold tuning, and a FastAPI inference service.

![System Architecture](docs/images/architecture.svg)

## UI Preview

![Frontend Dashboard](docs/images/ui-preview.svg)

## Key Features

- Behavioral and temporal feature engineering
- Stratified train/test split with model comparison
- Candidate model benchmarking (Logistic Regression, Random Forest, XGBoost in current artifact)
- Threshold-based fraud decisioning for single and batch inference
- FastAPI service with interactive Swagger UI
- Unit tests for preprocessing, training helpers, and API runtime

## Model Details

Current production artifact: `model.pkl`

- Best model: `xgboost`
- Target column: `is_fraud`
- Default decision threshold: `0.50`
- Inference output:
  - `fraud_probability` (0 to 1)
  - `is_fraud` (0/1 based on threshold)

Input features expected by API/model:

- `amount`
- `transaction_hour`
- `merchant_category`
- `foreign_transaction`
- `location_mismatch`
- `device_trust_score`
- `velocity_last_24h`
- `cardholder_age`

Engineered features include:

- Time cyclic encoding (`transaction_hour_sin`, `transaction_hour_cos`)
- Night-transaction flag
- Log amount and high-amount indicator
- Velocity proxy features
- Device-location and foreign-device risk interaction features

## Model Performance (Current `model.pkl`)

Metrics from `python -m src.evaluate --model model.pkl --data data/credit_card_fraud_10k.csv` at threshold `0.50`:

- Accuracy: `0.9960`
- Precision (fraud class): `0.7895`
- Recall (fraud class): `1.0000`
- F1-score (fraud class): `0.8824`
- ROC-AUC: `0.9996`
- PR-AUC: `0.9727`

Confusion matrix:

```text
[[1962,    8],
 [   0,   30]]
```

Notes:

- The model catches all frauds in this evaluation split (recall 1.0) with some false positives.
- Metrics can change when retraining due to data split, algorithm search settings, and dependency versions.

## Project Structure

```text
ml_project/
|-- app.py
|-- requirements.txt
|-- model.pkl
|-- data/
|-- frontend/
|-- src/
|   |-- api/
|   |-- preprocessing.py
|   |-- train.py
|   |-- evaluate.py
|   |-- threshold_tuning.py
|   |-- cost_analysis.py
|-- tests/
|-- docs/images/
```

## Requirements

- Python 3.10+ recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train model:

```bash
python -m src.train --data data/credit_card_fraud_10k.csv --output model.pkl
```

4. Run API:

```bash
uvicorn app:app --reload
```

5. Open dashboard and docs:

- UI: `http://127.0.0.1:8000/ui`
- Swagger: `http://127.0.0.1:8000/docs`
- OpenAPI spec: `http://127.0.0.1:8000/api/openapi.json`

## Training and Evaluation

Train with custom search settings:

```bash
python -m src.train --cv-folds 5 --search-iter 20
python -m src.train --no-smote
```

Evaluate model:

```bash
python -m src.evaluate --model model.pkl --data data/credit_card_fraud_10k.csv
```

Threshold tuning:

```bash
python -m src.threshold_tuning --mode f1
python -m src.threshold_tuning --mode cost --fp-cost 1 --fn-cost 15
```

Cost analysis export:

```bash
python -m src.cost_analysis --fp-cost 1 --fn-cost 15 --output data/cost_analysis.csv
```

## API Endpoints

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict/batch`
- `POST /reload-model`

Sample `POST /predict` payload:

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

## Testing

```bash
pytest
```
