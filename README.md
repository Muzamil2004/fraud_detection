# Credit Card Fraud Detection System

Production-oriented fraud detection project using classical ML and threshold/cost tuning.

## Project Structure

```text
fraud-detection-system/
|
|-- data/
|-- notebooks/
|   |-- eda.ipynb
|
|-- src/
|   |-- preprocessing.py
|   |-- train.py
|   |-- evaluate.py
|   |-- threshold_tuning.py
|   |-- cost_analysis.py
|
|-- model.pkl
|-- app.py
|-- requirements.txt
|-- README.md
```

## Tech Stack

- Data: Pandas, NumPy
- Models: Logistic Regression, Random Forest, XGBoost
- Imbalance Handling: SMOTE, class-weight tuning, threshold tuning
- Metrics: Precision, Recall, F1, ROC-AUC, PR-AUC, cost-based analysis
- Deployment: FastAPI, Joblib, Swagger docs

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --data data/credit_card_fraud_10k.csv --output model.pkl
```

Optional without SMOTE:

```bash
python -m src.train --no-smote
```

## Evaluate

```bash
python -m src.evaluate --model model.pkl --data data/credit_card_fraud_10k.csv
```

## Threshold Tuning

Maximize F1:

```bash
python -m src.threshold_tuning --mode f1
```

Minimize custom business cost:

```bash
python -m src.threshold_tuning --mode cost --fp-cost 1 --fn-cost 15
```

## Cost Analysis Table

```bash
python -m src.cost_analysis --fp-cost 1 --fn-cost 15 --output data/cost_analysis.csv
```

## Run API

```bash
uvicorn app:app --reload
```

- Frontend UI: http://127.0.0.1:8000/ui
- Swagger UI: http://127.0.0.1:8000/docs
- Health: GET `/`
- Prediction: POST `/predict`

## Sample Prediction Payload

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
