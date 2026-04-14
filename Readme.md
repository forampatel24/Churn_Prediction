# Customer Churn Prediction — Telecom

Predict whether a telecom customer will churn using demographic info, billing details, contract type, and service usage. Built with scikit-learn, XGBoost, and a Streamlit frontend.

---

## Project structure

```
customer-churn-prediction/
│
├── data/
│   └── telco.csv               ← Raw dataset (Kaggle)
│
├── notebooks/
│   └── EDA.ipynb               ← Exploratory data analysis
│
├── src/
│   ├── preprocess.py           ← Data cleaning & encoding
│   ├── train.py                ← Model training & evaluation
│   └── predict.py              ← Inference (single + batch)
│
├── model/
│   ├── churn_model.pkl         ← Trained Random Forest model
│   ├── scaler.pkl              ← StandardScaler
│   ├── imputer.pkl             ← SimpleImputer
│   ├── threshold.pkl           ← Optimal decision threshold
│   └── columns.pkl             ← Feature column order
│
├── app/
│   └── app.py                  ← Streamlit web app
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd customer-churn-prediction
pip install -r requirements.txt
```

### 2. Download the dataset

Download from Kaggle: https://www.kaggle.com/blastchar/telco-customer-churn  
Place the file at `data/telco.csv`.

---

## Usage

### Step 1 — Preprocess

```bash
python src/preprocess.py
```

Cleans raw data, encodes categoricals, and saves `data/telco_clean.csv`.

### Step 2 — Train

```bash
python src/train.py
```

Trains Logistic Regression, Random Forest, Gradient Boosting, SVM, and XGBoost.  
Applies SMOTE to handle class imbalance, tunes the decision threshold per model, and saves the best model artifacts to `model/`.

### Step 3 — Predict (optional CLI test)

```bash
python src/predict.py
```

Runs a quick single-customer prediction using a hardcoded sample.

For batch scoring from a CSV:

```python
from src.predict import predict_batch
results = predict_batch("data/new_customers.csv", output_path="data/predictions.csv")
```

### Step 4 — Run the Streamlit app

```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501` with two modes:
- **Single customer** — fill in a form and get an instant churn prediction with gauge chart and retention recommendations.
- **Batch CSV upload** — upload a CSV, score all customers, view risk breakdown, and download results.

---

## Model performance (best model: Random Forest)

| Metric | Score |
|---|---|
| Accuracy | 79% |
| AUC-ROC | 0.841 |
| Churn F1 | 0.636 |
| Churn precision | 0.58 |
| Churn recall | 0.71 |
| Decision threshold | 0.53 |

Class imbalance (~74% No / 26% Yes) handled with SMOTE on training data only.

---

## Key features used

- `tenure` — months with the company
- `MonthlyCharges` / `TotalCharges`
- `Contract` — month-to-month contracts are the strongest churn signal
- `InternetService` — fiber optic customers churn more
- `TenureBucket` — engineered: short / mid / long-term customer
- `AvgMonthlyCharge` — engineered: TotalCharges / (tenure + 1)
- `NumAddons` — engineered: count of active add-on services

---

## Dataset

**Telco Customer Churn** by IBM via Kaggle  
Link: https://www.kaggle.com/blastchar/telco-customer-churn  
7,043 customers · 21 features · Binary target: `Churn` (Yes / No)