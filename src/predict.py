import os
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")


def load_artifacts():
    """Load all saved model artifacts."""
    model     = pickle.load(open(os.path.join(MODEL_DIR, "churn_model.pkl"), "rb"))
    scaler    = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"),      "rb"))
    imputer   = pickle.load(open(os.path.join(MODEL_DIR, "imputer.pkl"),     "rb"))
    threshold = pickle.load(open(os.path.join(MODEL_DIR, "threshold.pkl"),   "rb"))
    columns   = pickle.load(open(os.path.join(MODEL_DIR, "columns.pkl"),     "rb"))
    return model, scaler, imputer, threshold, columns


def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Accept a dict of raw customer fields (same as telco_clean.csv columns)
    and return a DataFrame ready for the model pipeline.
    """
    df = pd.DataFrame([data])

    # --- same feature engineering as train.py ---
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)

    if "tenure" in df.columns:
        df["TenureBucket"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 36, 72],
            labels=[0, 1, 2]
        ).astype(float).fillna(0).astype(int)

    addon_cols = [c for c in df.columns if c in (
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    )]
    if addon_cols:
        df["NumAddons"] = sum(
            df[c].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
            for c in addon_cols
        )

    return df


def predict_single(data: dict) -> dict:
    """
    Predict churn for a single customer.

    Parameters
    ----------
    data : dict
        Raw customer fields (matching telco_clean.csv columns, excluding Churn).

    Returns
    -------
    dict with keys:
        - churn_prediction  : int  (1 = will churn, 0 = will stay)
        - churn_probability : float (0–1, probability of churn)
        - risk_level        : str  ("High" / "Medium" / "Low")
    """
    model, scaler, imputer, threshold, columns = load_artifacts()

    df = preprocess_input(data)

    # Align to training columns (add missing cols as 0, drop extras)
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]

    # Impute → Scale → Predict
    X = imputer.transform(df)
    X = scaler.transform(X)

    proba = model.predict_proba(X)[0][1]
    prediction = int(proba >= threshold)

    if proba >= 0.70:
        risk = "High"
    elif proba >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "churn_prediction":  prediction,
        "churn_probability": round(float(proba), 4),
        "risk_level":        risk,
    }


def predict_batch(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Predict churn for a batch of customers from a CSV file.

    Parameters
    ----------
    csv_path    : str  Path to input CSV (same schema as training data, no Churn column needed).
    output_path : str  Optional path to save results CSV.

    Returns
    -------
    DataFrame with original columns + churn_prediction, churn_probability, risk_level.
    """
    model, scaler, imputer, threshold, columns = load_artifacts()

    df_raw = pd.read_csv(csv_path)
    df = df_raw.copy()

    # Feature engineering
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)

    if "tenure" in df.columns:
        df["TenureBucket"] = pd.cut(
            df["tenure"], bins=[0, 12, 36, 72], labels=[0, 1, 2]
        ).astype(float).fillna(0).astype(int)

    addon_cols = [c for c in df.columns if c in (
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    )]
    if addon_cols:
        df["NumAddons"] = sum(
            df[c].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
            for c in addon_cols
        )

    # Align columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    X = df[columns]

    X = imputer.transform(X)
    X = scaler.transform(X)

    probas      = model.predict_proba(X)[:, 1]
    predictions = (probas >= threshold).astype(int)
    risk_levels = ["High" if p >= 0.70 else "Medium" if p >= 0.40 else "Low" for p in probas]

    df_raw["churn_prediction"]  = predictions
    df_raw["churn_probability"] = probas.round(4)
    df_raw["risk_level"]        = risk_levels

    if output_path:
        df_raw.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

    print(f"\nBatch prediction complete — {len(df_raw)} customers scored.")
    print(df_raw["risk_level"].value_counts().to_string())
    return df_raw


# -----------------------------
# Quick test when run directly
# -----------------------------
if __name__ == "__main__":
    sample_customer = {
        "tenure": 5,
        "MonthlyCharges": 85.0,
        "TotalCharges": 425.0,
        "Contract": 0,          # 0 = Month-to-month
        "PaymentMethod": 2,     # electronic check
        "InternetService": 1,   # Fiber optic
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "PaperlessBilling": 1,
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 0,
        "PhoneService": 1,
        "MultipleLines": 0,
    }

    result = predict_single(sample_customer)
    print("\n--- Single Customer Prediction ---")
    print(f"  Churn prediction  : {'YES - Will Churn' if result['churn_prediction'] else 'NO - Will Stay'}")
    print(f"  Churn probability : {result['churn_probability']:.1%}")
    print(f"  Risk level        : {result['risk_level']}")