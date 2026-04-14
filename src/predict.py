import os
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")


# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
def load_artifacts():
    model     = pickle.load(open(os.path.join(MODEL_DIR, "churn_model.pkl"), "rb"))
    scaler    = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
    imputer   = pickle.load(open(os.path.join(MODEL_DIR, "imputer.pkl"), "rb"))
    threshold = pickle.load(open(os.path.join(MODEL_DIR, "threshold.pkl"), "rb"))
    columns   = pickle.load(open(os.path.join(MODEL_DIR, "columns.pkl"), "rb"))
    return model, scaler, imputer, threshold, columns


# -----------------------------
# FEATURE ENGINEERING (MATCH TRAIN)
# -----------------------------
def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    # --- SAME AS TRAIN.PY ---

    # TotalCharges safe conversion
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Tenure bucket
    if "tenure" in df.columns:
        df["TenureBucket"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 36, 72],
            labels=[0, 1, 2]
        ).astype(float).fillna(0).astype(int)

    # Addon services
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


# -----------------------------
# SINGLE PREDICTION
# -----------------------------
def predict_single(data: dict):
    model, scaler, imputer, threshold, columns = load_artifacts()

    df = preprocess_input(data)

    # -----------------------------
    # COLUMN ALIGNMENT (CRITICAL)
    # -----------------------------
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]

    # -----------------------------
    # PIPELINE: IMPUTE → SCALE → PREDICT
    # -----------------------------
    X = imputer.transform(df)
    X = scaler.transform(X)

    proba = model.predict_proba(X)[0][1]
    prediction = int(proba >= threshold)

    # Risk segmentation
    if proba >= 0.7:
        risk = "High"
    elif proba >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "churn_prediction": prediction,
        "churn_probability": round(float(proba), 4),
        "risk_level": risk
    }


# -----------------------------
# BATCH PREDICTION
# -----------------------------
def predict_batch(csv_path, output_path=None):
    model, scaler, imputer, threshold, columns = load_artifacts()

    df_raw = pd.read_csv(csv_path)
    df = df_raw.copy()

    # SAME FEATURE ENGINEERING
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

    # Align columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    X = df[columns]

    X = imputer.transform(X)
    X = scaler.transform(X)

    probas = model.predict_proba(X)[:, 1]
    preds = (probas >= threshold).astype(int)

    df_raw["churn_prediction"] = preds
    df_raw["churn_probability"] = probas.round(4)
    df_raw["risk_level"] = [
        "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"
        for p in probas
    ]

    if output_path:
        df_raw.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")

    return df_raw


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    sample = {
        "tenure": 26,
        "MonthlyCharges": 45,
        "TotalCharges": 1080,
        "OnlineSecurity": "No",
        "TechSupport": "yes",
        "StreamingTV": "No",
        "StreamingMovies": "Yes"
    }

    result = predict_single(sample)

    print("\n" + "="*40)
    print(" CUSTOMER CHURN PREDICTION ")
    print("="*40)

    print(f"Prediction      : {'⚠️ WILL CHURN' if result['churn_prediction'] else '✅ WILL STAY'}")
    print(f"Probability     : {result['churn_probability']:.2%}")
    print(f"Risk Level      : {result['risk_level']}")

    print("="*40 + "\n")