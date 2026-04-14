import os
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_auc_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found. Install with: pip install xgboost")

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "telco_clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_FILE)
print(f"Dataset shape: {df.shape}")
print(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True).round(3)}\n")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
id_cols = [c for c in df.columns if c.lower() in ("customerid", "customer_id")]
df.drop(columns=id_cols, inplace=True, errors="ignore")

# Safe ratio — avoid divide-by-zero
if "TotalCharges" in df.columns and "tenure" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)

# Tenure bucket as plain integer (no pd.Categorical NaN edge cases)
if "tenure" in df.columns:
    df["TenureBucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 36, 72],
        labels=[0, 1, 2]
    ).astype(float).fillna(0).astype(int)

# Addon service count
addon_cols = [c for c in df.columns if c in (
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
)]
if addon_cols:
    df["NumAddons"] = sum(
        df[c].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
        for c in addon_cols
    )

# -----------------------------
# FEATURES / TARGET
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# -----------------------------
# IMPUTE NaNs BEFORE scaling and SMOTE
# -----------------------------
print(f"NaN count before imputation:\n{X.isnull().sum()[X.isnull().sum() > 0]}\n")

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

print(f"NaN count after imputation: {np.isnan(X_imputed).sum()}\n")

# -----------------------------
# SCALE
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# -----------------------------
# TRAIN / TEST SPLIT (stratified)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# SMOTE (applied only on training data)
# -----------------------------
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — train size: {X_train_sm.shape[0]}, "
      f"churn ratio: {y_train_sm.mean():.2f}\n")

# -----------------------------
# MODEL DEFINITIONS
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
    ),
    "SVM (RBF)": SVC(
        C=1, kernel="rbf", gamma="scale",
        class_weight="balanced", probability=True, random_state=42
    ),
}
if HAS_XGB:
    neg, pos = np.bincount(y_train_sm)
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="logloss", random_state=42, n_jobs=-1
    )

# -----------------------------
# TRAIN, THRESHOLD TUNE, EVALUATE
# -----------------------------
results = {}
best_f1 = 0
best_model_name = None
best_model_obj = None
best_threshold = 0.5

print("=" * 65)
print(f"{'Model':<22} {'Acc':>6} {'AUC':>6} {'F1':>6} {'Rec':>6} {'Thresh':>7}")
print("=" * 65)

for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)

    proba = model.predict_proba(X_test)[:, 1]

    # Find threshold that maximises churn F1
    best_t, best_f = 0.5, 0
    for t in np.arange(0.25, 0.60, 0.02):
        preds_t = (proba >= t).astype(int)
        f = f1_score(y_test, preds_t, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t

    preds = (proba >= best_t).astype(int)
    acc  = accuracy_score(y_test, preds)
    auc  = roc_auc_score(y_test, proba)
    rec  = classification_report(y_test, preds, output_dict=True)["1"]["recall"]

    results[name] = dict(acc=acc, auc=auc, f1=best_f, recall=rec, threshold=best_t)
    print(f"{name:<22} {acc:>6.3f} {auc:>6.3f} {best_f:>6.3f} {rec:>6.3f} {best_t:>7.2f}")

    if best_f > best_f1:
        best_f1, best_model_name = best_f, name
        best_model_obj, best_threshold = model, best_t

print("=" * 65)
print(f"\nBest model: {best_model_name}  (F1={best_f1:.3f}, threshold={best_threshold:.2f})")

# Detailed report for the winner
proba_best = best_model_obj.predict_proba(X_test)[:, 1]
preds_best = (proba_best >= best_threshold).astype(int)
print(f"\n--- {best_model_name} — Detailed Report ---")
print(classification_report(y_test, preds_best, target_names=["No Churn", "Churn"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds_best))

# 5-fold CV sanity check
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1 = cross_val_score(
    best_model_obj, X_train_sm, y_train_sm,
    cv=cv, scoring="f1", n_jobs=-1
).mean()
print(f"\n5-Fold CV F1 (train): {cv_f1:.3f}")

# -----------------------------
# SAVE ARTEFACTS
# -----------------------------
pickle.dump(best_model_obj,     open(os.path.join(MODEL_DIR, "churn_model.pkl"),  "wb"))
pickle.dump(scaler,             open(os.path.join(MODEL_DIR, "scaler.pkl"),       "wb"))
pickle.dump(imputer,            open(os.path.join(MODEL_DIR, "imputer.pkl"),      "wb"))
pickle.dump(best_threshold,     open(os.path.join(MODEL_DIR, "threshold.pkl"),    "wb"))
pickle.dump(X.columns.tolist(), open(os.path.join(MODEL_DIR, "columns.pkl"),     "wb"))

print("\nSaved: churn_model.pkl, scaler.pkl, imputer.pkl, threshold.pkl, columns.pkl")
print("Done!")