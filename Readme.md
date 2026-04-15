# 📡 Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a telecom customer will churn using behavioral, service, and billing data.

---

## 🚀 Project Overview

Customer churn is a major problem in the telecom industry. This project builds a **complete ML pipeline + interactive Streamlit app** to:

* Predict churn probability
* Classify customers into risk levels (Low / Medium / High)
* Provide actionable retention strategies

---

## 🧠 Problem Statement

Predict whether a customer will:

* `0 → Stay`
* `1 → Churn`

based on usage, billing, and service features.

---

## 📂 Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── telco_clean.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│
├── model/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── imputer.pkl
│   ├── threshold.pkl
│   ├── columns.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## 🔍 Data Processing

The dataset is downloaded from Kaggle and cleaned using:

* Removal of duplicates
* Conversion of `TotalCharges` to numeric
* Missing value handling using median
* Encoding using `pd.get_dummies()`

---

## ⚙️ Feature Engineering

Additional features created:

* `AvgMonthlyCharge` → spending behavior
* `TenureBucket` → customer segmentation
* `NumAddons` → number of services used

---

## 🤖 Model Training

Multiple models were trained and compared:

* Logistic Regression
* Random Forest
* Gradient Boosting
* SVM (RBF Kernel)
* XGBoost (if installed)

### 🔥 Key Techniques Used

* SMOTE (to handle class imbalance)
* StandardScaler (feature scaling)
* Median Imputation
* Stratified Train-Test Split
* Threshold Optimization (instead of fixed 0.5)

---

## 📊 Model Performance

Example output during training:

```
Model                  Acc     AUC     F1    Rec   Thresh
---------------------------------------------------------
SVM (RBF)              ~0.74   ~0.83  ~0.70  ~0.79  ~0.40
```

💡 Focus was on **F1-score and recall** for churn class.

---

## 🔮 Prediction System

Prediction pipeline:

```
Raw Input → Feature Engineering → Imputer → Scaler → Model → Probability → Threshold → Risk Level
```

### Output Example:

```
Prediction      : ⚠️ WILL CHURN
Probability     : 56.19%
Risk Level      : Medium
```

---

## 🌐 Streamlit Web App

The app provides:

### 🔹 Single Customer Prediction

* User-friendly input form
* Probability gauge chart
* Risk classification
* Retention recommendations

### 🔹 Batch Prediction

* Upload CSV
* Get predictions for all customers
* Risk breakdown visualization
* Download results

---

## ▶️ How to Run

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd customer-churn-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Training (optional)

```bash
python src/train.py
```

### 4. Run Streamlit App

```bash
streamlit run app/app.py
```

---

## 📈 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Streamlit
* Plotly
* Matplotlib, Seaborn

---

## 🧠 Key Insights (from EDA)

* Customers with **low tenure churn more**
* Customers with **higher monthly charges churn more**
* Customers with **fewer services are more likely to churn**
* Dataset is **imbalanced (~26% churn)**

---

## 🎯 Future Improvements

* Deploy on cloud (Streamlit Cloud / AWS)
* Add FastAPI backend
* Use advanced models (LightGBM, XGBoost tuning)
* Improve UI/UX

---

## 👤 Author

Foram Patel

---

## ⭐ Conclusion

This project demonstrates:

* End-to-end ML pipeline design
* Feature engineering
* Model optimization
* Real-world deployment using Streamlit


