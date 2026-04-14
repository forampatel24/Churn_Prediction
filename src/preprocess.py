import os
import subprocess
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
DATASET = "blastchar/telco-customer-churn"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_FOLDER, "telco_clean.csv")

KAGGLE_PATH = r"C:\Users\Dell'\AppData\Roaming\Python\Python312\Scripts\kaggle.exe"


# -----------------------------
# Download dataset
# -----------------------------
def download_dataset():
    os.makedirs(DATA_FOLDER, exist_ok=True)

    subprocess.run([
        KAGGLE_PATH,
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        DATA_FOLDER,
        "--unzip"
    ], check=True)

    print("Dataset downloaded!")


# -----------------------------
# Load dataset
# -----------------------------
def load_data():
    file_path = os.path.join(DATA_FOLDER, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    data = pd.read_csv(file_path)
    print("Loaded:", data.shape)

    return data


# -----------------------------
# Clean dataset
# -----------------------------
def clean_dataset(data):

    data.columns = data.columns.str.strip()

    # Fix TotalCharges
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

    # Drop ID
    if 'customerID' in data.columns:
        data.drop('customerID', axis=1, inplace=True)

    # Remove duplicates
    data.drop_duplicates(inplace=True)

    # Encode target
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

    # Encode categorical features
    data = pd.get_dummies(data, drop_first=True)

    return data


# -----------------------------
# Save dataset
# -----------------------------
def save_dataset(data):
    data.to_csv(OUTPUT_FILE, index=False)
    print("Saved clean dataset!")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    download_dataset()
    df = load_data()
    df = clean_dataset(df)
    save_dataset(df)