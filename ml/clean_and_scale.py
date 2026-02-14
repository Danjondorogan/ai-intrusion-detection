import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================
# CONFIGURATION
# ==============================

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "monday": "monday_processed.csv",
    "tuesday": "tuesday_processed.csv",
    "wednesday": "wednesday_processed.csv",
    "thursday": "thursday_processed.csv",
    "friday": "friday_processed.csv",
}

LABEL_COL = "label_binary"

SCALER_PATH = OUTPUT_DIR / "standard_scaler.joblib"
FEATURE_LIST_PATH = OUTPUT_DIR / "feature_columns.txt"

print("\nPHASE 5.2 — DATA CLEANING & NORMALIZATION\n")

# ==============================
# STEP 1: LOAD ALL DATA
# ==============================

datasets = {}
total_rows = 0

for day, file in FILES.items():
    path = DATA_DIR / file
    print(f"[INFO] Loading {file}")
    df = pd.read_csv(path)
    datasets[day] = df
    total_rows += len(df)

print(f"\n[INFO] Total rows loaded: {total_rows}")

# ==============================
# STEP 2: IDENTIFY CONSTANT COLUMNS (GLOBAL)
# ==============================

print("\n[STEP] Detecting constant columns across ALL days")

feature_columns = [
    c for c in datasets["monday"].columns if c != LABEL_COL
]

constant_columns = set()

for col in feature_columns:
    unique_values = set()
    for df in datasets.values():
        unique_values.update(df[col].dropna().unique())
        if len(unique_values) > 1:
            break
    if len(unique_values) <= 1:
        constant_columns.add(col)

print(f" Constant columns detected: {len(constant_columns)}")
for c in constant_columns:
    print("  -", c)

# ==============================
# STEP 3: CLEAN EACH DATASET
# ==============================

cleaned_datasets = {}

for day, df in datasets.items():
    print(f"\n[INFO] Cleaning {day}")

    # Drop constant columns
    df = df.drop(columns=constant_columns)

    # Replace Inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Count NaNs before fill
    nan_before = df.isna().sum().sum()

    # Fill NaNs with column median (safe for numeric features)
    df.fillna(df.median(numeric_only=True), inplace=True)

    nan_after = df.isna().sum().sum()

    print(f" NaN before: {nan_before}, NaN after: {nan_after}")

    cleaned_datasets[day] = df

# ==============================
# STEP 4: SEPARATE FEATURES & LABELS
# ==============================

print("\n[STEP] Separating features and labels")

X_train = cleaned_datasets["monday"].drop(columns=[LABEL_COL])
y_train = cleaned_datasets["monday"][LABEL_COL]

print(f" Training set shape: {X_train.shape}")

# ==============================
# STEP 5: FIT STANDARD SCALER (TRAIN ONLY)
# ==============================

print("\n[STEP] Fitting StandardScaler on TRAINING data")

scaler = StandardScaler()
scaler.fit(X_train)

joblib.dump(scaler, SCALER_PATH)
print(f" Scaler saved to: {SCALER_PATH}")

# Save feature list (CRITICAL for deployment)
feature_columns = list(X_train.columns)
with open(FEATURE_LIST_PATH, "w") as f:
    for col in feature_columns:
        f.write(col + "\n")

print(f" Feature list saved to: {FEATURE_LIST_PATH}")

# ==============================
# STEP 6: APPLY SCALING TO ALL DAYS
# ==============================

print("\n[STEP] Applying scaling to all datasets")

for day, df in cleaned_datasets.items():
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]

    X_scaled = scaler.transform(X)

    final_df = pd.DataFrame(X_scaled, columns=feature_columns)
    final_df[LABEL_COL] = y.values

    output_path = OUTPUT_DIR / f"{day}_final.csv"
    final_df.to_csv(output_path, index=False)

    print(f" Saved cleaned & scaled data: {output_path}")

print("\nPHASE 5.2 COMPLETED SUCCESSFULLY")
