import pandas as pd
import numpy as np
from pathlib import Path

# ==============================
# CONFIG
# ==============================
DATA_DIR = Path("data/processed")

FILES = [
    "monday_processed.csv",
    "tuesday_processed.csv",
    "wednesday_processed.csv",
    "thursday_processed.csv",
    "friday_processed.csv",
]

LABEL_COL = "label_binary"  # explicit, correct, research-grade

print("\nDATA QUALITY CHECK\n")

total_rows = 0
global_nan = 0
global_inf = 0
constant_columns_global = set()

for file in FILES:
    path = DATA_DIR / file
    print(f"[INFO] Checking {file}")

    df = pd.read_csv(path)
    rows, cols = df.shape
    total_rows += rows

    print(f" Rows: {rows}, Columns: {cols}")

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in {file}")

    feature_cols = [c for c in df.columns if c != LABEL_COL]

    # NaN & Inf checks
    nan_count = df[feature_cols].isna().sum().sum()
    inf_count = np.isinf(df[feature_cols]).sum().sum()

    global_nan += nan_count
    global_inf += inf_count

    print(f" NaN values: {nan_count}")
    print(f" Inf values: {inf_count}")

    # Constant columns
    constant_cols = [
        c for c in feature_cols
        if df[c].nunique(dropna=False) <= 1
    ]

    print(f" Constant columns: {len(constant_cols)}")

    for c in constant_cols:
        constant_columns_global.add(c)

    print(" Label distribution:")
    print(df[LABEL_COL].value_counts())

print("\nGLOBAL SUMMARY")
print(f" Total rows: {total_rows}")
print(f" Total NaN: {global_nan}")
print(f" Total Inf: {global_inf}")
print(f" Constant columns (unique): {len(constant_columns_global)}")

if constant_columns_global:
    print("\nConstant columns sample:")
    for c in list(constant_columns_global)[:10]:
        print(" -", c)
