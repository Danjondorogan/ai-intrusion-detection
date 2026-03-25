import pandas as pd
import numpy as np
import json
from pathlib import Path
import time
from typing import List

# =====================================================
# CONFIG
# =====================================================
INPUT_PATH = Path("data/temporal/temporal_windows_full.csv")
OUTPUT_DIR = Path("data/tensors")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = OUTPUT_DIR / "X_lstm.npy"
Y_PATH = OUTPUT_DIR / "y_lstm.npy"
SCHEMA_PATH = OUTPUT_DIR / "feature_schema.json"

# Match the current LSTM/backend contract
WINDOW_SIZE = 10

# =====================================================
# LOAD DATA
# =====================================================
print("[INFO] Loading temporal window dataset")
start_time = time.time()

if not INPUT_PATH.exists():
    raise RuntimeError(f"Missing input file: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)

if df.empty:
    raise RuntimeError("Temporal window dataset is empty")

if "label_binary" not in df.columns:
    raise RuntimeError("Missing label_binary column")

print(f"[INFO] Total rows loaded: {len(df)}")

# =====================================================
# IDENTIFY TEMPORAL FEATURE COLUMNS
# =====================================================
print("[INFO] Detecting temporal feature columns")

temporal_columns = [c for c in df.columns if c.startswith("t")]

if not temporal_columns:
    raise RuntimeError("No temporal feature columns found")

# Validate structure: t0_..., t1_..., ... t9_...
time_steps = sorted(
    {
        int(c[1:c.index("_")])
        for c in temporal_columns
        if c.startswith("t") and "_" in c and c[1:c.index("_")].isdigit()
    }
)

if time_steps != list(range(WINDOW_SIZE)):
    raise RuntimeError(f"Expected timesteps 0-{WINDOW_SIZE - 1}, got {time_steps}")

# Extract base feature names (after tX_)
base_features = sorted(
    set(c.split("_", 1)[1] for c in temporal_columns if "_" in c)
)

num_features = len(base_features)

if num_features < 10:
    raise RuntimeError("Too few features detected — pipeline is broken")

print(f"[INFO] Window size: {WINDOW_SIZE}")
print(f"[INFO] Features per timestep: {num_features}")

# =====================================================
# LOCK FEATURE ORDER (CRITICAL)
# =====================================================
ordered_columns: List[str] = []

for t in range(WINDOW_SIZE):
    for f in base_features:
        col = f"t{t}_{f}"
        if col not in df.columns:
            raise RuntimeError(f"Missing column: {col}")
        ordered_columns.append(col)

expected_total = WINDOW_SIZE * num_features
if len(ordered_columns) != expected_total:
    raise RuntimeError("Feature ordering mismatch")

print(f"[INFO] Total flattened features: {expected_total}")

# =====================================================
# BUILD LSTM TENSORS
# =====================================================
print("[INFO] Building LSTM tensors")

X_flat = df[ordered_columns].values.astype(np.float32)
y = df["label_binary"].values.astype(np.int64)

num_samples = int(len(df))

try:
    X = X_flat.reshape(num_samples, WINDOW_SIZE, num_features)
except Exception as e:
    raise RuntimeError(f"Reshape failed: {e}")

# =====================================================
# SANITY CHECKS
# =====================================================
print("[INFO] Running sanity checks")

if X.shape[0] != y.shape[0]:
    raise RuntimeError("X/y sample count mismatch")

if X.shape[1] != WINDOW_SIZE:
    raise RuntimeError("Incorrect timestep dimension")

if X.shape[2] != num_features:
    raise RuntimeError("Incorrect feature dimension")

if not np.isfinite(X).all():
    raise RuntimeError("NaN or Inf detected in X tensor")

# Make Pylance happy by forcing a concrete ndarray before np.unique
y_np = np.asarray(y, dtype=np.int64)
unique_labels = np.unique(y_np)
print(f"[INFO] Labels present: {unique_labels}")

# =====================================================
# SAVE OUTPUTS
# =====================================================
print("[INFO] Saving tensors")

np.save(str(X_PATH), X.astype(np.float32))
np.save(str(Y_PATH), y_np)

schema = {
    "window_size": int(WINDOW_SIZE),
    "num_features": int(num_features),
    "flattened_features": int(expected_total),
    "feature_names": base_features,
    "temporal_order": [
        f"t{t}_{f}" for t in range(WINDOW_SIZE) for f in base_features
    ],
    "num_samples": int(num_samples),
}

with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)

elapsed = time.time() - start_time

print("[SUCCESS] LSTM tensor generation complete")
print(f"[INFO] X shape: {X.shape}")
print(f"[INFO] y shape: {y_np.shape}")
print(f"[INFO] Saved → {X_PATH}")
print(f"[INFO] Saved → {Y_PATH}")
print(f"[INFO] Saved → {SCHEMA_PATH}")
print(f"[INFO] Elapsed time: {elapsed:.2f}s")