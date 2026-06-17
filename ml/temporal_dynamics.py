import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

INPUT_PATH = Path("data/attack_families/dos/dos_temporal_windows.csv")
OUTPUT_PATH = Path("data/attack_families/dos/dos_temporal_dynamics.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD
# -----------------------------
df = pd.read_csv(INPUT_PATH)

if df.empty:
    raise RuntimeError("Input temporal windows file is empty")

if "window_start" not in df.columns:
    raise RuntimeError("Missing required column: window_start")

if "window_end" not in df.columns:
    raise RuntimeError("Missing required column: window_end")

df = df.sort_values("window_start").reset_index(drop=True)

META_COLS = ["window_start", "window_end"]
BASE_COLS = [c for c in df.columns if c not in META_COLS]

if not BASE_COLS:
    raise RuntimeError("No base feature columns found after removing metadata columns")

# Ensure all base columns are numeric
for col in BASE_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

# -----------------------------
# BLOCK 1: FIRST-ORDER DELTAS
# -----------------------------
delta_features = {}
for col in BASE_COLS:
    delta_features[f"{col}_delta"] = df[col].diff().fillna(0.0)

delta_df = pd.DataFrame(delta_features)

# -----------------------------
# BLOCK 2: ROLLING STATISTICS
# -----------------------------
ROLL_WINDOWS = [3, 5, 10]
rolling_frames: list[pd.DataFrame] = []

for w in ROLL_WINDOWS:
    roll_mean_dict = {}
    roll_std_dict = {}

    for col in BASE_COLS:
        roll_mean_dict[f"{col}_roll{w}_mean"] = (
            df[col].rolling(window=w, min_periods=1).mean()
        )
        roll_std_dict[f"{col}_roll{w}_std"] = (
            df[col].rolling(window=w, min_periods=1).std().fillna(0.0)
        )

    rolling_frames.append(pd.DataFrame(roll_mean_dict))
    rolling_frames.append(pd.DataFrame(roll_std_dict))

# -----------------------------
# BLOCK 3: BURST FEATURES
# -----------------------------
if "flow_count_delta" in delta_df.columns:
    flow_count_delta = delta_df["flow_count_delta"]
else:
    flow_count_delta = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)

if "attack_density" in df.columns:
    attack_density = pd.to_numeric(df["attack_density"], errors="coerce").fillna(0.0)
else:
    attack_density = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)

burst_df = pd.DataFrame(
    {
        "burst_score": np.abs(flow_count_delta).astype(np.float32)
        * attack_density.astype(np.float32)
    }
)

# -----------------------------
# BLOCK 4: PERSISTENCE FEATURES
# -----------------------------
if "flow_count" in df.columns:
    flow_count = pd.to_numeric(df["flow_count"], errors="coerce").fillna(0.0)
else:
    flow_count = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)

persistence_score = (
    flow_count.rolling(window=5, min_periods=1)
    .apply(lambda x: float(np.sum(x > 0)), raw=True)
    .fillna(0.0)
)

persistence_df = pd.DataFrame(
    {"persistence_score": persistence_score.astype(np.float32)}
)

# -----------------------------
# FINAL CONCAT
# -----------------------------
frames: Tuple[pd.DataFrame, ...] = (
    df[META_COLS],
    df[BASE_COLS],
    delta_df,
    *tuple(rolling_frames),
    burst_df,
    persistence_df,
)

final_df = pd.concat(frames, axis=1)

# HARD DEFRAGMENT (GUARANTEED)
final_df = final_df.copy()

# -----------------------------
# SAVE
# -----------------------------
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved temporal dynamics → {OUTPUT_PATH}")
print(f"Rows → {final_df.shape[0]}")
print(f"Features → {final_df.shape[1]}")