import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("data/attack_families/dos/dos_temporal_windows.csv")
OUTPUT_PATH = Path("data/attack_families/dos/dos_temporal_dynamics.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD
# -----------------------------
df = pd.read_csv(INPUT_PATH)

if df.empty:
    raise RuntimeError("Input temporal windows file is empty")

df = df.sort_values("window_start").reset_index(drop=True)

META_COLS = ["window_start", "window_end"]
BASE_COLS = [c for c in df.columns if c not in META_COLS]

# -----------------------------
# BLOCK 1: FIRST-ORDER DELTAS
# -----------------------------
delta_features = {}
for col in BASE_COLS:
    delta_features[f"{col}_delta"] = df[col].diff().fillna(0)

delta_df = pd.DataFrame(delta_features)

# -----------------------------
# BLOCK 2: ROLLING STATISTICS
# -----------------------------
ROLL_WINDOWS = [3, 5, 10]

rolling_frames = []

for w in ROLL_WINDOWS:
    roll_mean_dict = {}
    roll_std_dict = {}

    for col in BASE_COLS:
        roll_mean_dict[f"{col}_roll{w}_mean"] = (
            df[col].rolling(window=w, min_periods=1).mean()
        )
        roll_std_dict[f"{col}_roll{w}_std"] = (
            df[col].rolling(window=w, min_periods=1).std().fillna(0)
        )

    rolling_frames.append(pd.DataFrame(roll_mean_dict))
    rolling_frames.append(pd.DataFrame(roll_std_dict))

# -----------------------------
# BLOCK 3: BURST FEATURES
# -----------------------------
burst_features = {
    "burst_score": (
        delta_df.get("flow_count_delta", 0).abs()
        * df.get("attack_density", 0)
    )
}

burst_df = pd.DataFrame(burst_features)

# -----------------------------
# BLOCK 4: PERSISTENCE FEATURES
# -----------------------------
persistence_features = {
    "persistence_score": (
        df.get("flow_count", 0)
        .rolling(window=5, min_periods=1)
        .apply(lambda x: np.sum(x > 0), raw=True)
    )
}

persistence_df = pd.DataFrame(persistence_features)

# -----------------------------
# FINAL CONCAT (SINGLE PASS)
# -----------------------------
final_df = pd.concat(
    [
        df[META_COLS],
        df[BASE_COLS],
        delta_df,
        *rolling_frames,
        burst_df,
        persistence_df,
    ],
    axis=1,
    copy=False,
)

# HARD DEFRAGMENT (GUARANTEED)
final_df = final_df.copy()

# -----------------------------
# SAVE
# -----------------------------
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved temporal dynamics → {OUTPUT_PATH}")
print(f"Rows → {final_df.shape[0]}")
print(f"Features → {final_df.shape[1]}")
