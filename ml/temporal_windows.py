import pandas as pd
import numpy as np
from pathlib import Path
import sys
import csv
import time
from collections import defaultdict

# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_DIR = Path("data/final")
OUTPUT_DIR = Path("data/temporal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "temporal_windows_full.csv"
STATS_PATH = OUTPUT_DIR / "temporal_generation_stats.csv"

LABEL_COL = "label_binary"

WINDOW_SIZE = 5
STRIDE = 1
CHUNK_SIZE = 50_000

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FAIL_ON_SCHEMA_MISMATCH = True
FAIL_ON_NAN = True

# ==========================================================
# LOGGING + FAIL
# ==========================================================

def log(msg):
    print(f"[INFO] {msg}")

def warn(msg):
    print(f"[WARN] {msg}")

def hard_fail(msg):
    print("\n[FATAL]")
    print(msg)
    sys.exit(1)

# ==========================================================
# FILE DISCOVERY
# ==========================================================

if not INPUT_DIR.exists():
    hard_fail("data/final directory does not exist")

input_files = sorted(INPUT_DIR.glob("*_final.csv"))

if not input_files:
    hard_fail("No *_final.csv files found")

log(f"Discovered {len(input_files)} input files")

# ==========================================================
# SCHEMA LOCKING
# ==========================================================

schema_probe = pd.read_csv(input_files[0], nrows=100)

if LABEL_COL not in schema_probe.columns:
    hard_fail(f"{LABEL_COL} missing in schema file")

numeric_features = []
for col in schema_probe.columns:
    if col == LABEL_COL:
        continue
    if pd.api.types.is_numeric_dtype(schema_probe[col]):
        numeric_features.append(col)

numeric_features = sorted(numeric_features)

if len(numeric_features) == 0:
    hard_fail("No numeric features detected")

FEATURE_COUNT = len(numeric_features)
EXPECTED_FLAT_FEATURES = WINDOW_SIZE * FEATURE_COUNT

log(f"Locked feature count: {FEATURE_COUNT}")
log(f"Window size: {WINDOW_SIZE}")
log(f"Flattened features per sample: {EXPECTED_FLAT_FEATURES}")

# ==========================================================
# OUTPUT HEADER
# ==========================================================

header = ["window_id", "label_binary", "source_day"]

for t in range(WINDOW_SIZE):
    for col in numeric_features:
        header.append(f"t{t}_{col}")

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

# ==========================================================
# GLOBAL STATS
# ==========================================================

global_stats = defaultdict(int)
start_time = time.time()

window_id = 0

# ==========================================================
# PROCESS EACH FILE
# ==========================================================

for file_path in input_files:

    day = file_path.stem.replace("_final", "").lower()
    log(f"Processing {file_path.name}")

    file_rows = 0
    file_windows = 0
    prev_tail_values = None
    prev_tail_labels = None

    chunk_index = 0

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):

        chunk_index += 1

        if LABEL_COL not in chunk.columns:
            hard_fail(f"{LABEL_COL} missing in {file_path.name}")

        for col in numeric_features:
            if col not in chunk.columns:
                if FAIL_ON_SCHEMA_MISMATCH:
                    hard_fail(f"Feature {col} missing in {file_path.name}")
                else:
                    warn(f"Missing {col}, filling zeros")
                    chunk[col] = 0.0

        chunk = chunk[numeric_features + [LABEL_COL]]

        chunk[numeric_features] = chunk[numeric_features].apply(
            pd.to_numeric, errors="coerce"
        )

        if FAIL_ON_NAN and chunk[numeric_features].isna().any().any():
            warn("NaNs detected, forcing zero fill")
            chunk[numeric_features] = chunk[numeric_features].fillna(0.0)

        values = chunk[numeric_features].values.astype(np.float32)
        labels = chunk[LABEL_COL].values.astype(int)

        if prev_tail_values is not None:
            values = np.vstack([prev_tail_values, values])
            labels = np.concatenate([prev_tail_labels, labels])

        total_rows = len(values)
        file_rows += len(chunk)

        if total_rows < WINDOW_SIZE:
            prev_tail_values = values
            prev_tail_labels = labels
            continue

        end_limit = total_rows - WINDOW_SIZE + 1

        with open(OUTPUT_PATH, "a", newline="") as f:
            writer = csv.writer(f)

            for start in range(0, end_limit, STRIDE):

                end = start + WINDOW_SIZE

                window = values[start:end]
                label = int(labels[end - 1])

                row = [window_id, label, day]

                for t in range(WINDOW_SIZE):
                    row.extend(window[t].tolist())

                writer.writerow(row)

                window_id += 1
                file_windows += 1

        prev_tail_values = values[-(WINDOW_SIZE - 1):]
        prev_tail_labels = labels[-(WINDOW_SIZE - 1):]

        if chunk_index % 5 == 0:
            log(
                f"  chunk {chunk_index} | rows {file_rows} | windows {window_id}"
            )

    global_stats[day + "_rows"] = file_rows
    global_stats[day + "_windows"] = file_windows

    log(f"Finished {day}: rows={file_rows}, windows={file_windows}")

# ==========================================================
# FINAL VALIDATION
# ==========================================================

elapsed = time.time() - start_time

if window_id == 0:
    hard_fail("No temporal windows generated")

stats_df = pd.DataFrame(
    [{"metric": k, "value": v} for k, v in global_stats.items()]
)
stats_df["total_windows"] = window_id
stats_df["elapsed_seconds"] = elapsed

stats_df.to_csv(STATS_PATH, index=False)

# ==========================================================
# FINAL REPORT
# ==========================================================

log("TEMPORAL WINDOW GENERATION COMPLETE")
log(f"Saved dataset → {OUTPUT_PATH}")
log(f"Saved stats → {STATS_PATH}")
log(f"Total windows → {window_id}")
log(f"Elapsed time → {elapsed:.2f}s")
