import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("=== CLASS DISTRIBUTION PLOT ===")

# -----------------------------
# PATHS
# -----------------------------
DATA_PATH = Path("data/processed/processed_flows.csv")
OUTPUT_PATH = Path("docs/class_distribution.png")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

if "binary_label" not in df.columns:
    raise RuntimeError("binary_label column missing")

# -----------------------------
# COUNT CLASSES
# -----------------------------
counts = df["binary_label"].value_counts().sort_index()

print("[INFO] Class counts:")
print(counts)

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(5,4))

counts.plot(kind="bar", width=0.5)

plt.title("Class Distribution (Normal vs Attack)", fontsize=12)
plt.xlabel("Class")
plt.ylabel("Number of Samples")

plt.xticks([0,1], ["Normal", "Attack"], rotation=0)

plt.tight_layout()

# -----------------------------
# SAVE
# -----------------------------
plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print(f"[SUCCESS] Saved → {OUTPUT_PATH}")