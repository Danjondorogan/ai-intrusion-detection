import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
import tensorflow as tf

print("=== THRESHOLD ANALYSIS ===")

# -----------------------------
# LOAD DATA
# -----------------------------
X = np.load("data/tensors/X_lstm.npy", mmap_mode="r")
y = np.load("data/tensors/y_lstm.npy")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("models/dos_lstm_final.keras")

print("[INFO] Model loaded")

# -----------------------------
# USE SUBSET (FAST)
# -----------------------------
# ✅ RANDOM SAMPLING (FIX)
idx = np.random.choice(len(X), size=20000, replace=False)

X_sample = X[idx]
y_sample = y[idx]
# -----------------------------
# PREDICT
# -----------------------------
y_prob = model.predict(X_sample, batch_size=256).ravel()

# -----------------------------
# THRESHOLD ANALYSIS
# -----------------------------
thresholds = np.linspace(0, 1, 20)

precisions = []
recalls = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    p = precision_score(y_sample, y_pred, zero_division=0)
    r = recall_score(y_sample, y_pred, zero_division=0)

    precisions.append(p)
    recalls.append(r)

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(thresholds, precisions, marker='o', label="Precision")
plt.plot(thresholds, recalls, marker='s', label="Recall")

plt.title("Threshold vs Precision & Recall")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("docs/threshold_analysis.png", dpi=300)
plt.show()

print("[SUCCESS] Saved → docs/threshold_analysis.png")