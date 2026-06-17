import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("=== PROBABILITY DISTRIBUTION ===")

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
# RANDOM SAMPLE (IMPORTANT)
# -----------------------------
idx = np.random.choice(len(X), size=20000, replace=False)

X_sample = X[idx]
y_sample = y[idx]

# -----------------------------
# PREDICT
# -----------------------------
y_prob = model.predict(X_sample, batch_size=256).ravel()

# -----------------------------
# SPLIT BY CLASS
# -----------------------------
normal_probs = y_prob[y_sample == 0]
attack_probs = y_prob[y_sample == 1]

# -----------------------------
# STATS
# -----------------------------
min_prob = y_prob.min()
max_prob = y_prob.max()
mean_prob = y_prob.mean()

normal_avg = normal_probs.mean()
attack_avg = attack_probs.mean()

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(9,6))

plt.hist(normal_probs, bins=50, alpha=0.6, label="Normal")
plt.hist(attack_probs, bins=50, alpha=0.6, label="Attack")

plt.title("Prediction Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")

plt.legend()
plt.grid(True)

# -----------------------------
# ADD TEXT BOX (🔥 IMPORTANT)
# -----------------------------
textstr = (
    f"Min: {min_prob:.2e}\n"
    f"Max: {max_prob:.6f}\n"
    f"Mean: {mean_prob:.4f}\n"
    f"Normal Avg: {normal_avg:.4f}\n"
    f"Attack Avg: {attack_avg:.4f}"
)

plt.gca().text(
    0.65, 0.75, textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.tight_layout()
plt.savefig("docs/probability_distribution.png", dpi=300)
plt.show()