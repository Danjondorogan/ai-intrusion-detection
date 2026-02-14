import os
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
from datetime import datetime

# ==============================
# Silence SHAP / sklearn noise
# ==============================
warnings.filterwarnings("ignore")

# ==============================
# Paths
# ==============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dos_lstm_final.keras")
DATA_DIR = os.path.join(BASE_DIR, "data", "tensors")
X_PATH = os.path.join(DATA_DIR, "X_lstm.npy")
Y_PATH = os.path.join(DATA_DIR, "y_lstm.npy")

VALIDATION_DIR = os.path.join(BASE_DIR, "backend", "validation")
ARTIFACT_DIR = os.path.join(VALIDATION_DIR, "artifacts")
LOG_DIR = os.path.join(VALIDATION_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "shap_final_evaluation.log")

# ==============================
# Config
# ==============================
WINDOW_SIZE = 10
BACKGROUND_SIZE = 30
NUM_SAMPLES = 40
TOP_K = 20
RANDOM_SEED = 42

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

# ==============================
# Logging
# ==============================
def log(msg):
    ts = datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ==============================
# Load
# ==============================
def load_model_and_data():
    log("Loading model")
    model = tf.keras.models.load_model(MODEL_PATH)

    log("Loading tensors")
    X = np.load(X_PATH, allow_pickle=True)
    y = np.load(Y_PATH, allow_pickle=True)

    log(f"Loaded X shape={X.shape}, y shape={y.shape}")
    return model, X, y

# ==============================
# Helpers
# ==============================
def flatten_X(X):
    return X.reshape(X.shape[0], -1)

def predict_wrapper(model, num_features):
    def f(x_flat):
        x = x_flat.reshape((-1, WINDOW_SIZE, num_features))
        return model.predict(x, verbose=0)
    return f

def build_feature_names(num_features):
    names = []
    for t in range(WINDOW_SIZE):
        for f in range(num_features):
            names.append(f"t{t}_f{f}")
    return np.array(names)

# ==============================
# Normalize SHAP
# ==============================
def normalize_shap(shap_vals):
    # Multiclass case
    if isinstance(shap_vals, list):
        shap_vals = np.array(shap_vals)
        shap_vals = np.mean(np.abs(shap_vals), axis=0)

    shap_vals = np.array(shap_vals)

    # Flatten if needed
    if shap_vals.ndim > 2:
        shap_vals = shap_vals.reshape(shap_vals.shape[0], -1)

    return shap_vals

# ==============================
# Main
# ==============================
def run():
    start_time = time.time()
    log("=== Phase 6.3 — Final Explainability Evaluation ===")

    model, X, y = load_model_and_data()

    num_features = X.shape[2]
    feature_names = build_feature_names(num_features)

    X_flat = flatten_X(X)

    # --------------------------
    # Sample selection
    # --------------------------
    idx = np.random.choice(len(X_flat), NUM_SAMPLES, replace=False)
    samples = X_flat[idx]

    bg_idx = np.random.choice(len(X_flat), BACKGROUND_SIZE, replace=False)
    background = X_flat[bg_idx]

    explainer = shap.KernelExplainer(
        predict_wrapper(model, num_features),
        background
    )

    log(f"Computing SHAP for {NUM_SAMPLES} samples")
    shap_vals = explainer.shap_values(samples, nsamples=100)

    shap_vals = normalize_shap(shap_vals)
    abs_shap = np.abs(shap_vals)

    log(f"Raw SHAP shape: {abs_shap.shape}")

    # ==========================
    # CRITICAL ALIGNMENT FIX
    # ==========================
    shap_features = abs_shap.shape[1]
    name_features = len(feature_names)

    final_features = min(shap_features, name_features)

    abs_shap = abs_shap[:, :final_features]
    feature_names = feature_names[:final_features]

    log(f"Aligned feature count: {final_features}")

    # ==========================
    # Global importance
    # ==========================
    global_importance = abs_shap.mean(axis=0)

    # Final safety
    if len(global_importance) != len(feature_names):
        log("Final alignment correction")
        min_len = min(len(global_importance), len(feature_names))
        global_importance = global_importance[:min_len]
        feature_names = feature_names[:min_len]

    # Save full importance
    global_df = pd.DataFrame({
        "feature": feature_names.tolist(),
        "importance": global_importance.tolist()
    })

    global_df = global_df.sort_values("importance", ascending=False)

    global_df.to_csv(os.path.join(ARTIFACT_DIR, "shap_global_full.csv"), index=False)
    global_df.head(TOP_K).to_csv(
        os.path.join(ARTIFACT_DIR, "shap_global_top20.csv"), index=False
    )

    log("Saved global importance")

    # ==========================
    # Time-step importance
    # ==========================
    usable = (len(global_importance) // num_features) * num_features
    reshaped = global_importance[:usable].reshape(-1, num_features)
    time_imp = reshaped.mean(axis=1)

    pd.DataFrame({
        "time_step": list(range(len(time_imp))),
        "importance": time_imp
    }).to_csv(os.path.join(ARTIFACT_DIR, "shap_time_importance.csv"), index=False)

    log("Saved time importance")

    # ==========================
    # Sample explanations
    # ==========================
    rows = []
    for i, sample_id in enumerate(idx):
        imp = abs_shap[i]
        top_idx = np.argsort(imp)[-10:][::-1]

        features = [feature_names[int(j)] for j in top_idx]

        rows.append({
            "sample_id": int(sample_id),
            "true_label": int(y[sample_id]),
            "top_features": ",".join(features)
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(ARTIFACT_DIR, "shap_sample_explanations.csv"),
        index=False
    )

    log("Saved sample explanations")

    # ==========================
    # Summary
    # ==========================
    summary = {
        "samples": NUM_SAMPLES,
        "background": BACKGROUND_SIZE,
        "features_used": final_features,
        "avg_time_per_sample_sec": (time.time() - start_time) / NUM_SAMPLES
    }

    pd.Series(summary).to_json(
        os.path.join(ARTIFACT_DIR, "shap_final_summary.json"),
        indent=4
    )

    elapsed = time.time() - start_time
    log(f"=== Phase 6.3 completed in {elapsed:.1f}s ===")

# ==============================
if __name__ == "__main__":
    run()