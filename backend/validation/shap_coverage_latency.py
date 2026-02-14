import os
import time
import json
import numpy as np
import tensorflow as tf
import shap
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "dos_lstm_final.keras")
DATA_DIR = os.path.join(BASE_DIR, "data", "tensors")
X_PATH = os.path.join(DATA_DIR, "X_lstm.npy")
Y_PATH = os.path.join(DATA_DIR, "y_lstm.npy")

VALIDATION_DIR = os.path.join(BASE_DIR, "backend", "validation")
ARTIFACT_DIR = os.path.join(VALIDATION_DIR, "artifacts")
LOG_DIR = os.path.join(VALIDATION_DIR, "logs")

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "shap_coverage_latency.log")

BENIGN_LABEL = 0
WINDOW_SIZE = 10
NUM_EVAL_SAMPLES = 30
BACKGROUND_SIZE = 20
SHAP_NSAMPLES = 100
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def log(msg):
    ts = datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def load_model_and_data():
    log("Loading model")
    model = tf.keras.models.load_model(MODEL_PATH)
    log("Loading tensors")
    X = np.load(X_PATH, allow_pickle=True)
    y = np.load(Y_PATH, allow_pickle=True)
    log(f"Loaded X shape={X.shape}, y shape={y.shape}")
    return model, X, y

def flatten_X(X):
    return X.reshape(X.shape[0], -1)

def predict_wrapper(model):
    def f(x_flat):
        x = x_flat.reshape((-1, WINDOW_SIZE, int(x_flat.shape[1] / WINDOW_SIZE)))
        return model.predict(x, verbose=0)
    return f

def select_samples(X, y):
    idx = np.random.choice(len(X), NUM_EVAL_SAMPLES, replace=False)
    return idx.tolist()

def run_phase_6_2():
    start_total = time.time()
    log("=== Phase 6.2 — SHAP Coverage & Latency ===")

    model, X, y = load_model_and_data()
    X_flat = flatten_X(X)

    sample_indices = select_samples(X, y)
    log(f"Selected {len(sample_indices)} samples for SHAP latency evaluation")

    bg_idx = np.random.choice(len(X_flat), BACKGROUND_SIZE, replace=False)
    background = X_flat[bg_idx]

    explainer = shap.KernelExplainer(
        predict_wrapper(model),
        background,
        link="logit"
    )

    latency_records = []
    success = 0
    failures = 0

    for i, idx in enumerate(sample_indices):
        x_sample = X_flat[idx:idx+1]
        t0 = time.time()
        try:
            _ = explainer.shap_values(x_sample, nsamples=SHAP_NSAMPLES)
            dt = time.time() - t0
            latency_records.append(dt)
            success += 1
            log(f"Sample {i+1}/{len(sample_indices)} SHAP OK — {dt:.3f}s")
        except Exception as e:
            failures += 1
            log(f"Sample {i+1}/{len(sample_indices)} SHAP FAILED — {e}")

    coverage = success / len(sample_indices)
    avg_latency = float(np.mean(latency_records)) if latency_records else None
    p95_latency = float(np.percentile(latency_records, 95)) if latency_records else None

    summary = {
        "phase": "6.2",
        "total_samples": len(sample_indices),
        "successful_explanations": success,
        "failed_explanations": failures,
        "coverage_ratio": coverage,
        "avg_latency_seconds": avg_latency,
        "p95_latency_seconds": p95_latency,
        "shap_nsamples": SHAP_NSAMPLES,
        "background_size": BACKGROUND_SIZE
    }

    out_json = os.path.join(ARTIFACT_DIR, "shap_coverage_latency_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start_total
    log(f"Saved SHAP coverage & latency summary: {out_json}")
    log(f"=== Phase 6.2 completed in {elapsed:.1f}s ===")

if __name__ == "__main__":
    run_phase_6_2()