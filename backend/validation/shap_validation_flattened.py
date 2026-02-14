import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dos_lstm_final.keras")
DATA_DIR = os.path.join(BASE_DIR, "data", "tensors")
X_PATH = os.path.join(DATA_DIR, "X_lstm.npy")
Y_PATH = os.path.join(DATA_DIR, "y_lstm.npy")

VALIDATION_DIR = os.path.join(BASE_DIR, "backend", "validation")
ARTIFACT_DIR = os.path.join(VALIDATION_DIR, "artifacts")
LOG_DIR = os.path.join(VALIDATION_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "shap_validation_flattened.log")

BENIGN_LABEL = 0
WINDOW_SIZE = 10
NUM_SAMPLES_PER_GROUP = 5
BACKGROUND_SIZE = 30
POOL_SIZE = 100000
TOP_K = 10
RANDOM_SEED = 42

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

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
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    log(f"Loaded X shape={X.shape}, y shape={y.shape}")
    return model, X, y

def flatten_X(X):
    return X.reshape(X.shape[0], -1)

def predict_wrapper(model):
    def f(x_flat):
        x = x_flat.reshape((-1, WINDOW_SIZE, x_flat.shape[1] // WINDOW_SIZE))
        return model.predict(x, verbose=0)
    return f

def select_samples(model, X, y):
    rng = np.random.default_rng(RANDOM_SEED)
    pool_idx = rng.choice(len(X), min(POOL_SIZE, len(X)), replace=False)

    preds = model.predict(X[pool_idx], batch_size=2048, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    conf = np.max(preds, axis=1)

    tp = pool_idx[(y[pool_idx] != BENIGN_LABEL) & (y_pred == y[pool_idx]) & (conf > 0.8)]
    fp = pool_idx[(y[pool_idx] == BENIGN_LABEL) & (y_pred != BENIGN_LABEL)]
    low = pool_idx[conf < 0.5]

    rng.shuffle(tp)
    rng.shuffle(fp)
    rng.shuffle(low)

    return {
        "true_positive": tp[:NUM_SAMPLES_PER_GROUP].tolist(),
        "false_positive": fp[:NUM_SAMPLES_PER_GROUP].tolist(),
        "low_confidence": low[:NUM_SAMPLES_PER_GROUP].tolist()
    }

def build_feature_names(num_features):
    return [f"t{t}_f{f}" for t in range(WINDOW_SIZE) for f in range(num_features)]

def reduce_shap_to_2d(shap_vals):
    shap_vals = np.array(shap_vals)

    if shap_vals.ndim == 3:
        shap_vals = np.mean(np.abs(shap_vals), axis=0)

    elif shap_vals.ndim == 2:
        shap_vals = np.abs(shap_vals)

    else:
        raise RuntimeError(f"Unexpected SHAP shape {shap_vals.shape}")

    return shap_vals

def run_shap_validation():
    start = time.time()
    log("=== Phase 6.1 SHAP Validation (Flattened KernelExplainer) ===")

    model, X, y = load_model_and_data()
    X_flat = flatten_X(X)
    feature_names = build_feature_names(X.shape[2])

    groups = select_samples(model, X, y)
    log(f"Selected samples: { {k: len(v) for k,v in groups.items()} }")

    bg_idx = np.random.choice(len(X_flat), BACKGROUND_SIZE, replace=False)
    background = X_flat[bg_idx]

    explainer = shap.KernelExplainer(
        predict_wrapper(model),
        background,
        link="logit"
    )

    rows = []

    for group, indices in groups.items():
        if not indices:
            continue

        log(f"Computing SHAP for group={group}")
        samples_flat = X_flat[indices]

        shap_vals = explainer.shap_values(samples_flat, nsamples=200)

        if isinstance(shap_vals, list):
            shap_vals = np.mean(np.abs(np.stack(shap_vals, axis=0)), axis=0)

        shap_2d = reduce_shap_to_2d(shap_vals)

        for i, sample_id in enumerate(indices):
            sample_shap = shap_2d[i]

            top_idx = np.argsort(sample_shap)[-TOP_K:][::-1]
            top_idx = top_idx.astype(int).tolist()

            top_feats = [feature_names[j] for j in top_idx]

            rows.append({
                "sample_id": int(sample_id),
                "group": group,
                "true_label": int(y[sample_id]),
                "top_features": ",".join(top_feats)
            })

        mean_shap = shap_2d.mean(axis=0)
        top_global = np.argsort(mean_shap)[-TOP_K:][::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(
            [feature_names[int(i)] for i in top_global][::-1],
            mean_shap[top_global][::-1]
        )
        plt.title(f"Global SHAP (Flattened) — {group}")
        plt.tight_layout()
        out = os.path.join(ARTIFACT_DIR, f"figure_shap_flattened_{group}.png")
        plt.savefig(out, dpi=200)
        plt.close()

        log(f"Saved plot: {out}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(ARTIFACT_DIR, "shap_validation_flattened_summary.csv")
    df.to_csv(out_csv, index=False)
    log("Saved SHAP CSV: " + out_csv)

    log(f"=== Phase 6.1 completed in {time.time()-start:.1f}s ===")

if __name__ == "__main__":
    run_shap_validation()