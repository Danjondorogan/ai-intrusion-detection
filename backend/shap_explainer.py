import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from backend.inference import model, WINDOW_SIZE, NUM_FEATURES

logger = logging.getLogger("SHAPExplainer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

BASE_DIR = Path(".")
SCHEMA_PATH = BASE_DIR / "data" / "tensors" / "feature_schema.json"
SHAP_OUTPUT_DIR = BASE_DIR / "docs" / "shap_outputs"
SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLATTENED_FEATURES = WINDOW_SIZE * NUM_FEATURES

def load_feature_names() -> Optional[List[str]]:
    if not SCHEMA_PATH.exists():
        return None
    try:
        with open(SCHEMA_PATH, "r") as f:
            schema = json.load(f)
        names = (
            schema.get("feature_names")
            or schema.get("feature_columns")
            or schema.get("temporal_feature_columns")
        )
        if names and len(names) == NUM_FEATURES:
            return names
    except Exception:
        pass
    return None

FEATURE_NAMES = load_feature_names()

def lstm_predict_wrapper(X_flat: np.ndarray) -> np.ndarray:
    X_lstm = X_flat.reshape(-1, WINDOW_SIZE, NUM_FEATURES)
    return model.predict(X_lstm, verbose=0)

def build_background(n: int = 50) -> np.ndarray:
    return np.zeros((n, FLATTENED_FEATURES), dtype=np.float32)

BACKGROUND = build_background()
explainer = shap.KernelExplainer(lstm_predict_wrapper, BACKGROUND)

def flatten_temporal(x: np.ndarray) -> np.ndarray:
    return x.reshape(1, FLATTENED_FEATURES)

def restore_temporal(x: np.ndarray) -> np.ndarray:
    return x.reshape(WINDOW_SIZE, NUM_FEATURES)

def aggregate_importance(shap_matrix: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(shap_matrix), axis=0)

def feature_name(i: int) -> str:
    if FEATURE_NAMES and i < len(FEATURE_NAMES):
        return FEATURE_NAMES[i]
    return f"Feature {i}"

def plot_bar_importance(values: np.ndarray, session_id: str, top_k: int) -> str:
    idx = np.argsort(values)[-top_k:][::-1]
    names = [feature_name(i) for i in idx]
    vals = values[idx]

    plt.figure(figsize=(12, 6))
    plt.barh(names[::-1], vals[::-1])
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title("Top Feature Contributions")
    plt.tight_layout()

    path = SHAP_OUTPUT_DIR / f"shap_feature_importance_{session_id}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return str(path)

def plot_temporal_heatmap(matrix: np.ndarray, session_id: str) -> str:
    vmax = np.percentile(np.abs(matrix), 95)
    vmax = max(vmax, 1e-6)

    plt.figure(figsize=(16, 6))
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={"label": "SHAP Contribution"}
    )
    plt.xlabel("Feature Index")
    plt.ylabel("Time Step")
    plt.title("Temporal SHAP Attribution")
    plt.tight_layout()

    path = SHAP_OUTPUT_DIR / f"shap_temporal_{session_id}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return str(path)

def explain_lstm_decision(
    temporal_tensor: np.ndarray,
    session_id: str,
    max_samples: int = 200,
    top_k: int = 10
) -> Dict[str, Any]:

    if temporal_tensor.shape != (1, WINDOW_SIZE, NUM_FEATURES):
        raise ValueError("Invalid temporal tensor shape")

    flat = flatten_temporal(temporal_tensor)
    shap_values = explainer.shap_values(flat, nsamples=max_samples)
    shap_flat = shap_values[0][0]

    if shap_flat.shape[0] != FLATTENED_FEATURES:
        shap_flat = np.zeros(FLATTENED_FEATURES, dtype=np.float32)

    shap_matrix = restore_temporal(shap_flat)
    aggregated = aggregate_importance(shap_matrix)

    bar_path = plot_bar_importance(aggregated, session_id, top_k)
    heat_path = plot_temporal_heatmap(shap_matrix, session_id)

    top_indices = np.argsort(aggregated)[-top_k:][::-1]
    top_features = [
        {
            "feature_index": int(i),
            "feature_name": feature_name(i),
            "mean_abs_shap": float(aggregated[i]),
        }
        for i in top_indices
    ]

    return {
        "session_id": session_id,
        "plot_path": bar_path,
        "temporal_plot_path": heat_path,
        "top_features": top_features,
        "window_size": WINDOW_SIZE,
        "num_features": NUM_FEATURES,
        "explanation_type": "temporal_lstm_shap",
        "skipped": False,
    }
