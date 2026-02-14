import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACT_DIR = os.path.join(BASE_DIR, "backend", "validation", "artifacts")
MODEL_PATH = os.path.join(BASE_DIR, "models", "dos_lstm_final.keras")
X_PATH = os.path.join(BASE_DIR, "data", "tensors", "X_lstm.npy")
Y_PATH = os.path.join(BASE_DIR, "data", "tensors", "y_lstm.npy")

BENIGN_LABEL = 0
THRESHOLDS = np.linspace(0.1, 0.99, 45)
BATCH_SIZE = 2048
SAMPLE_SIZE = 50000

os.makedirs(ARTIFACT_DIR, exist_ok=True)

def load_data():
    model = tf.keras.models.load_model(MODEL_PATH)
    X = np.load(X_PATH, allow_pickle=True)
    y = np.load(Y_PATH, allow_pickle=True)
    idx = np.random.choice(len(X), min(SAMPLE_SIZE, len(X)), replace=False)
    return model, X[idx], y[idx]

def evaluate_thresholds(model, X, y):
    probs = model.predict(X, batch_size=BATCH_SIZE, verbose=1)
    pred_class = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)

    rows = []

    for T in THRESHOLDS:
        uncertain = confidence < T
        decided = ~uncertain

        decided_y = y[decided]
        decided_pred = pred_class[decided]

        benign_mask = decided_y == BENIGN_LABEL
        attack_mask = ~benign_mask

        fp = np.sum((decided_pred != BENIGN_LABEL) & benign_mask)
        fn = np.sum((decided_pred == BENIGN_LABEL) & attack_mask)
        tp = np.sum((decided_pred != BENIGN_LABEL) & attack_mask)

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        rows.append({
            "threshold": float(T),
            "attack_rate": float(np.mean((decided_pred != BENIGN_LABEL))),
            "benign_rate": float(np.mean((decided_pred == BENIGN_LABEL))),
            "uncertain_rate": float(np.mean(uncertain)),
            "false_positive_rate": float(fp / (np.sum(benign_mask) + 1e-9)),
            "false_negative_rate": float(fn / (np.sum(attack_mask) + 1e-9)),
            "precision": float(precision),
            "recall": float(recall)
        })

    return pd.DataFrame(rows)

def generate_plots(df):
    plt.figure(figsize=(9,6))
    plt.plot(df["threshold"], df["false_positive_rate"], label="False Positive Rate")
    plt.plot(df["threshold"], df["false_negative_rate"], label="False Negative Rate")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Rate")
    plt.title("FP / FN vs Confidence Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "figure_fp_fn_vs_threshold_decision.png"))
    plt.close()

    plt.figure(figsize=(9,6))
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall vs Confidence Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "figure_precision_recall_vs_threshold_decision.png"))
    plt.close()

    plt.figure(figsize=(9,6))
    plt.plot(df["threshold"], df["uncertain_rate"])
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Uncertain Fraction")
    plt.title("Decision Deferral vs Threshold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "figure_uncertainty_vs_threshold.png"))
    plt.close()

def select_operating_point(df):
    df = df[(df["false_negative_rate"] < 0.01)]
    best = df.sort_values("false_positive_rate").iloc[0]
    return best.to_dict()

def main():
    model, X, y = load_data()
    df = evaluate_thresholds(model, X, y)

    csv_path = os.path.join(ARTIFACT_DIR, "decision_threshold_analysis.csv")
    df.to_csv(csv_path, index=False)

    generate_plots(df)

    best = select_operating_point(df)

    summary_path = os.path.join(ARTIFACT_DIR, "evaluation_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        summary = {}

    summary["recommended_threshold"] = best
    summary["decision_logic"] = "three_state_confidence_thresholding"
    summary["phase_5_1_completed"] = True
    summary["timestamp"] = datetime.utcnow().isoformat()

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Phase 5.1 decision logic completed")

if __name__ == "__main__":
    main()
