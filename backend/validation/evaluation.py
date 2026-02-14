import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import classification_report

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dos_lstm_final.keras")
DATA_DIR = os.path.join(BASE_DIR, "data", "tensors")
X_PATH = os.path.join(DATA_DIR, "X_lstm.npy")
Y_PATH = os.path.join(DATA_DIR, "y_lstm.npy")
VALIDATION_DIR = os.path.join(BASE_DIR, "backend", "validation")
ARTIFACT_DIR = os.path.join(VALIDATION_DIR, "artifacts")
LOG_DIR = os.path.join(VALIDATION_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "evaluation.log")

WINDOW_SIZE = 10
BENIGN_LABEL = 0
DEFAULT_SAMPLE_SIZE = 50000
DEFAULT_BATCH_SIZE = 2048
DEFAULT_SAMPLE_INFERENCE = 100
DEFAULT_THRESHOLDS = np.linspace(0.01, 0.99, 60)

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEVEN_CLASS_GROUPS = [
    [0],
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10, 11],
    [12, 13, 14, 15],
    [16, 17, 18],
    [19, 20, 21],
    [22, 23, 24, 25, 26]
]

def build_7class_map(groups):
    m = {}
    for new_label, group in enumerate(groups):
        for orig in group:
            m[int(orig)] = int(new_label)
    return m

CLASS_MAP_7 = build_7class_map(SEVEN_CLASS_GROUPS)

def log(msg):
    ts = datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def load_model_and_tensors(model_path=MODEL_PATH, x_path=X_PATH, y_path=Y_PATH):
    log("Loading model: " + model_path)
    model = tf.keras.models.load_model(model_path)
    log("Model loaded")
    log("Loading tensors X and y")
    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    log(f"Loaded tensors: X.shape={getattr(X,'shape',None)} y.shape={getattr(y,'shape',None)}")
    return model, X, y

def ensure_temporal_shape(X):
    if X.ndim == 3 and X.shape[1] == WINDOW_SIZE:
        return X
    raise ValueError(f"Unsupported X shape: {getattr(X,'shape',None)}")

def sample_inference_logging(model, X, y, sample_count=DEFAULT_SAMPLE_INFERENCE, batch_size=DEFAULT_BATCH_SIZE):
    n = len(X)
    sample_count = min(sample_count, n)
    indices = np.random.choice(n, sample_count, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    preds = model.predict(X_sample, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_conf = np.max(preds, axis=1)
    rows = []
    for i in range(sample_count):
        idx = int(indices[i])
        rows.append({"index": idx, "true": int(y_sample[i]), "pred": int(y_pred[i]), "conf": float(y_conf[i])})
        log(f"Idx={idx} | True={int(y_sample[i])} | Pred={int(y_pred[i])} | Conf={y_conf[i]:.4f}")
    p = os.path.join(ARTIFACT_DIR, "sample_inference_log.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    log("Saved sample inference log: " + p)
    return rows

def batch_evaluation(model, X, y, sample_size=DEFAULT_SAMPLE_SIZE, batch_size=DEFAULT_BATCH_SIZE, seed=None):
    n = len(X)
    if seed is not None:
        np.random.seed(seed)
    sample_size = min(sample_size, n)
    indices = np.random.choice(n, sample_size, replace=False)
    X_batch = X[indices]
    y_batch = y[indices]
    log(f"Running batch inference on {len(X_batch)} samples (batch_size={batch_size})")
    preds = model.predict(X_batch, batch_size=batch_size, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_conf = np.max(preds, axis=1)
    benign_mask = (y_batch == BENIGN_LABEL)
    attack_mask = ~benign_mask
    fp = int(np.sum(benign_mask & (y_pred != BENIGN_LABEL)))
    fn = int(np.sum(attack_mask & (y_pred == BENIGN_LABEL)))
    benign_total = int(np.sum(benign_mask))
    attack_total = int(np.sum(attack_mask))
    fpr = fp / (benign_total + 1e-9) if benign_total > 0 else None
    fnr = fn / (attack_total + 1e-9) if attack_total > 0 else None
    results = {
        "sampled": int(sample_size),
        "indices": indices,
        "y_batch": y_batch,
        "y_pred": y_pred,
        "probs": preds,
        "confidences": y_conf,
        "benign_total": benign_total,
        "attack_total": attack_total,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr
    }
    log(f"Batch results: samples={sample_size}, benign={benign_total}, attack={attack_total}, FP={fp}, FN={fn}")
    return results

def compute_confusion_matrix(y_true, y_pred, num_classes=None):
    if num_classes is None:
        num_classes = int(max(int(y_true.max()), int(y_pred.max())) + 1)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def per_class_metrics_from_cm(cm):
    tp = np.diag(cm).astype(float)
    support = np.sum(cm, axis=1).astype(float)
    predicted = np.sum(cm, axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    metrics = []
    for i in range(len(tp)):
        metrics.append({"class": int(i), "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])})
    return metrics, precision, recall, f1, support

def save_confusion_csv(cm, path):
    n = cm.shape[0]
    df = pd.DataFrame(cm, index=range(n), columns=range(n))
    df.to_csv(path, index=True)
    log("Saved confusion matrix csv: " + path)

def threshold_sweep_from_probs(probs, y_true, thresholds=DEFAULT_THRESHOLDS, benign_label=BENIGN_LABEL):
    p_benign = probs[:, int(benign_label)]
    p_attack = 1.0 - p_benign
    y_true_binary = (y_true != benign_label).astype(int)
    rows = []
    for t in thresholds:
        y_pred_binary = (p_attack >= t).astype(int)
        tp = int(np.sum((y_pred_binary == 1) & (y_true_binary == 1)))
        tn = int(np.sum((y_pred_binary == 0) & (y_true_binary == 0)))
        fp = int(np.sum((y_pred_binary == 1) & (y_true_binary == 0)))
        fn = int(np.sum((y_pred_binary == 0) & (y_true_binary == 1)))
        fpr = fp / (fp + tn + 1e-9)
        fnr = fn / (fn + tp + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        coverage = float(np.mean(p_attack >= t))
        rows.append({"threshold": float(t), "false_positive_rate": float(fpr), "false_negative_rate": float(fnr), "precision": float(precision), "recall": float(recall), "coverage": coverage})
    return pd.DataFrame(rows)

def save_threshold_csv(df, path):
    df.to_csv(path, index=False)
    log("Saved threshold sweep CSV: " + path)

def save_per_class_metrics(metrics, path):
    pd.DataFrame(metrics).to_csv(path, index=False)
    log("Saved per-class metrics: " + path)

def save_summary(summary, path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    log("Saved summary JSON: " + path)

def plot_confusion_matrix_counts(cm, outpath, figsize=(10,8)):
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (counts)")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    log("Saved confusion matrix counts image: " + outpath)

def plot_confusion_matrix_normalized(cm, outpath, figsize=(13,11)):
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    plt.figure(figsize=figsize)
    im = plt.imshow(cm_norm, cmap="Blues")
    cbar = plt.colorbar(im)
    cbar.set_label("Recall (True Positive Rate)")
    plt.title("Figure: Normalized Confusion Matrix (Recall per Class)\nCICIDS2017 LSTM")
    plt.xlabel("Predicted Class Label")
    plt.ylabel("True Class Label")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    log("Saved normalized confusion matrix image: " + outpath)

def plot_fp_breakdown(y_true, y_pred, outpath, benign_label=BENIGN_LABEL, figsize=(14,6)):
    benign_mask = (y_true == benign_label)
    benign_preds = y_pred[benign_mask]
    if len(benign_preds) == 0:
        log("No benign samples found for FP breakdown")
        return
    classes, counts = np.unique(benign_preds, return_counts=True)
    plt.figure(figsize=figsize)
    plt.bar(classes.astype(str), counts)
    plt.title("False Positive Distribution\nBenign Traffic Misclassified as Attack Classes")
    plt.xlabel("Predicted Attack Class")
    plt.ylabel("Number of Benign Samples (Count)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    log("Saved false positive breakdown image: " + outpath)

def plot_confidence_distributions(y_true, y_pred, confidences, outpath, figsize=(12,6)):
    correct = (y_pred == y_true)
    incorrect = ~correct
    plt.figure(figsize=figsize)
    if np.any(correct):
        plt.hist(confidences[correct], bins=50, alpha=0.7, label="Correct Predictions", density=True)
    if np.any(incorrect):
        plt.hist(confidences[incorrect], bins=50, alpha=0.7, label="Incorrect Predictions", density=True)
    plt.title("Prediction Confidence Distribution\nCorrect vs Incorrect Classifications")
    plt.xlabel("Softmax Confidence Score")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    log("Saved confidence distribution image: " + outpath)

def plot_threshold_tradeoffs(df, outpath_fpfn, outpath_pr, outpath_cov):
    plt.figure(figsize=(10,6))
    plt.plot(df["threshold"], df["false_positive_rate"], label="False Positive Rate")
    plt.plot(df["threshold"], df["false_negative_rate"], label="False Negative Rate")
    plt.xlabel("Attack Probability Threshold")
    plt.ylabel("Error Rate")
    plt.title("Binary IDS Error Rates vs Decision Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath_fpfn, dpi=200)
    plt.close()
    log("Saved FP/FN vs threshold image: " + outpath_fpfn)
    plt.figure(figsize=(10,6))
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.xlabel("Attack Probability Threshold")
    plt.ylabel("Metric Value")
    plt.title("Precision–Recall Tradeoff vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath_pr, dpi=200)
    plt.close()
    log("Saved precision/recall vs threshold image: " + outpath_pr)
    plt.figure(figsize=(10,6))
    plt.plot(df["threshold"], df["coverage"])
    plt.xlabel("Attack Probability Threshold")
    plt.ylabel("Fraction of Samples Classified")
    plt.title("Decision Coverage vs Threshold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath_cov, dpi=200)
    plt.close()
    log("Saved coverage vs threshold image: " + outpath_cov)

def generate_classification_report_csv(y_true, y_pred, outpath):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    for label, metrics in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append({"class": label, "precision": metrics.get("precision",0), "recall": metrics.get("recall",0), "f1": metrics.get("f1-score",0), "support": metrics.get("support",0)})
    pd.DataFrame(rows).to_csv(outpath, index=False)
    log("Saved classification report csv: " + outpath)

def map_labels(y, mapping):
    mapped = np.array([mapping.get(int(x), -1) for x in y], dtype=int)
    if np.any(mapped == -1):
        unmapped = np.unique(y[mapped == -1])
        log(f"Warning: found unmapped labels: {unmapped}. They will be assigned to last bucket.")
        max_m = max(mapping.values()) if len(mapping) else 0
        mapped[mapped == -1] = max_m
    return mapped

def run_all(model_path=MODEL_PATH, x_path=X_PATH, y_path=Y_PATH, sample_inference=DEFAULT_SAMPLE_INFERENCE, batch_sample_size=DEFAULT_SAMPLE_SIZE, batch_size=DEFAULT_BATCH_SIZE, thresholds=DEFAULT_THRESHOLDS):
    start = time.time()
    log("=== Unified Evaluation: start ===")
    model, X, y = load_model_and_tensors(model_path, x_path, y_path)
    if len(X) == 0:
        raise RuntimeError("X tensor is empty")
    X = ensure_temporal_shape(X)
    sample_inference_logging(model, X, y, sample_count=sample_inference, batch_size=batch_size)
    batch_results = batch_evaluation(model, X, y, sample_size=batch_sample_size, batch_size=batch_size)
    y_batch = batch_results["y_batch"]
    y_pred = batch_results["y_pred"]
    probs = batch_results["probs"]
    confidences = batch_results["confidences"]
    cm = compute_confusion_matrix(y_batch, y_pred)
    cm_csv = os.path.join(ARTIFACT_DIR, "confusion_matrix_counts.csv")
    save_confusion_csv(cm, cm_csv)
    pcm_path = os.path.join(ARTIFACT_DIR, "per_class_metrics.csv")
    metrics, p, r, f1, support = per_class_metrics_from_cm(cm)
    save_per_class_metrics(metrics, pcm_path)
    plot_confusion_matrix_counts(cm, os.path.join(ARTIFACT_DIR, "figure_confusion_matrix_counts.png"))
    plot_confusion_matrix_normalized(cm, os.path.join(ARTIFACT_DIR, "figure_confusion_matrix_normalized.png"))
    plot_fp_breakdown(y_batch, y_pred, os.path.join(ARTIFACT_DIR, "figure_false_positive_breakdown.png"))
    plot_confidence_distributions(y_batch, y_pred, confidences, os.path.join(ARTIFACT_DIR, "figure_confidence_distribution.png"))
    generate_classification_report_csv(y_batch, y_pred, os.path.join(ARTIFACT_DIR, "classification_report.csv"))
    thresh_df = threshold_sweep_from_probs(probs, y_batch, thresholds=thresholds)
    thresh_csv = os.path.join(ARTIFACT_DIR, "threshold_sweep.csv")
    save_threshold_csv(thresh_df, thresh_csv)
    plot_threshold_tradeoffs(thresh_df, os.path.join(ARTIFACT_DIR, "figure_fp_fn_vs_threshold.png"), os.path.join(ARTIFACT_DIR, "figure_precision_recall_vs_threshold.png"), os.path.join(ARTIFACT_DIR, "figure_coverage_vs_threshold.png"))
    summary = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "samples_evaluated": int(batch_results["sampled"]),
        "benign_samples": int(batch_results["benign_total"]),
        "attack_samples": int(batch_results["attack_total"]),
        "false_positives": int(batch_results["false_positives"]),
        "false_negatives": int(batch_results["false_negatives"]),
        "false_positive_rate": batch_results["false_positive_rate"],
        "false_negative_rate": batch_results["false_negative_rate"],
        "confidence": {
            "correct": confidences[(y_pred == y_batch)].tolist() if len(confidences) else [],
            "incorrect": confidences[(y_pred != y_batch)].tolist() if len(confidences) else []
        },
        "artifact_paths": {
            "confusion_matrix_counts": cm_csv,
            "per_class_metrics": pcm_path,
            "threshold_sweep": thresh_csv,
            "classification_report": os.path.join(ARTIFACT_DIR, "classification_report.csv")
        }
    }
    summary_path = os.path.join(ARTIFACT_DIR, "evaluation_summary.json")
    save_summary(summary, summary_path)

    log("Starting 7-class consolidation step")
    mapping = CLASS_MAP_7
    y_batch_7 = map_labels(y_batch, mapping)
    y_pred_7 = map_labels(y_pred, mapping)
    cm7 = compute_confusion_matrix(y_batch_7, y_pred_7, num_classes = len(SEVEN_CLASS_GROUPS))
    cm7_csv = os.path.join(ARTIFACT_DIR, "confusion_matrix_counts_7class.csv")
    save_confusion_csv(cm7, cm7_csv)
    metrics7, p7, r7, f17, support7 = per_class_metrics_from_cm(cm7)
    pcm7_path = os.path.join(ARTIFACT_DIR, "per_class_metrics_7class.csv")
    save_per_class_metrics(metrics7, pcm7_path)
    plot_confusion_matrix_counts(cm7, os.path.join(ARTIFACT_DIR, "figure_confusion_matrix_counts_7class.png"))
    plot_confusion_matrix_normalized(cm7, os.path.join(ARTIFACT_DIR, "figure_confusion_matrix_normalized_7class.png"))
    plot_fp_breakdown(y_batch_7, y_pred_7, os.path.join(ARTIFACT_DIR, "figure_false_positive_breakdown_7class.png"), benign_label=0)
    plot_confidence_distributions(y_batch_7, y_pred_7, confidences, os.path.join(ARTIFACT_DIR, "figure_confidence_distribution_7class.png"))
    generate_classification_report_csv(y_batch_7, y_pred_7, os.path.join(ARTIFACT_DIR, "classification_report_7class.csv"))
    thresh7_df = threshold_sweep_from_probs(probs, y_batch_7, thresholds=thresholds, benign_label=0)
    thresh7_csv = os.path.join(ARTIFACT_DIR, "threshold_sweep_7class.csv")
    save_threshold_csv(thresh7_df, thresh7_csv)
    plot_threshold_tradeoffs(thresh7_df, os.path.join(ARTIFACT_DIR, "figure_fp_fn_vs_threshold_7class.png"), os.path.join(ARTIFACT_DIR, "figure_precision_recall_vs_threshold_7class.png"), os.path.join(ARTIFACT_DIR, "figure_coverage_vs_threshold_7class.png"))

    summary["7class"] = {
        "mapping": CLASS_MAP_7,
        "confusion_matrix_counts": cm7_csv,
        "per_class_metrics": pcm7_path,
        "threshold_sweep": thresh7_csv,
        "classification_report": os.path.join(ARTIFACT_DIR, "classification_report_7class.csv")
    }
    save_summary(summary, summary_path)

    elapsed = time.time() - start
    log(f"=== Unified Evaluation: completed in {elapsed:.1f}s ===")
    return {"summary": summary, "confusion_matrix": cm, "per_class_metrics": metrics, "thresholds": thresh_df, "confusion_matrix_7class": cm7, "per_class_metrics_7class": metrics7, "thresholds_7class": thresh7_df}

def main(argv):
    sample_inference = DEFAULT_SAMPLE_INFERENCE
    batch_sample_size = DEFAULT_SAMPLE_SIZE
    batch_size = DEFAULT_BATCH_SIZE
    thresholds = DEFAULT_THRESHOLDS
    if "--sample" in argv:
        try:
            sample_inference = int(argv[argv.index("--sample") + 1])
        except Exception:
            pass
    if "--batch" in argv:
        try:
            batch_sample_size = int(argv[argv.index("--batch") + 1])
        except Exception:
            pass
    if "--bsize" in argv:
        try:
            batch_size = int(argv[argv.index("--bsize") + 1])
        except Exception:
            pass
    run_all(sample_inference=sample_inference, batch_sample_size=batch_sample_size, batch_size=batch_size, thresholds=thresholds)

if __name__ == "__main__":
    main(sys.argv)
