import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "backend", "validation", "artifacts")

CONF_MATRIX_CSV = os.path.join(ARTIFACTS_DIR, "confusion_matrix_counts.csv")
THRESHOLD_CSV = os.path.join(ARTIFACTS_DIR, "threshold_sweep.csv")
SAMPLE_LOG_CSV = os.path.join(ARTIFACTS_DIR, "sample_inference_log.csv")

FIG_DIR = ARTIFACTS_DIR
os.makedirs(FIG_DIR, exist_ok=True)


def load_confusion_matrix():
    df = pd.read_csv(CONF_MATRIX_CSV, header=None)
    return df.values


def plot_confusion_matrix_counts(cm):
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Figure 1: Confusion Matrix (Counts)\nCICIDS2017 LSTM IDS")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure_confusion_matrix_counts.png"))
    plt.close()


def plot_confusion_matrix_normalized(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm, row_sums, where=row_sums != 0)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, label="Recall (True Positive Rate)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Figure 2: Normalized Confusion Matrix (Recall per Class)\nCICIDS2017 LSTM IDS")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure_confusion_matrix_normalized.png"))
    plt.close()


def plot_false_positive_breakdown(cm):
    benign_row = cm[0]
    fp_counts = benign_row.copy()
    fp_counts[0] = 0

    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(fp_counts)), fp_counts)
    plt.xlabel("Predicted Attack Class")
    plt.ylabel("Number of Benign Samples")
    plt.title("Figure 3: False Positive Distribution\nBenign Traffic Misclassified as Attacks")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure_false_positive_breakdown.png"))
    plt.close()


def plot_threshold_curves():
    df = pd.read_csv(THRESHOLD_CSV)

    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.xlabel("Attack Probability Threshold")
    plt.ylabel("Metric Value")
    plt.title("Figure 4: Precision–Recall Tradeoff vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure_precision_recall_vs_threshold.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["false_positive_rate"], label="False Positive Rate")
    plt.plot(df["threshold"], df["false_negative_rate"], label="False Negative Rate")
    plt.xlabel("Attack Probability Threshold")
    plt.ylabel("Error Rate")
    plt.title("Figure 5: FP/FN Rates vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure_fp_fn_vs_threshold.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["coverage"])
    plt.xlabel("Attack Probability Threshold")
    plt.ylabel("Fraction of Samples Classified")
    plt.title("Figure 6: Decision Coverage vs Threshold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure_coverage_vs_threshold.png"))
    plt.close()


def plot_confidence_distribution():
    df = pd.read_csv(SAMPLE_LOG_CSV)

    correct = df[df["true"] == df["pred"]]["conf"].values
    incorrect = df[df["true"] != df["pred"]]["conf"].values

    plt.figure(figsize=(9, 5))
    plt.hist(correct, bins=50, density=True, alpha=0.7, label="Correct Predictions")
    plt.hist(incorrect, bins=50, density=True, alpha=0.7, label="Incorrect Predictions")
    plt.xlabel("Softmax Confidence Score")
    plt.ylabel("Probability Density")
    plt.title("Figure 7: Prediction Confidence Distribution\nCorrect vs Incorrect Classifications")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure_confidence_distribution.png"))
    plt.close()


def main():
    cm = load_confusion_matrix()
    plot_confusion_matrix_counts(cm)
    plot_confusion_matrix_normalized(cm)
    plot_false_positive_breakdown(cm)
    plot_threshold_curves()
    plot_confidence_distribution()
    print("All visualization figures generated successfully.")


if __name__ == "__main__":
    main()
