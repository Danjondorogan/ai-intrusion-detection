from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "dos_lstm_final.keras"

X_PATH = BASE_DIR / "data" / "tensors" / "X_lstm.npy"
Y_PATH = BASE_DIR / "data" / "tensors" / "y_lstm.npy"

SCHEMA_PATH = (
    BASE_DIR
    / "data"
    / "tensors"
    / "feature_schema.json"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "final_presentation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# VISUAL STYLE
# ==========================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 20,
})

PRIMARY = "#2563eb"
SUCCESS = "#10b981"
WARNING = "#f59e0b"
DANGER = "#ef4444"
DARK = "#111827"
LIGHT = "#f8fafc"

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("AI IDS PRESENTATION RESULTS GENERATOR")
print("=" * 70)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Loading tensors...")

X = np.load(
    X_PATH,
    mmap_mode="r"
)

y = np.load(
    Y_PATH
)

with open(
    SCHEMA_PATH,
    "r"
) as f:
    schema = json.load(f)

FEATURE_NAMES = schema["feature_names"]

TOTAL_SAMPLES = len(y)

WINDOW_SIZE = X.shape[1]

NUM_FEATURES = X.shape[2]

NORMAL_COUNT = int(
    np.sum(y == 0)
)

ATTACK_COUNT = int(
    np.sum(y == 1)
)

print(f"Samples      : {TOTAL_SAMPLES:,}")
print(f"Window Size  : {WINDOW_SIZE}")
print(f"Features     : {NUM_FEATURES}")
print(f"Normal       : {NORMAL_COUNT:,}")
print(f"Attack       : {ATTACK_COUNT:,}")

# ==========================================================
# DATASET OVERVIEW
# ==========================================================

print("\nGenerating Dataset Overview...")

fig = plt.figure(
    figsize=(14, 8)
)

fig.patch.set_facecolor("white")

plt.axis("off")

overview_text = f"""
AI-BASED INTRUSION DETECTION SYSTEM

Dataset Summary

Total Samples           : {TOTAL_SAMPLES:,}

Normal Samples          : {NORMAL_COUNT:,}

Attack Samples          : {ATTACK_COUNT:,}

Temporal Window Size    : {WINDOW_SIZE}

Features Per Timestep   : {NUM_FEATURES}

Model Input Shape       : ({WINDOW_SIZE}, {NUM_FEATURES})

Architecture            : LSTM Neural Network

Application             : Real-Time DoS Detection

Output                  : Binary Classification
"""

plt.text(
    0.05,
    0.95,
    overview_text,
    fontsize=16,
    va="top",
    family="monospace"
)

plt.title(
    "Dataset Overview and Research Configuration",
    pad=25,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "01_dataset_overview.png",
    dpi=500,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# CLASS DISTRIBUTION
# ==========================================================

print("Generating Class Distribution...")

fig, ax = plt.subplots(
    figsize=(10, 7)
)

bars = ax.bar(
    ["Normal", "Attack"],
    [NORMAL_COUNT, ATTACK_COUNT],
    color=[SUCCESS, DANGER],
    width=0.6
)

for bar in bars:

    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 3000,
        f"{height:,}",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

ax.set_title(
    "Dataset Class Distribution",
    pad=20,
    fontweight="bold"
)

ax.set_ylabel(
    "Number of Samples"
)

ax.grid(
    alpha=0.3,
    linestyle="--"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "02_class_distribution.png",
    dpi=500
)

plt.close()

# ==========================================================
# DATASET SUMMARY TABLE
# ==========================================================

print("Generating Dataset Summary Table...")

summary_df = pd.DataFrame({
    "Metric": [
        "Total Samples",
        "Normal Samples",
        "Attack Samples",
        "Window Size",
        "Features",
        "Input Shape",
        "Data Type",
        "Model Type"
    ],
    "Value": [
        f"{TOTAL_SAMPLES:,}",
        f"{NORMAL_COUNT:,}",
        f"{ATTACK_COUNT:,}",
        WINDOW_SIZE,
        NUM_FEATURES,
        f"({WINDOW_SIZE},{NUM_FEATURES})",
        str(X.dtype),
        "LSTM"
    ]
})

fig, ax = plt.subplots(
    figsize=(10, 4)
)

ax.axis("off")

table = ax.table(
    cellText=summary_df.values.tolist(),
    colLabels=summary_df.columns.tolist(),
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.3, 2)

plt.title(
    "Dataset Summary Table",
    pad=20,
    fontweight="bold"
)

plt.savefig(
    RESULTS_DIR /
    "03_dataset_summary_table.png",
    dpi=500,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# FEATURE STATISTICS
# ==========================================================

print("Generating Feature Statistics...")

sample_data = np.array(
    X[:5000]
)

flattened = sample_data.reshape(
    -1,
    NUM_FEATURES
)

feature_std = flattened.std(
    axis=0
)

top_idx = np.argsort(
    feature_std
)[-15:]

stats_df = pd.DataFrame({
    "Feature": [
        FEATURE_NAMES[i]
        for i in top_idx
    ],
    "Std Dev": feature_std[top_idx]
})

stats_df = stats_df.sort_values(
    "Std Dev",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(12, 8)
)

ax.barh(
    stats_df["Feature"],
    stats_df["Std Dev"],
    color=PRIMARY
)

ax.set_title(
    "Top 15 Most Variable Features",
    pad=20,
    fontweight="bold"
)

ax.set_xlabel(
    "Standard Deviation"
)

ax.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "04_feature_statistics.png",
    dpi=500
)

plt.close()

# ==========================================================
# EVALUATION DATA
# ==========================================================

print("\nPreparing Evaluation Dataset...")

EVAL_SIZE = min(
    25000,
    len(X)
)

rng = np.random.default_rng(42)

eval_idx = rng.choice(
    len(X),
    EVAL_SIZE,
    replace=False
)

X_eval = np.array(
    X[eval_idx]
)

y_eval = y[eval_idx]

print(
    f"Evaluation Samples: {len(X_eval):,}"
)

# ==========================================================
# MODEL PREDICTIONS
# ==========================================================

print("\nRunning Model Predictions...")

y_prob = model.predict(
    X_eval,
    batch_size=512,
    verbose=1
).flatten()

y_pred = (
    y_prob >= 0.5
).astype(int)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

print("Generating Confusion Matrix...")

cm = confusion_matrix(
    y_eval,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(
    figsize=(9, 8)
)

im = ax.imshow(
    cm,
    cmap="Blues"
)

labels = [
    [f"TN\n{tn:,}",
     f"FP\n{fp:,}"],
    [f"FN\n{fn:,}",
     f"TP\n{tp:,}"]
]

for i in range(2):
    for j in range(2):

        ax.text(
            j,
            i,
            labels[i][j],
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="black"
        )

ax.set_xticks([0,1])
ax.set_yticks([0,1])

ax.set_xticklabels(
    ["Normal","Attack"],
    fontsize=13
)

ax.set_yticklabels(
    ["Normal","Attack"],
    fontsize=13
)

ax.set_xlabel(
    "Predicted Class",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel(
    "Actual Class",
    fontsize=14,
    fontweight="bold"
)

ax.set_title(
    "Confusion Matrix - DoS Detection",
    fontsize=18,
    fontweight="bold",
    pad=20
)

plt.colorbar(
    im,
    fraction=0.046,
    pad=0.04
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "05_confusion_matrix.png",
    dpi=600
)

plt.close()

# ==========================================================
# ROC CURVE
# ==========================================================

print("Generating ROC Curve...")

fpr, tpr, _ = roc_curve(
    y_eval,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

ax.plot(
    fpr,
    tpr,
    color=PRIMARY,
    linewidth=4,
    label=f"AUC = {roc_auc:.5f}"
)

ax.plot(
    [0,1],
    [0,1],
    "--",
    color="gray",
    linewidth=2
)

ax.fill_between(
    fpr,
    tpr,
    alpha=0.15,
    color=PRIMARY
)

ax.set_title(
    "Receiver Operating Characteristic (ROC)",
    fontsize=18,
    fontweight="bold",
    pad=20
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.legend(
    loc="lower right"
)

ax.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "06_roc_curve.png",
    dpi=600
)

plt.close()

# ==========================================================
# PRECISION RECALL CURVE
# ==========================================================

print("Generating Precision Recall Curve...")

precision, recall, pr_thresholds = precision_recall_curve(
    y_eval,
    y_prob
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

ax.plot(
    recall,
    precision,
    linewidth=4,
    color=SUCCESS,
    label="Precision-Recall Curve"
)

ax.fill_between(
    recall,
    precision,
    alpha=0.20,
    color=SUCCESS
)

ax.set_title(
    "Precision Recall Curve",
    fontsize=18,
    fontweight="bold",
    pad=20
)

ax.set_xlabel(
    "Recall",
    fontsize=13
)

ax.set_ylabel(
    "Precision",
    fontsize=13
)

ax.legend()

ax.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "07_precision_recall_curve.png",
    dpi=600
)

plt.close()

# ==========================================================
# THRESHOLD OPTIMIZATION
# ==========================================================

print("Generating Threshold Optimization...")

thresholds = np.arange(
    0.05,
    0.96,
    0.01
)

f1_scores = []

for threshold in thresholds:

    preds = (
        y_prob >= threshold
    ).astype(int)

    score = f1_score(
        y_eval,
        preds
    )

    f1_scores.append(
        score
    )

f1_scores = np.array(
    f1_scores
)

best_idx = np.argmax(
    f1_scores
)

best_threshold = thresholds[
    best_idx
]

best_f1 = f1_scores[
    best_idx
]

fig, ax = plt.subplots(
    figsize=(10, 8)
)

ax.plot(
    thresholds,
    f1_scores,
    linewidth=3,
    color=WARNING
)

ax.scatter(
    best_threshold,
    best_f1,
    s=150,
    color=DANGER,
    zorder=5
)

ax.annotate(
    f"Best Threshold = {best_threshold:.2f}\nF1 = {best_f1:.4f}",
    (
        best_threshold,
        best_f1
    ),
    xytext=(20,20),
    textcoords="offset points",
    bbox=dict(
        boxstyle="round",
        fc="white"
    )
)

ax.set_title(
    "Threshold Optimization Analysis",
    fontsize=18,
    fontweight="bold",
    pad=20
)

ax.set_xlabel(
    "Decision Threshold"
)

ax.set_ylabel(
    "F1 Score"
)

ax.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "08_threshold_optimization.png",
    dpi=600
)

plt.close()

# ==========================================================
# PROBABILITY DISTRIBUTION
# ==========================================================

print("Generating Probability Distribution...")

normal_probs = y_prob[
    y_eval == 0
]

attack_probs = y_prob[
    y_eval == 1
]

fig, ax = plt.subplots(
    figsize=(11,8)
)

ax.hist(
    normal_probs,
    bins=40,
    alpha=0.65,
    color=SUCCESS,
    density=True,
    label="Normal Traffic"
)

ax.hist(
    attack_probs,
    bins=40,
    alpha=0.65,
    color=DANGER,
    density=True,
    label="Attack Traffic"
)

ax.axvline(
    0.5,
    color="black",
    linestyle="--",
    linewidth=3,
    label="Threshold"
)

ax.set_title(
    "Prediction Probability Distribution",
    fontsize=18,
    fontweight="bold",
    pad=20
)

ax.set_xlabel(
    "Predicted Attack Probability"
)

ax.set_ylabel(
    "Density"
)

ax.legend()

ax.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "09_probability_distribution.png",
    dpi=600
)

plt.close()

# ==========================================================
# PERFORMANCE DASHBOARD
# ==========================================================

print("Generating Performance Dashboard...")

accuracy = accuracy_score(
    y_eval,
    y_pred
)

precision_metric = precision_score(
    y_eval,
    y_pred
)

recall_metric = recall_score(
    y_eval,
    y_pred
)

f1_metric = f1_score(
    y_eval,
    y_pred
)

performance_df = pd.DataFrame({
    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ],
    "Value":[
        accuracy,
        precision_metric,
        recall_metric,
        f1_metric,
        roc_auc
    ]
})

fig, ax = plt.subplots(
    figsize=(10,4)
)

ax.axis("off")

display_df = performance_df.copy()

display_df["Value"] = display_df["Value"].apply(
    lambda x: f"{x:.6f}"
)

table = ax.table(
    cellText=display_df.values.tolist(),
    colLabels=display_df.columns.tolist(),
    cellLoc="center",
    loc="center"
)
table.auto_set_font_size(False)

table.set_fontsize(13)

table.scale(
    1.5,
    2.2
)

plt.title(
    "Model Performance Summary",
    fontsize=18,
    fontweight="bold",
    pad=20
)

plt.savefig(
    RESULTS_DIR /
    "10_performance_dashboard.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("\nModel Evaluation Figures Complete")

# ==========================================================
# PART 3 : SHAP EXPLAINABILITY ANALYSIS
# ==========================================================

import shap
import json

print("\n" + "=" * 70)
print("PART 3 - SHAP EXPLAINABILITY")
print("=" * 70)

with open(
    "data/tensors/feature_schema.json",
    "r"
) as f:
    schema = json.load(f)

feature_names = schema["feature_names"]

print("Creating SHAP background dataset...")

BACKGROUND_SIZE = 30
EXPLAIN_SIZE = 20

background_idx = np.random.choice(
    len(X_eval),
    BACKGROUND_SIZE,
    replace=False
)

explain_idx = np.random.choice(
    len(X_eval),
    EXPLAIN_SIZE,
    replace=False
)

background = X_eval[background_idx]
samples = X_eval[explain_idx]

print("Building SHAP DeepExplainer...")

print("\nSHAP DEBUG")
print("Background Shape:", background.shape)
print("Explain Shape:", samples.shape)
print("Model Input:", model.input_shape)

print("Building SHAP KernelExplainer...")

background_flat = background.reshape(
    background.shape[0],
    -1
)

samples_flat = samples.reshape(
    samples.shape[0],
    -1
)

def predict_fn(x):

    x = x.reshape(
        x.shape[0],
        WINDOW_SIZE,
        NUM_FEATURES
    )

    return model.predict(
        x,
        verbose=0
    )

explainer = shap.KernelExplainer(
    predict_fn,
    background_flat[:20]
)

print("Computing SHAP values...")

shap_values = explainer.shap_values(
    samples_flat[:20],
    nsamples=100
)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(
    shap_values
)

print(
    "SHAP Shape:",
    shap_values.shape
)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values)

print("Raw SHAP Shape:", shap_values.shape)

if len(shap_values.shape) == 4:
    shap_values = shap_values[:, :, :, 0]

print("Processed SHAP Shape:", shap_values.shape)
# ==========================================================
# TEMPORAL FEATURE AGGREGATION
# ==========================================================

print("Aggregating temporal feature importance...")

feature_importance_flat = np.mean(
    np.abs(shap_values),
    axis=0
)

feature_importance = (
    feature_importance_flat
    .reshape(
        WINDOW_SIZE,
        NUM_FEATURES
    )
    .mean(axis=0)
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

importance_df.to_csv(
    RESULTS_DIR /
    "10_feature_importance_table.csv",
    index=False
)

# ==========================================================
# TOP 15 FEATURES BAR CHART
# ==========================================================

top15 = importance_df.head(15)

fig, ax = plt.subplots(
    figsize=(12, 8)
)

bars = ax.barh(
    top15["Feature"][::-1],
    top15["Importance"][::-1],
    color="#2563EB"
)

ax.set_title(
    "Top 15 Most Influential Features",
    fontsize=18,
    fontweight="bold",
    pad=20
)

ax.set_xlabel(
    "Mean Absolute SHAP Value",
    fontsize=12
)

ax.grid(
    axis="x",
    alpha=0.3
)

for bar in bars:

    width = bar.get_width()

    ax.text(
        width,
        bar.get_y() + bar.get_height()/2,
        f"{width:.4f}",
        va="center",
        fontsize=10
    )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "11_top15_features.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# FEATURE IMPORTANCE TABLE IMAGE
# ==========================================================

fig, ax = plt.subplots(
    figsize=(12, 7)
)

ax.axis("off")

top15_display = top15.copy()

top15_display["Importance"] = (
    top15_display["Importance"]
    .round(6)
    .astype(str)
)

table = ax.table(
    cellText=top15_display.values.tolist(),
    colLabels=top15_display.columns.tolist(),
    cellLoc="center",
    loc="center"
)   

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.4, 2.0)

plt.title(
    "Top 15 Feature Importance Ranking",
    fontsize=18,
    fontweight="bold",
    pad=25
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "12_feature_importance_table.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# SHAP SUMMARY STYLE PLOT
# ==========================================================

print("Generating SHAP Summary Plot...")

summary_importance = (
    importance_df
    .head(20)
)

fig, ax = plt.subplots(
    figsize=(12, 8)
)

import matplotlib.cm as cm

colors = cm.get_cmap("Blues")(
    np.linspace(
        0.40,
        0.95,
        len(summary_importance)
    )
)

ax.barh(
    summary_importance["Feature"][::-1],
    summary_importance["Importance"][::-1],
    color=colors
)

ax.set_title(
    "Global SHAP Feature Importance",
    fontsize=18,
    fontweight="bold",
    pad=20
)

ax.set_xlabel(
    "Average Impact on Model Output",
    fontsize=12
)

ax.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "13_shap_summary.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# SINGLE SAMPLE WATERFALL APPROXIMATION
# ==========================================================

print("Generating Waterfall Plot...")

sample_shap = (
    shap_values[0]
    .reshape(
        WINDOW_SIZE,
        NUM_FEATURES
    )
    .mean(axis=0)
)

waterfall_df = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": sample_shap
})

waterfall_df = (
    waterfall_df
    .reindex(
        waterfall_df["Contribution"]
        .abs()
        .sort_values(
            ascending=False
        )
        .index
    )
    .head(12)
)

fig, ax = plt.subplots(
    figsize=(12, 8)
)

colors = [
    "#DC2626"
    if v > 0
    else "#2563EB"
    for v in waterfall_df["Contribution"]
]

ax.barh(
    waterfall_df["Feature"][::-1],
    waterfall_df["Contribution"][::-1],
    color=colors[::-1]
)

ax.axvline(
    0,
    color="black",
    linewidth=1
)

ax.set_title(
    "Local Explanation (Waterfall Style)",
    fontsize=18,
    fontweight="bold",
    pad=20
)

ax.set_xlabel(
    "Contribution to Prediction",
    fontsize=12
)

ax.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "14_shap_waterfall.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("\nSHAP analysis complete.")