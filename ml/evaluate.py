"""
Explainable AI (SHAP) for Intrusion Detection System
Provides global and local explanations
"""

import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

print("=== SHAP EXPLAINABILITY STARTED ===")

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/processed/processed_flows.csv"
MODEL_PATH = "ml/models/rf_ids_model.joblib"
OUTPUT_DIR = "docs/shap_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load data and model
# -----------------------------
df = pd.read_csv(DATA_PATH)

if "binary_label" not in df.columns:
    raise RuntimeError("binary_label column missing")

X = df.drop(columns=["binary_label", "multiclass_label"], errors="ignore")
y = df["binary_label"]

model = joblib.load(MODEL_PATH)

print("[INFO] Model and data loaded")
print(f"[INFO] Data shape: {X.shape}")

# -----------------------------
# SHAP Explainer
# -----------------------------
explainer = shap.TreeExplainer(model)

# Sample for performance
X_sample = X.sample(n=min(2000, len(X)), random_state=42)

# 🔥 FIX: Align y with sampled X (CRITICAL)
y_sample = y.loc[X_sample.index]

shap_values = explainer.shap_values(X_sample)

# -----------------------------
# SAFE HANDLING
# -----------------------------
if isinstance(shap_values, list):
    shap_values_to_use = shap_values[1]
else:
    shap_values_to_use = shap_values

shap_values_to_use = np.array(shap_values_to_use)

# Fix expected value
expected_value = explainer.expected_value

if isinstance(expected_value, (list, np.ndarray)):
    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

expected_value = float(np.array(expected_value).reshape(-1)[0])

print("[INFO] SHAP values computed")

# -----------------------------
# Global Feature Importance
# -----------------------------
print("[INFO] Generating global feature importance plot")

plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values_to_use,
    X_sample,
    plot_type="bar",
    max_display=6,
    show=False
)

plt.title("Global Feature Importance (SHAP)", fontsize=14)
plt.xlabel("Mean |SHAP Value| (Impact on Model Output)")
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "global_feature_importance.png"), dpi=300)
plt.close()

print("[INFO] Global feature importance saved")

# -----------------------------
# Local Explanation (Single Sample)
# -----------------------------
print("[INFO] Generating local explanation")

# 🔥 FIX: Use aligned y_sample
attack_indices = np.where(y_sample.values == 1)[0]

if len(attack_indices) > 0:
    sample_index = attack_indices[0]
else:
    sample_index = 0

single_shap = shap_values_to_use[sample_index]

# Handle multi-output
if len(single_shap.shape) == 2:
    single_shap = single_shap[:, 1]

single_data = X_sample.iloc[sample_index]

# 🔥 FINAL FIX: Ensure correct length
feature_names = list(X_sample.columns)

min_len = min(len(single_shap), len(feature_names))

single_shap = single_shap[:min_len]
single_data = single_data.iloc[:min_len]
feature_names = feature_names[:min_len]

plt.figure(figsize=(10, 6))

shap.plots.waterfall(
    shap.Explanation(
        values=single_shap,
        base_values=expected_value,
        data=single_data.values,
        feature_names=[str(f) for f in feature_names]
    ),
    max_display=6,
    show=False
)

plt.title("SHAP Local Explanation (Attack Sample)", fontsize=14)
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "local_explanation_sample.png"), dpi=300)
plt.close()

print("[INFO] Local explanation saved")

print("=== SHAP EXPLAINABILITY COMPLETED ===")