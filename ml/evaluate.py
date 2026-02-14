"""
Explainable AI (SHAP) for Intrusion Detection System
Provides global and local explanations
"""

import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

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

X = df.drop(columns=["binary_label", "multiclass_label"])
y = df["binary_label"]

model = joblib.load(MODEL_PATH)

print("Model and data loaded.")


# -----------------------------
# SHAP Explainer
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("SHAP values computed.")


# -----------------------------
# Global Feature Importance
# -----------------------------
plt.figure()
shap.summary_plot(
    shap_values[1],
    X,
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "global_feature_importance.png"))
plt.close()

print("Global feature importance saved.")


# -----------------------------
# Local Explanation (Single Sample)
# -----------------------------
sample_index = 10  # change to inspect other samples

plt.figure()
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][sample_index],
    X.iloc[sample_index],
    matplotlib=True,
    show=False
)
plt.savefig(os.path.join(OUTPUT_DIR, "local_explanation_sample.png"))
plt.close()

print("Local explanation saved.")


print("=== SHAP EXPLAINABILITY COMPLETED ===")
