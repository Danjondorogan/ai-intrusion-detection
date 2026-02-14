import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix

import xgboost as xgb

# --------------------------
# Files
# --------------------------
train_files = [
    "data/final/tuesday_final.csv",
    "data/final/wednesday_final.csv"
]

val_file = "data/final/thursday_final.csv"
label_col = "label_binary"

# --------------------------
# Load training data (for model)
# --------------------------
train_df = pd.concat(
    [pd.read_csv(f) for f in train_files],
    ignore_index=True
)

X_train = train_df.drop(columns=[label_col])
y_train = train_df[label_col]

# --------------------------
# Load validation data
# --------------------------
val_df = pd.read_csv(val_file)
X_val = val_df.drop(columns=[label_col])
y_val = val_df[label_col]

# --------------------------
# Compute class weight
# --------------------------
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

# --------------------------
# Train model (same as before)
# --------------------------
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------
# Get probabilities
# --------------------------
probs = model.predict_proba(X_val)[:, 1]

# --------------------------
# Threshold sweep
# --------------------------
thresholds = np.arange(0.05, 0.96, 0.05)

results = []

for t in thresholds:
    preds = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    results.append({
        "threshold": round(t, 2),
        "recall_attack": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "alerts_total": int(fp + tp)
    })

# --------------------------
# Display results
# --------------------------
results_df = pd.DataFrame(results)
print(results_df)
