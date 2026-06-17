import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

print("Loading data...")

df = pd.read_csv("data/processed/monday_processed.csv")

# Remove label
X = df.drop(columns=["label_binary"])

# Same cleaning used in training pipeline
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(numeric_only=True), inplace=True)

print("Features:", X.shape[1])

scaler = StandardScaler()
scaler.fit(X)

joblib.dump(
    scaler,
    "data/final/standard_scaler.joblib"
)

print("Scaler rebuilt successfully")
print("n_features_in_ =", scaler.n_features_in_)