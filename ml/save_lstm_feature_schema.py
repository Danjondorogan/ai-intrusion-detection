import numpy as np
import json
from pathlib import Path

DATA_DIR = Path("data/attack_families/dos")
OUT_PATH = DATA_DIR / "lstm_feature_schema.json"

X = np.load(DATA_DIR / "X_lstm.npy")

num_timesteps = X.shape[1]
num_features = X.shape[2]

schema = {
    "window_size": num_timesteps,
    "num_features": num_features,
    "description": "LSTM temporal feature contract – DO NOT CHANGE"
}

with open(OUT_PATH, "w") as f:
    json.dump(schema, f, indent=2)

print("Saved LSTM feature schema →", OUT_PATH)
print(schema)
