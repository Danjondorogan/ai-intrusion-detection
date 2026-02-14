import json
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path
from collections import deque
from typing import Dict, Any

# ======================================================
# PATH CONFIG
# ======================================================
MODEL_PATH = Path("models/dos_lstm_final.keras")
SCALER_PATH = Path("data/final/standard_scaler.joblib")
SCHEMA_PATH = Path("data/tensors/feature_schema.json")

# ======================================================
# LOAD MODEL
# ======================================================
print("[INFO] Loading trained LSTM model")
model = tf.keras.models.load_model(MODEL_PATH)

# ======================================================
# LOAD SCALER
# ======================================================
print("[INFO] Loading StandardScaler")
scaler = joblib.load(SCALER_PATH)

# ======================================================
# LOAD FEATURE SCHEMA
# ======================================================
print("[INFO] Loading feature schema")
with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)

# ======================================================
# REQUIRED KEYS (MATCH ACTUAL SCHEMA)
# ======================================================
required_keys = [
    "window_size",
    "num_features",
    "flattened_features",
]

for key in required_keys:
    if key not in schema:
        raise RuntimeError(f"Schema missing required key: {key}")

WINDOW_SIZE = int(schema["window_size"])
NUM_FEATURES = int(schema["num_features"])
FLATTENED_FEATURES = int(schema["flattened_features"])

print(f"[INFO] Window size: {WINDOW_SIZE}")
print(f"[INFO] Features per timestep: {NUM_FEATURES}")
print(f"[INFO] Flattened features: {FLATTENED_FEATURES}")

# ======================================================
# TEMPORAL BUFFER
# ======================================================
class TemporalBuffer:
    def __init__(self, window_size: int, num_features: int):
        self.window_size = window_size
        self.num_features = num_features
        self.buffer = deque(maxlen=window_size)

    def add(self, vector: np.ndarray):
        if vector.shape != (self.num_features,):
            raise ValueError(
                f"Feature shape mismatch: expected ({self.num_features},), "
                f"got {vector.shape}"
            )
        self.buffer.append(vector)

    def is_ready(self) -> bool:
        return len(self.buffer) == self.window_size

    def get_tensor(self) -> np.ndarray:
        if not self.is_ready():
            raise RuntimeError("Temporal buffer not full yet")
        return np.array(self.buffer, dtype=np.float32).reshape(
            1, self.window_size, self.num_features
        )

# ======================================================
# ONLINE INFERENCE ENGINE
# ======================================================
class OnlineLSTMInference:
    def __init__(self):
        self.buffer = TemporalBuffer(WINDOW_SIZE, NUM_FEATURES)

    def preprocess(self, raw_vector: np.ndarray) -> np.ndarray:
        if raw_vector.shape != (NUM_FEATURES,):
            raise ValueError(
                f"Expected raw feature vector of shape ({NUM_FEATURES},), "
                f"got {raw_vector.shape}"
            )

        X = raw_vector.reshape(1, -1)
        X_scaled = scaler.transform(X)
        return X_scaled.flatten()

    def predict(self, raw_vector: np.ndarray) -> Dict[str, Any]:
        scaled_vector = self.preprocess(raw_vector)
        self.buffer.add(scaled_vector)

        if not self.buffer.is_ready():
            return {
                "status": "warming_up",
                "timesteps_collected": len(self.buffer.buffer),
                "required_timesteps": WINDOW_SIZE,
            }

        X_lstm = self.buffer.get_tensor()
        probability = float(model.predict(X_lstm, verbose=0)[0][0])

        return {
            "status": "ready",
            "dos_probability": probability,
            "prediction": int(probability >= 0.5),
        }

# ======================================================
# SAFE CLI TEST
# ======================================================
if __name__ == "__main__":
    print("\n[TEST] Running safe online inference test")

    engine = OnlineLSTMInference()
    dummy_vector = np.zeros(NUM_FEATURES, dtype=np.float32)

    for i in range(WINDOW_SIZE + 3):
        result = engine.predict(dummy_vector)
        print(f"[STEP {i}] → {result}")
