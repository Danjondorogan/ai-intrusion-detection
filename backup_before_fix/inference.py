import json
import time
import logging
import numpy as np
import joblib
import tensorflow as tf

from pathlib import Path
from collections import deque
from typing import Dict, Any, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("OnlineLSTMInference")

MODEL_PATH = Path("models/dos_lstm_final.keras")
SCALER_PATH = Path("data/final/standard_scaler.joblib")
SCHEMA_PATH = Path("data/tensors/feature_schema.json")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")

if not SCHEMA_PATH.exists():
    raise FileNotFoundError(f"Schema not found at {SCHEMA_PATH}")

logger.info("Loading LSTM model")
model = tf.keras.models.load_model(MODEL_PATH)

logger.info("Loading scaler")
scaler = joblib.load(SCALER_PATH)

with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)

def resolve_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    if "window_size" not in schema:
        raise RuntimeError("Schema missing window_size")

    window_size = int(schema["window_size"])

    if "num_features" in schema:
        num_features = int(schema["num_features"])
    elif "features_per_timestep" in schema:
        num_features = int(schema["features_per_timestep"])
    else:
        raise RuntimeError("Schema missing feature count")

    flattened = schema.get("flattened_features", window_size * num_features)

    feature_columns = (
        schema.get("feature_columns")
        or schema.get("temporal_feature_columns")
        or schema.get("feature_names")
    )

    return {
        "window_size": window_size,
        "num_features": num_features,
        "flattened_features": flattened,
        "feature_columns": feature_columns,
    }

resolved = resolve_schema(schema)

WINDOW_SIZE = resolved["window_size"]
NUM_FEATURES = resolved["num_features"]
FLATTENED_FEATURES = resolved["flattened_features"]
FEATURE_COLUMNS = resolved["feature_columns"]

logger.info(f"Window size: {WINDOW_SIZE}")
logger.info(f"Features per timestep: {NUM_FEATURES}")

class TemporalBuffer:
    def __init__(self, window_size: int, num_features: int):
        self.window_size = window_size
        self.num_features = num_features
        self.buffer = deque(maxlen=window_size)

    def reset(self) -> None:
        self.buffer.clear()

    def add(self, vector: np.ndarray) -> None:
        if vector.shape != (self.num_features,):
            raise ValueError(
                f"Expected ({self.num_features},) got {vector.shape}"
            )
        self.buffer.append(vector)

    def is_ready(self) -> bool:
        return len(self.buffer) == self.window_size

    def size(self) -> int:
        return len(self.buffer)

    def get_tensor(self) -> np.ndarray:
        if not self.is_ready():
            raise RuntimeError("Temporal buffer not full")
        return np.array(self.buffer, dtype=np.float32).reshape(
            1, self.window_size, self.num_features
        )

class OnlineLSTMInference:
    PROB_THRESHOLD = 0.5
    REQUIRED_CONSECUTIVE_DETECTIONS = 3

    def __init__(self):
        self.buffer = TemporalBuffer(WINDOW_SIZE, NUM_FEATURES)
        self.consecutive_detections = 0
        self.total_predictions = 0
        self._last_lstm_tensor: Optional[np.ndarray] = None
        self._last_probability: Optional[float] = None
        self._last_timestamp: Optional[float] = None

    def preprocess(self, raw_vector: np.ndarray) -> np.ndarray:
        if raw_vector.shape != (NUM_FEATURES,):
            raise ValueError(
                f"Expected ({NUM_FEATURES},) got {raw_vector.shape}"
            )
        X = raw_vector.reshape(1, -1)
        X_scaled = scaler.transform(X)
        return X_scaled.flatten()

    def _severity(self, p: float) -> str:
        if p < 0.30:
            return "NORMAL"
        if p < 0.50:
            return "SUSPICIOUS"
        if p < 0.70:
            return "ATTACK_LOW"
        if p < 0.90:
            return "ATTACK_MEDIUM"
        return "ATTACK_CRITICAL"

    def predict(self, raw_vector: np.ndarray) -> Dict[str, Any]:
        start = time.time()
        self.total_predictions += 1

        scaled = self.preprocess(raw_vector)
        self.buffer.add(scaled)

        if not self.buffer.is_ready():
            return {
                "status": "warming_up",
                "timesteps_collected": self.buffer.size(),
                "required_timesteps": WINDOW_SIZE,
            }

        X_lstm = self.buffer.get_tensor()
        probability = float(model.predict(X_lstm, verbose=0)[0][0])

        self._last_lstm_tensor = X_lstm.copy()
        self._last_probability = probability
        self._last_timestamp = time.time()

        if probability >= self.PROB_THRESHOLD:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0

        confirmed = (
            self.consecutive_detections >= self.REQUIRED_CONSECUTIVE_DETECTIONS
        )

        latency = (time.time() - start) * 1000.0

        return {
            "status": "attack" if confirmed else "monitoring",
            "severity": self._severity(probability),
            "dos_probability": probability,
            "prediction": int(confirmed),
            "consecutive_detections": self.consecutive_detections,
            "total_predictions": self.total_predictions,
            "latency_ms": round(latency, 3),
        }

    def get_temporal_tensor(self) -> np.ndarray:
        if self._last_lstm_tensor is None:
            raise RuntimeError("No completed inference available")
        return self._last_lstm_tensor

    def get_last_probability(self) -> float:
        if self._last_probability is None:
            raise RuntimeError("No completed inference available")
        return float(self._last_probability)

    def reset(self) -> None:
        self.buffer.reset()
        self.consecutive_detections = 0
        self.total_predictions = 0
        self._last_lstm_tensor = None
        self._last_probability = None
        self._last_timestamp = None

if __name__ == "__main__":
    engine = OnlineLSTMInference()
    dummy = np.zeros(NUM_FEATURES, dtype=np.float32)
    for i in range(WINDOW_SIZE + 3):
        print(engine.predict(dummy))
