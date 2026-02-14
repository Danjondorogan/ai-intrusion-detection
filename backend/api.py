# backend/api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist
from typing import List, Optional
import numpy as np
import tensorflow as tf
import os
import logging

# -------- CONFIG ----------
# Either set environment variable MODEL_PATH, or it defaults to repo-level models/...
MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "dos_lstm_final.keras")
)

# Number of time-steps and features expected by the model
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 10))
FEATURE_COUNT = int(os.environ.get("FEATURE_COUNT", 86))  # adjust if needed

# Allowed CORS origins (add your frontend origin)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8000",
]

# -------- LOGGING ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend.api")

# -------- APP & CORS ----------
app = FastAPI(title="AI Intrusion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],  # for dev use; tighten for prod
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# -------- Pydantic models ----------
class PredictRequest(BaseModel):
    session_id: Optional[str] = "anonymous"
    # Ensure we only accept lists of floats. We can't enforce the exact length here
    # with conlist because frontend may send different sized vectors; we'll validate server-side.
    features: List[float]

class PredictResponse(BaseModel):
    session_id: str
    prediction: int
    dos_probability: float
    status: str
    required_timesteps: int
    timesteps_collected: int

# -------- MODEL LOADING ----------
MODEL = None

def load_model_safe(path: str):
    global MODEL
    if MODEL is not None:
        return MODEL
    if not os.path.exists(path):
        msg = f"Model file not found at: {path}"
        log.error(msg)
        raise FileNotFoundError(msg)
    log.info(f"Loading model from: {path}")
    MODEL = tf.keras.models.load_model(path)
    log.info("Model loaded successfully")
    return MODEL

# Try to load at import time (non-blocking if missing)
try:
    load_model_safe(MODEL_PATH)
except Exception as e:
    log.warning(f"Model not loaded at startup: {e}")

# -------- ROUTES ----------
@app.get("/health")
async def health():
    return {"status": "API running"}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    # 1) validate features length
    features = req.features
    if not isinstance(features, list):
        raise HTTPException(status_code=422, detail="features must be a list of numbers")
    if len(features) != FEATURE_COUNT:
        raise HTTPException(
            status_code=422,
            detail=f"features must be length {FEATURE_COUNT} (got {len(features)})",
        )

    # 2) ensure model loaded
    try:
        model = load_model_safe(MODEL_PATH)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    # 3) prepare data for prediction
    try:
        x = np.array(features, dtype=np.float32).reshape((1, WINDOW_SIZE, FEATURE_COUNT // WINDOW_SIZE))
        # Note: depends on how you flattened features; if features is shape (WINDOW_SIZE * num_features),
        # set FEATURE_COUNT = WINDOW_SIZE * num_features and reshape accordingly:
        x = np.array(features, dtype=np.float32).reshape((1, WINDOW_SIZE, FEATURE_COUNT // WINDOW_SIZE))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"feature reshape failed: {e}")

    # 4) model predict
    try:
        preds = model.predict(x, verbose=0)
        # If model outputs a single probability:
        if preds.ndim == 2 and preds.shape[1] >= 2:
            # multiclass - take probability of class 1
            prob = float(preds[0, 1])
            pred_label = int(np.argmax(preds, axis=1)[0])
        else:
            # single-output prob
            prob = float(preds.flatten()[0])
            pred_label = int(prob >= 0.5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"model prediction failed: {e}")

    # 5) build response
    required_timesteps = WINDOW_SIZE
    timesteps_collected = WINDOW_SIZE  # adapt if streaming buffer used

    status = "attack" if prob >= 0.5 else "normal"
    response = {
        "session_id": req.session_id or "anonymous",
        "prediction": pred_label,
        "dos_probability": round(prob, 6),
        "status": status,
        "required_timesteps": required_timesteps,
        "timesteps_collected": timesteps_collected,
    }
    return response

# Optional: endpoint to get SHAP/global/time files
@app.get("/shap/global")
async def shap_global():
    # return the CSV or JSON from artifacts if you have them in validation/artifacts
    art = os.path.abspath(os.path.join(os.path.dirname(__file__), "validation", "artifacts", "shap_global_top20.csv"))
    if os.path.exists(art):
        import csv, json
        with open(art, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows
    raise HTTPException(status_code=404, detail="shap global not found")