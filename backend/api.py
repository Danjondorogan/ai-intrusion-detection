from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import tensorflow as tf
import os
import logging

MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "dos_lstm_final.keras")
)

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 10))
FEATURES_PER_STEP = int(os.environ.get("FEATURES_PER_STEP", 86))
FEATURE_COUNT = WINDOW_SIZE * FEATURES_PER_STEP

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8000",
]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend.api")

app = FastAPI(title="AI Intrusion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    session_id: Optional[str] = "anonymous"
    features: List[float]

class PredictResponse(BaseModel):
    session_id: str
    prediction: int
    dos_probability: float
    status: str
    required_timesteps: int
    timesteps_collected: int

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

try:
    load_model_safe(MODEL_PATH)
except Exception as e:
    log.warning(f"Model not loaded at startup: {e}")

@app.get("/health")
async def health():
    return {"status": "API running"}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    features = req.features

    if not isinstance(features, list):
        raise HTTPException(status_code=422, detail="features must be a list of numbers")

    if len(features) != FEATURE_COUNT:
        raise HTTPException(
            status_code=422,
            detail=f"features must be length {FEATURE_COUNT} (got {len(features)})",
        )

    try:
        model = load_model_safe(MODEL_PATH)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    try:
        x = np.array(features, dtype=np.float32).reshape((1, WINDOW_SIZE, FEATURES_PER_STEP))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"feature reshape failed: {e}")

    try:
        preds = model.predict(x, verbose=0)

        if preds.ndim == 2 and preds.shape[1] == 1:
            prob = float(preds[0, 0])
        elif preds.ndim == 2 and preds.shape[1] >= 2:
            prob = float(preds[0, 1])
        else:
            prob = float(np.array(preds).reshape(-1)[0])

        pred_label = 1 if prob >= 0.5 else 0

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"model prediction failed: {e}")

    response = {
        "session_id": req.session_id or "anonymous",
        "prediction": pred_label,
        "dos_probability": round(prob, 6),
        "status": "attack" if pred_label == 1 else "normal",
        "required_timesteps": WINDOW_SIZE,
        "timesteps_collected": WINDOW_SIZE,
    }
    return response

@app.get("/shap/global")
async def shap_global():
    art = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "validation",
            "artifacts",
            "shap_global_top20.csv",
        )
    )
    if os.path.exists(art):
        import csv
        with open(art, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows

    raise HTTPException(status_code=404, detail="shap global not found")
