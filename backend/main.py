from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Dict, List, Optional
import threading
import numpy as np
import time

from backend.inference import OnlineLSTMInference
from backend.shap_explainer import explain_lstm_decision


app = FastAPI(
    title="AI Intrusion Detection System",
    description="Online LSTM-based DoS Detection with Stateful Explainability",
    version="1.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SESSION_ENGINES: Dict[str, OnlineLSTMInference] = {}
SESSION_LAST_TENSOR: Dict[str, np.ndarray] = {}
SESSION_LAST_UPDATE: Dict[str, float] = {}

_lock = threading.Lock()


class PredictRequest(BaseModel):
    session_id: str
    features: List[float]


class PredictResponse(BaseModel):
    status: str
    dos_probability: Optional[float] = None
    prediction: Optional[int] = None
    timesteps_collected: Optional[int] = None
    required_timesteps: Optional[int] = None
    consecutive_detections: Optional[int] = None
    latency_ms: Optional[float] = None


class ExplainResponse(BaseModel):
    session_id: str
    skipped: Optional[bool] = None
    reason: Optional[str] = None
    dos_probability: Optional[float] = None
    plot_path: Optional[str] = None
    temporal_plot_path: Optional[str] = None
    top_features: Optional[List[dict]] = None
    window_size: Optional[int] = None
    num_features: Optional[int] = None
    explanation_type: Optional[str] = None


@app.on_event("startup")
def startup_event():
    print("[INFO] FastAPI server started")
    print("[INFO] Inference engine ready")
    print("[INFO] Explainability engine ready")


@app.on_event("shutdown")
def shutdown_event():
    with _lock:
        SESSION_ENGINES.clear()
        SESSION_LAST_TENSOR.clear()
        SESSION_LAST_UPDATE.clear()
    print("[INFO] Server shutdown, sessions cleared")


@app.options("/{path:path}")
async def options_handler(path: str):
    return Response(status_code=200)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    if len(request.features) != 84:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 84 features, got {len(request.features)}",
        )

    try:
        raw_vector = np.asarray(request.features, dtype=np.float32)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid feature values")

    with _lock:
        if request.session_id not in SESSION_ENGINES:
            SESSION_ENGINES[request.session_id] = OnlineLSTMInference()
        engine = SESSION_ENGINES[request.session_id]

    try:
        result = engine.predict(raw_vector)

        if engine.buffer.is_ready():
            SESSION_LAST_TENSOR[request.session_id] = engine.get_temporal_tensor()
            SESSION_LAST_UPDATE[request.session_id] = time.time()

        return result

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/explain/{session_id}", response_model=ExplainResponse)
async def explain(session_id: str):
    with _lock:
        if session_id not in SESSION_ENGINES:
            raise HTTPException(status_code=404, detail="Session not found")

        if session_id not in SESSION_LAST_TENSOR:
            raise HTTPException(
                status_code=400,
                detail="No completed inference available for this session",
            )

        temporal_tensor = SESSION_LAST_TENSOR[session_id]

    try:
        explanation = explain_lstm_decision(
            temporal_tensor=temporal_tensor,
            session_id=session_id,
            max_samples=150,
            top_k=10,
        )
        return explanation

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/sessions")
async def list_sessions():
    with _lock:
        return {
            "active_sessions": list(SESSION_ENGINES.keys()),
            "explainable_sessions": list(SESSION_LAST_TENSOR.keys()),
        }


@app.post("/reset/{session_id}")
async def reset_session(session_id: str):
    with _lock:
        if session_id not in SESSION_ENGINES:
            raise HTTPException(status_code=404, detail="Session not found")

        SESSION_ENGINES[session_id].reset()
        SESSION_LAST_TENSOR.pop(session_id, None)
        SESSION_LAST_UPDATE.pop(session_id, None)

    return {"status": "reset", "session_id": session_id}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sessions": len(SESSION_ENGINES),
        "explainable": len(SESSION_LAST_TENSOR),
    }
