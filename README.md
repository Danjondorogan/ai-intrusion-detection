# AI Intrusion Detection System (DoS detection)

**Repository:** https://github.com/Danjondorogan/ai-intrusion-detection

**Short summary**

This project is a research-grade AI Intrusion Detection System (IDS) that detects Denial-of-Service (DoS) attacks using a trained LSTM neural network. It includes an ML pipeline for preparing data and training, a FastAPI backend for real-time inference, a React + TypeScript frontend dashboard for visualization, and SHAP-based explainability for model decisions.

# Table of contents

1. [Project overview](#project-overview)  
2. [Repository structure](#repository-structure)  
3. [Model details](#model-details)  
4. [Backend API (FastAPI)](#backend-api-fastapi)  
5. [Frontend (React + TypeScript)](#frontend-react--typescript)  
6. [Local setup (Windows / PowerShell)](#local-setup-windows--powershell)  
7. [Run backend and test](#run-backend-and-test)  
8. [Run frontend (dev)](#run-frontend-dev)  
9. [SHAP explainability](#shap-explainability)  
10. [Retraining the model](#retraining-the-model)  
11. [Docker (optional)](#docker-optional)  
12. [CI / Tests / Publishing](#ci--tests--publishing)  
13. [Dataset used](#dataset-used)  
14. [Data & privacy](#data--privacy)  
15. [Troubleshooting](#troubleshooting)  
16. [Ethical use](#ethical-use)  
17. [Credits & license](#credits--license)

---

## Project overview

The system detects DoS attacks using a temporal LSTM model that looks at short windows of network flow features. The frontend sends flattened feature vectors (80 floats) to the backend. The backend shapes them into `(1, 10, 8)` and runs the model to return a probability and label.

Main goals:

- Real-time inference on incoming network flow features
- Clean API for dashboard integration
- Human-facing dashboard showing probability, traffic chart, logs, and per-feature explainability using SHAP
- Research-ready repository for publication

---

## Repository structure
ai-intrusion-detection/
├─ backend/
│ ├─ api.py # FastAPI endpoint(s)
│ ├─ inference.py # Model loading & prediction helpers
│ ├─ main.py # FastAPI app / startup
│ ├─ shap_explainer.py # SHAP helper (explainability)
│ ├─ requirements.txt # Backend requirements (optional)
│ └─ utils/
│ └─ ...
├─ frontend/
│ ├─ index.html
│ ├─ package.json
│ ├─ tsconfig.json
│ ├─ vite.config.ts
│ └─ src/
│ ├─ App.tsx
│ ├─ main.tsx
│ ├─ services/api.ts # API calls to backend
│ └─ components/ # UI components (Gauge, Chart, ControlPanel...)
├─ ml/
│ ├─ train_lstm.py
│ ├─ prepare_lstm_tensors.py
│ └─ evaluation/
├─ models/
│ └─ dos_lstm_final.keras # Saved model for inference
├─ data/
│ └─ sample/
├─ docs/
│ └─ report.pdf
├─ Dockerfile
├─ README.md # <-- you are here
└─ LICENSE


---

## Model details

- **Model type:** LSTM (binary classifier)  
- **Purpose:** Detect DoS from network flow features  
- **Saved model:** `models/dos_lstm_final.keras`  
- **Expected input shape to the model:** `(1, WINDOW_SIZE, FEATURES_PER_STEP)`  
  - `WINDOW_SIZE = 10`  
  - `FEATURES_PER_STEP = 8`  
- **Frontend flattened input:** `10 × 8 = 80` floats per prediction

Make sure the frontend always sends exactly **80 floats** in the expected order. The backend validates length and reshapes into `(1, 10, 8)`.

---

## Backend API (FastAPI)

**File:** `backend/api.py`

### Endpoint

**POST** `/predict`

**Request JSON**

```json
{
  "session_id": "string",
  "features": [ /* exactly 80 floats */ ]
}
```

Response JSON:
{
  "session_id": "string",
  "prediction": 0,
  "dos_probability": 0.123,
  "status": "normal" | "attack",
  "required_timesteps": 10,
  "timesteps_collected": 10
}

prediction: 0 = normal, 1 = attack
dos_probability: model probability (0.0 - 1.0)
status: "attack" or "normal"
required_timesteps: window size (10)
timesteps_collected: how many timesteps are currently buffered

Notes
The model is loaded once at startup for speed.
CORS is enabled during development. Configure allowed origins in production.
backend/inference.py provides helpers to validate, reshape, and call the model.

Frontend (React + TypeScript)

Located under frontend/.
src/services/api.ts contains the API client that calls POST /predict.
ControlPanel provides a button to send a single sample (80 features).

UI includes:
Probability gauge
Traffic chart (history of probabilities)
Metrics card (last prediction, latency)
Log viewer (session logs)
Explainability panel (SHAP)
During development, configure the frontend to use http://127.0.0.1:8000 as the backend base URL or set an environment variable for the backend host.

Local setup (Windows / PowerShell)
Important: Use Python 3.10 (TensorFlow compatibility). If you have multiple Python versions, prefer py -3.10.

1) Create & activate virtual environment
py -3.10 -m venv .venv
.\.venv\Scripts\Activate

2) Install Python dependencies (backend + ML)
# if backend/requirements.txt exists:
pip install -r backend/requirements.txt

# otherwise:
pip install fastapi uvicorn numpy pandas scikit-learn joblib shap matplotlib tensorflow==2.15 seaborn xgboost

3) Install Node dependencies (frontend)
cd frontend
npm install
cd ..

Run backend and test

Start backend: uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

Example test (PowerShell)

# prepare an 80-element test array (all zeros)
$features = @(for ($i=0; $i -lt 80; $i++) { 0.0 })
$body = @{
  session_id = "local-test-1"
  features = $features
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body

Run frontend (development)

From repo root:
cd frontend
npm run dev

Open the dev URL (usually http://localhost:5173) in the browser. Make sure backend is running.

SHAP explainability

SHAP code is in backend/shap_explainer.py.
It uses the trained model and a small background dataset to compute per-feature attributions.
For performance, compute SHAP offline or cache values; real-time SHAP can be slow.
Example endpoint: POST /explain (same body as /predict) → returns an array of 80 SHAP attribution values and optional top-k feature names.
Developer note: Use a small background set (e.g., 100 samples) or an approximate explainer for production.

Retraining the model

If you need to retrain:
Download the original CICIDS2017 dataset (see dataset section) and place raw CSV/XLSX in data/raw/.
Run ml/prepare_lstm_tensors.py to create tensors and the feature schema.
Train with ml/train_lstm.py (adjust hyperparameters at the top).
Save the trained model to models/dos_lstm_final.keras.
Commit the new model (keep file size reasonable; consider Git LFS if large).
Important: Keep the same input order, length (80), and scaling approach used in inference.

Docker (optional)

A basic Dockerfile is included. Example:
docker build -t ai-ids .
docker run -p 8000:8000 ai-ids
If you use Docker, ensure the model and any required small data files are included in the image or mounted as a volume.

CI / Tests / Publishing

Add GitHub Actions workflows for linting & unit tests.
Add backend/tests/test_api.py and frontend/src/__tests__ for basic checks.
Do NOT commit raw datasets. Use .gitignore to exclude data/raw/ and .venv/.

Dataset used

This project uses the CICIDS2017_improved dataset for training and evaluation.
The CICIDS2017 dataset is a commonly used benchmark for intrusion detection research. It contains realistic network traffic including both normal behavior and many attack scenarios (DoS, DDoS, brute force, port scans, botnets, web attacks). For this project, the dataset was cleaned and preprocessed into a form suitable for LSTM training.

Official dataset source:
https://www.unb.ca/cic/datasets/ids-2017.html

To reproduce training: download the original dataset, place raw files into data/raw/, and run the preprocessing scripts in ml/.

Data & privacy

Raw dataset files are not included in this repository because they are large and may contain sensitive network traffic. The repository includes:
small sample inputs for testing (data/sample/)
feature schema (tensors/feature_schema.json)
the trained model (models/dos_lstm_final.keras)

Follow your institution's policies and local laws when working with real network traffic. Anonymize or remove sensitive fields before sharing.

Troubleshooting
ImportError: No module named 'tensorflow'
→ Use Python 3.10 and reinstall dependencies: py -3.10 -m venv .venv then pip install ....
422 Unprocessable Entity from /predict
→ The frontend must send exactly 80 floats. Check the features array length in the request.
VS Code shows missing imports but code runs
→ Make sure VS Code uses the .venv interpreter (Ctrl+Shift+P → Python: Select Interpreter → .venv\Scripts\python.exe) and reload the window.
SHAP is slow
→ Use fewer background samples or an approximate explainer; consider running SHAP offline and caching results.

Ethical use

This project is intended for cybersecurity research and education only. Do not use it for unauthorized monitoring or intrusion. Always follow institutional rules and local laws when working with network traffic.

Credits & license
License: MIT (see LICENSE)
Dataset: CICIDS2017 (University of New Brunswick)
Model: LSTM trained using cleaned CICIDS2017_improved
