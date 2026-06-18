# AI Intrusion Detection System using Deep Learning for DoS Attack Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green)
![React](https://img.shields.io/badge/React-TypeScript-blue)
![License](https://img.shields.io/badge/License-MIT-red)

---

## Project Overview

This project is a Artificial Intelligence based Intrusion Detection System (IDS) developed for detecting Denial-of-Service (DoS) attacks in network traffic using a Long Short-Term Memory (LSTM) neural network.

The system combines machine learning, explainable AI, backend API development, and frontend visualization into a complete end-to-end cybersecurity platform.

The machine learning pipeline processes temporal network flow sequences extracted from the CICIDS2017 Improved dataset and learns attack patterns across multiple timesteps. A trained LSTM model performs real-time classification of network traffic as either normal or malicious.

The project consists of:

- Deep Learning based DoS attack detection using LSTM
- FastAPI backend for real-time inference
- React + TypeScript dashboard for visualization
- SHAP explainability framework for model interpretation
- Automated evaluation and reporting pipeline
- Publication and presentation quality result generation

Main goals:

- Real-time network intrusion detection
- Explainable AI driven cybersecurity decisions
- High-performance temporal attack classification
- Interactive monitoring dashboard
- Research-grade reproducibility and evaluation

---

## Key Features

### Machine Learning

- Temporal sequence learning using LSTM
- Binary classification:
  - Normal Traffic
  - DoS Attack Traffic
- Automatic preprocessing pipeline
- Feature scaling and tensor generation
- Model evaluation and visualization

### Backend

- FastAPI REST API
- Real-time predictions
- Session-based inference
- JSON response interface
- Model persistence

### Frontend

- React + TypeScript dashboard
- Live prediction monitoring
- Attack probability visualization
- Detection history tracking
- Interactive user controls

### Explainability

- SHAP based feature attribution
- Global feature importance
- Local prediction explanations
- Waterfall visualizations
- Model interpretability reports

---

## System Architecture

```
Network Traffic
       ↓
Feature Extraction
       ↓
Temporal Sequence Generation
       ↓
Feature Scaling
       ↓
LSTM Neural Network
       ↓
Attack Probability
       ↓
FastAPI Backend
       ↓
React Dashboard
       ↓
Explainability Engine (SHAP)
```

---

## Repository Structure

```
ai-intrusion-detection/
├── backend/
│   ├── api.py
│   ├── inference.py
│   ├── main.py
│   ├── shap_explainer.py
│   └── ...
│
├── frontend/
│   ├── src/
│   ├── components/
│   ├── services/
│   ├── App.tsx
│   └── ...
│
├── ml/
│   ├── prepare_lstm_tensors.py
│   ├── train_lstm.py
│   ├── generate_final_presentation_result.py
│   └── evaluation/
│
├── models/
│   ├── dos_lstm_best.keras
│   └── dos_lstm_final.keras
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── tensors/
│
├── results/
│   ├── final_presentation/
│   ├── evaluation/
│   └── explainability/
│
├── README.md
└── LICENSE
```

---

## Dataset Information

This project uses the CICIDS2017 Improved Dataset and converts raw network flow records into temporal sequences suitable for LSTM training.

**Official Source:** https://www.unb.ca/cic/datasets/ids-2017.html

Final Tensor Shape: `(312165, 10, 84)`

| Dimension | Value | Description              |
| --------- | ----- | ------------------------ |
| 312,165   | axis 0 | Total sequences          |
| 10        | axis 1 | Temporal window size     |
| 84        | axis 2 | Features per timestep    |

---

## Model Details

### Dataset Statistics

| Metric                    | Value        |
| ------------------------- | ------------ |
| Total Samples             | 312,165      |
| Normal Samples            | 175,411      |
| Attack Samples            | 136,754      |
| Sequence Length           | 10 Timesteps |
| Features per Timestep     | 84           |
| Input Shape               | (10, 84)     |
| Total Features per Sample | 840          |

### Model Architecture

- **Model Type:** Long Short-Term Memory (LSTM)
- **Classification Type:** Binary Classification
- **Purpose:** Detection of Denial-of-Service (DoS) Attacks
- **Saved Model:** `models/dos_lstm_final.keras`

```
Input Layer → (10, 84)
       ↓
LSTM (128 Units)
       ↓
Batch Normalization
       ↓
Dropout
       ↓
LSTM (64 Units)
       ↓
Batch Normalization
       ↓
Dropout
       ↓
Dense (64 Units)
       ↓
Dense (1 Unit, Sigmoid)
```

### Model Parameters

| Parameter Type         | Count   |
| ---------------------- | ------- |
| Total Parameters       | 163,457 |
| Trainable Parameters   | 163,073 |
| Non-Trainable Parameters | 384   |

### Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 99.95% |
| Precision | 99.96% |
| Recall    | 99.92% |
| F1 Score  | 99.94% |
| ROC-AUC   | ~1.00  |

### Classification Report

**Normal Traffic (Class 0)**

| Metric    | Score  |
| --------- | ------ |
| Precision | 0.9994 |
| Recall    | 0.9997 |
| F1 Score  | 0.9995 |

**DoS Attack Traffic (Class 1)**

| Metric    | Score  |
| --------- | ------ |
| Precision | 0.9996 |
| Recall    | 0.9992 |
| F1 Score  | 0.9994 |

---

## Backend API

**Base Endpoint:** `POST /predict`

**Request:**

```json
{
    "session_id": "test_session",
    "features": [ /* exactly 840 floats */ ]
}
```

**Input tensor shape reshaped to:** `(1, 10, 84)`

**Response:**

```json
{
    "session_id": "test_session",
    "prediction": 1,
    "dos_probability": 0.9987,
    "status": "attack"
}
```

---

## Frontend Dashboard

The web dashboard provides:

- Real-time attack probability gauge
- Prediction history
- Detection logs
- Traffic visualization
- Performance metrics
- Explainability outputs

ControlPanel provides controls for sending network flow sequences containing 840 features (10 timesteps × 84 features).

Built using:

- React
- TypeScript
- Vite
- Recharts

---

## Explainable AI (SHAP)

The system integrates SHAP explainability to understand model decisions.

### Generated Explainability Outputs

- Top 15 Feature Importance Ranking
- Global SHAP Summary Plot
- Feature Importance Table
- Waterfall Style Local Explanations
- Feature Contribution Analysis

The explainability pipeline is also integrated into the automated result generation framework used for project presentations and evaluation.

---

## Evaluation and Visualization Pipeline

The project contains an automated result generation framework located in:

```
ml/generate_final_presentation_result.py
```

This script generates publication and presentation quality visualizations including:

1. Dataset Overview
2. Class Distribution
3. Dataset Summary Table
4. Feature Statistics
5. Confusion Matrix
6. ROC Curve
7. Precision Recall Curve
8. Threshold Optimization
9. Prediction Probability Distribution
10. Performance Dashboard
11. Top Feature Importance
12. Feature Importance Table
13. SHAP Summary Plot
14. SHAP Waterfall Plot

All figures are exported as high-resolution PNG files and stored inside the `results/` directory.

---

## Local Setup

**Create virtual environment:**

```powershell
python -m venv .venv
```

**Activate:**

```powershell
.\.venv\Scripts\activate
```

**Install dependencies:**

```powershell
pip install -r requirements.txt
```

---

## Running the Backend

```powershell
uvicorn backend.main:app --reload
```

Backend URL: `http://127.0.0.1:8000`

---

## Running the Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL: `http://localhost:5173`

---

## Reproducing Training

1. Download the CICIDS2017 dataset from the official source
2. Place raw CSV files in `data/raw/`
3. Run `ml/prepare_lstm_tensors.py` to generate tensors
4. Run `ml/train_lstm.py` to train the LSTM model
5. Trained model will be saved to `models/`
6. Run the evaluation pipeline to generate metrics and figures
7. Run `ml/generate_final_presentation_result.py` for explainability outputs

---

## Applications

- Intrusion Detection Systems
- Network Security Monitoring
- Smart Cyber-Physical Systems
- Critical Infrastructure Protection
- Real-Time Threat Detection
- Security Operations Centers (SOC)

---

## Ethical Use

This project is intended solely for:

- Academic Research
- Educational Purposes
- Cybersecurity Training
- Defensive Security Applications

Users must comply with all applicable laws, institutional policies, and ethical guidelines when deploying or testing intrusion detection systems.

---

## Credits & License

**Author**

Rudrapriya Singh Chauhan
B.Tech Cyber Physical Systems
Manipal Institute of Technology
Manipal Academy of Higher Education

**Dataset**

CICIDS2017 Dataset
Canadian Institute for Cybersecurity
University of New Brunswick

**License**

MIT License — see [LICENSE](LICENSE) for complete details.