import os
import json
import time
import numpy as np
import tensorflow as tf
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dos_lstm_final.keras")
ARTIFACT_DIR = os.path.join(BASE_DIR, "backend", "validation", "artifacts")
LOG_DIR = os.path.join(BASE_DIR, "backend", "validation", "logs")
LOG_FILE = os.path.join(LOG_DIR, "two_stage_decision.log")

WINDOW_SIZE = 10
BENIGN_LABEL = 0

DEFAULT_CONF_THRESHOLD = 0.70

HIGH_FP_CLASSES = {1, 2, 4, 5, 10, 12, 19, 20, 21, 22, 26}

os.makedirs(LOG_DIR, exist_ok=True)

def log(msg):
    ts = datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

class TwoStageIDS:
    def __init__(self, model_path=MODEL_PATH, threshold=DEFAULT_CONF_THRESHOLD):
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold
        log("TwoStageIDS initialized")

    def stage1_predict(self, X):
        probs = self.model.predict(X, verbose=0)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        return preds, confs, probs

    def stage2_decide(self, pred, conf):
        if conf < self.threshold:
            return {
                "final_decision": "UNCERTAIN",
                "action": "ESCALATE",
                "reason": "LOW_CONFIDENCE",
                "predicted_class": int(pred),
                "confidence": float(conf)
            }

        if pred in HIGH_FP_CLASSES:
            return {
                "final_decision": "UNCERTAIN",
                "action": "ESCALATE",
                "reason": "HIGH_FP_CLASS",
                "predicted_class": int(pred),
                "confidence": float(conf)
            }

        if pred == BENIGN_LABEL:
            return {
                "final_decision": "BENIGN",
                "action": "ALLOW",
                "reason": "CONFIDENT_BENIGN",
                "predicted_class": int(pred),
                "confidence": float(conf)
            }

        return {
            "final_decision": "ATTACK",
            "action": "ALERT",
            "reason": "CONFIDENT_ATTACK",
            "predicted_class": int(pred),
            "confidence": float(conf)
        }

    def process_batch(self, X):
        preds, confs, probs = self.stage1_predict(X)
        decisions = []
        for p, c in zip(preds, confs):
            decisions.append(self.stage2_decide(int(p), float(c)))
        return decisions

def demo_run():
    log("Starting two-stage decision demo")
    X_dummy = np.random.rand(8, WINDOW_SIZE, 86).astype(np.float32)
    ids = TwoStageIDS()
    decisions = ids.process_batch(X_dummy)
    out_path = os.path.join(ARTIFACT_DIR, "two_stage_decision_demo.json")
    with open(out_path, "w") as f:
        json.dump(decisions, f, indent=2)
    log("Demo decisions saved to " + out_path)

if __name__ == "__main__":
    demo_run()
