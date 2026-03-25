from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

ROOT = Path(".")
REPORT_DIR = ROOT / "docs" / "report_assets"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# 1) Class distribution
# -------------------------
def load_labels():
    y_path = ROOT / "data" / "tensors" / "y_lstm.npy"
    if y_path.exists():
        y = np.load(y_path)
        return y.astype(int)

    # fallback: count label_binary from final CSVs
    final_dir = ROOT / "data" / "final"
    labels = []
    if final_dir.exists():
        for csv_path in final_dir.glob("*_final.csv"):
            df = pd.read_csv(csv_path)
            if "label_binary" in df.columns:
                labels.extend(df["label_binary"].dropna().astype(int).tolist())
    return np.array(labels, dtype=int)

y = load_labels()
if len(y) > 0:
    counts = np.bincount(y)
    plt.figure(figsize=(6, 5))
    plt.bar(["Normal", "Attack"], counts[:2])
    plt.title("Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "class_distribution.png", dpi=220)
    plt.close()

# -------------------------
# 2) Feature tensor shape diagram
# -------------------------
schema_path = ROOT / "data" / "tensors" / "feature_schema.json"
if schema_path.exists():
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    window_size = schema.get("window_size", 10)
    num_features = schema.get("num_features", 86)
    flat = schema.get("flattened_features", window_size * num_features)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    blocks = [
        (0.04, 0.35, 0.18, 0.25, f"Flattened input\n{flat} features"),
        (0.28, 0.35, 0.18, 0.25, f"Reshape\n({window_size}, {num_features})"),
        (0.52, 0.35, 0.18, 0.25, "LSTM model"),
        (0.76, 0.35, 0.18, 0.25, "Binary output\n(normal / attack)"),
    ]

    for x, y0, w, h, txt in blocks:
        patch = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.02", fill=False, linewidth=1.5)
        ax.add_patch(patch)
        ax.text(x + w/2, y0 + h/2, txt, ha="center", va="center", fontsize=11)

    for x1, x2 in [(0.22, 0.28), (0.46, 0.52), (0.70, 0.76)]:
        ax.add_patch(FancyArrowPatch((x1, 0.475), (x2, 0.475), arrowstyle="->", mutation_scale=16, linewidth=1.4))

    ax.text(0.5, 0.88, "Feature Tensor Flow", ha="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.10, "Safe presentation view: shows data flow only, not private dataset rows.", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_tensor_shape_diagram.png", dpi=220)
    plt.close()

# -------------------------
# 3) Training history plots
# -------------------------
hist_path = REPORT_DIR / "training_history.json"
if hist_path.exists():
    hist = json.loads(hist_path.read_text(encoding="utf-8"))
    if "loss" in hist and "val_loss" in hist:
        plt.figure(figsize=(8, 5))
        plt.plot(hist["loss"], label="Training loss")
        plt.plot(hist["val_loss"], label="Validation loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "training_loss.png", dpi=220)
        plt.close()

    if "accuracy" in hist and "val_accuracy" in hist:
        plt.figure(figsize=(8, 5))
        plt.plot(hist["accuracy"], label="Training accuracy")
        plt.plot(hist["val_accuracy"], label="Validation accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "training_accuracy.png", dpi=220)
        plt.close()

# -------------------------
# 4) Confusion matrix
# -------------------------
cm_path = REPORT_DIR / "confusion_matrix.npy"
if cm_path.exists():
    cm = np.load(cm_path)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred Normal", "Pred Attack"])
    plt.yticks([0, 1], ["Actual Normal", "Actual Attack"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=220)
    plt.close()

# -------------------------
# 5) Model summary screenshot (from saved text)
# -------------------------
summary_txt = REPORT_DIR / "model_summary.txt"
if summary_txt.exists():
    txt = summary_txt.read_text(encoding="utf-8")
    plt.figure(figsize=(12, 8))
    plt.axis("off")
    plt.text(
        0.01, 0.99, txt,
        ha="left", va="top",
        family="monospace", fontsize=9
    )
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_architecture_summary.png", dpi=220)
    plt.close()

# -------------------------
# 6) Backend response screenshot
# -------------------------
response_json = REPORT_DIR / "backend_response.json"
if response_json.exists():
    response_text = response_json.read_text(encoding="utf-8")
    plt.figure(figsize=(10, 5))
    plt.axis("off")
    plt.text(0.01, 0.98, "Backend prediction response", ha="left", va="top", fontsize=14, weight="bold")
    plt.text(0.01, 0.85, response_text, ha="left", va="top", family="monospace", fontsize=11)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "backend_prediction_response.png", dpi=220)
    plt.close()

print(f"Assets generated in: {REPORT_DIR}")