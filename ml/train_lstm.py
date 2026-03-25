import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json
import time

# =====================================================
# CONFIG
# =====================================================
DATA_DIR = Path("data/tensors")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = DATA_DIR / "X_lstm.npy"
Y_PATH = DATA_DIR / "y_lstm.npy"
SCHEMA_PATH = DATA_DIR / "feature_schema.json"

BEST_MODEL_PATH = str(MODEL_DIR / "dos_lstm_best.keras")
FINAL_MODEL_PATH = str(MODEL_DIR / "dos_lstm_final.keras")

EPOCHS = 30
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# =====================================================
# LOAD DATA
# =====================================================
print("[INFO] Loading LSTM tensors")

X = np.load(X_PATH, mmap_mode="r").astype(np.float32)
y = np.load(Y_PATH).astype(np.int64)

print(f"[INFO] X shape: {X.shape}")
print(f"[INFO] y shape: {y.shape}")

if len(X) != len(y):
    raise RuntimeError("X/y length mismatch")

with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)

WINDOW_SIZE = schema["window_size"]
NUM_FEATURES = schema["num_features"]

# =====================================================
# TRAIN / VALIDATION SPLIT
# =====================================================
print("[INFO] Creating train/validation split")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VALIDATION_SPLIT,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(f"[INFO] Train samples: {len(X_train)}")
print(f"[INFO] Val samples:   {len(X_val)}")

# =====================================================
# CLASS WEIGHTS
# =====================================================
print("[INFO] Computing class weights")

classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = {int(k): float(v) for k, v in zip(classes, weights)}

print("[INFO] Class weights:", class_weights)

# =====================================================
# MODEL
# =====================================================
print("[INFO] Building LSTM model")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW_SIZE, NUM_FEATURES)),

    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.LSTM(64),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

model.summary()

# =====================================================
# CALLBACKS
# =====================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
    ),
]

# =====================================================
# TRAIN
# =====================================================
print("[INFO] Starting training")
start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

elapsed = time.time() - start_time
print(f"[INFO] Training finished in {elapsed/60:.2f} minutes")

# =====================================================
# SAVE TRAINING GRAPHS (IMPORTANT)
# =====================================================
print("[INFO] Saving training graphs")

plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(RESULTS_DIR / "loss.png")
plt.close()

plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(RESULTS_DIR / "accuracy.png")
plt.close()

# =====================================================
# EVALUATION
# =====================================================
print("\n[INFO] Evaluating model")

y_pred_prob = model.predict(X_val, batch_size=BATCH_SIZE)
y_pred = (y_pred_prob >= 0.5).astype(int).ravel()

report = classification_report(y_val, y_pred, digits=4)
cm = confusion_matrix(y_val, y_pred)

print("\n[INFO] Classification Report")
print(report)

print("\n[INFO] Confusion Matrix")
print(cm)

# Save report
with open(RESULTS_DIR / "classification_report.txt", "w") as f:
    f.write(report)

# Save confusion matrix CSV
np.savetxt(RESULTS_DIR / "confusion_matrix.csv", cm, delimiter=",")

# Plot confusion matrix
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig(RESULTS_DIR / "confusion_matrix.png")
plt.close()

# =====================================================
# SAVE FINAL MODEL
# =====================================================
model.save(FINAL_MODEL_PATH)
print(f"\n[SUCCESS] Final model saved → {FINAL_MODEL_PATH}")