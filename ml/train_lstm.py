import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import json
import time

# =====================================================
# CONFIG
# =====================================================
DATA_DIR = Path("data/tensors")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

X = np.load(X_PATH, mmap_mode="r")
y = np.load(Y_PATH)

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
# CLASS WEIGHTS (CRITICAL)
# =====================================================
print("[INFO] Computing class weights")

classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, weights))

print("[INFO] Class weights:", class_weights)

# =====================================================
# MODEL
# =====================================================
print("[INFO] Building LSTM model")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW_SIZE, NUM_FEATURES)),

    tf.keras.layers.LSTM(
        128,
        return_sequences=True,
        activation="tanh"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.LSTM(
        64,
        return_sequences=False,
        activation="tanh"
    ),
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
# EVALUATION
# =====================================================
print("\n[INFO] Evaluating model")

y_pred_prob = model.predict(X_val, batch_size=BATCH_SIZE)
y_pred = (y_pred_prob >= 0.5).astype(int).ravel()

print("\n[INFO] Classification Report")
print(classification_report(y_val, y_pred, digits=4))

print("\n[INFO] Confusion Matrix")
print(confusion_matrix(y_val, y_pred))

# =====================================================
# SAVE FINAL MODEL
# =====================================================
model.save(FINAL_MODEL_PATH)
print(f"\n[SUCCESS] Final model saved → {FINAL_MODEL_PATH}")
