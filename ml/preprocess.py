import pandas as pd
from pathlib import Path

# ================================
# CONFIG
# ================================
DATA_DIR = Path("data/raw/CICIDS2017_improved")

FILES = {
    "monday": "monday.csv",
    "tuesday": "tuesday.csv",
    "wednesday": "wednesday.csv",
    "thursday": "thursday.csv",
    "friday": "friday.csv",
}

# ================================
# SAFE LOAD FUNCTION
# ================================
def load_sample(day: str, n_rows: int = 5):
    """
    Load a very small sample of a dataset
    for inspection ONLY (safe for RAM).
    """
    file_path = DATA_DIR / FILES[day]
    df = pd.read_csv(file_path, nrows=n_rows)
    return df


if __name__ == "__main__":
    df_sample = load_sample("monday", n_rows=5)

    print("\n=== SAMPLE ROWS ===")
    print(df_sample)

    print("\n=== COLUMNS ===")
    print(list(df_sample.columns))

    print("\n=== DATA TYPES ===")
    print(df_sample.dtypes)
# ================================
# FEATURE GROUP DEFINITIONS
# ================================

IDENTIFIER_COLUMNS = [
    "id",
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
]

META_COLUMNS = [
    "Attempted Category",
]

LABEL_COLUMN = "Label"


def classify_columns(df):
    all_columns = set(df.columns)

    identifier_cols = [c for c in IDENTIFIER_COLUMNS if c in all_columns]
    meta_cols = [c for c in META_COLUMNS if c in all_columns]
    label_col = LABEL_COLUMN if LABEL_COLUMN in all_columns else None

    feature_cols = list(
        all_columns - set(identifier_cols) - set(meta_cols) - {label_col}
    )

    return {
        "identifiers": identifier_cols,
        "metadata": meta_cols,
        "features": sorted(feature_cols),
        "label": label_col,
    }


if __name__ == "__main__":
    df_sample = load_sample("monday", n_rows=5)

    column_groups = classify_columns(df_sample)

    print("\n=== COLUMN GROUP SUMMARY ===")
    for k, v in column_groups.items():
        if isinstance(v, list):
            print(f"{k}: {len(v)} columns")
        else:
            print(f"{k}: {v}")
# ================================
# LABEL DISTRIBUTION ANALYSIS
# ================================

def analyze_label_distribution(day: str):
    file_path = DATA_DIR / FILES[day]

    # Load ONLY label column to save memory
    df = pd.read_csv(file_path, usecols=[LABEL_COLUMN])

    label_counts = df[LABEL_COLUMN].value_counts()

    print(f"\n=== LABEL DISTRIBUTION ({day.upper()}) ===")
    print(label_counts)
    print("\nTotal samples:", label_counts.sum())


if __name__ == "__main__":
    analyze_label_distribution("monday")
# ================================
# ATTACK DAY LABEL ANALYSIS
# ================================

def analyze_attack_days():
    attack_days = ["tuesday", "wednesday", "thursday", "friday"]

    for day in attack_days:
        file_path = DATA_DIR / FILES[day]

        df = pd.read_csv(file_path, usecols=[LABEL_COLUMN])

        print(f"\n=== LABEL DISTRIBUTION ({day.upper()}) ===")
        print(df[LABEL_COLUMN].value_counts())
        print("Total samples:", len(df))


if __name__ == "__main__":
    analyze_attack_days()
# ================================
# BINARY LABEL MAPPING
# ================================

def map_label_binary(label: str) -> int:
    """
    BENIGN -> 0
    Any attack -> 1
    """
    if label == "BENIGN":
        return 0
    return 1

if __name__ == "__main__":
    test_labels = ["BENIGN", "DoS Hulk", "SSH-Patator", "Botnet"]
    print("\n=== BINARY LABEL TEST ===")
    for lbl in test_labels:
        print(lbl, "->", map_label_binary(lbl))

# ================================
# CHUNK-BASED PREPROCESSING
# ================================

def preprocess_day(
    day: str,
    chunk_size: int = 100_000,
):
    """
    Memory-safe preprocessing of one day of traffic.
    """
    input_path = DATA_DIR / FILES[day]
    output_path = Path("data/processed") / f"{day}_processed.csv"

    print(f"\n[INFO] Processing {day.upper()}")

    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(input_path, chunksize=chunk_size):
        # Drop identifier & metadata columns
        chunk = chunk.drop(columns=IDENTIFIER_COLUMNS + META_COLUMNS, errors="ignore")

        # Map label to binary
        chunk["label_binary"] = chunk[LABEL_COLUMN].apply(map_label_binary)

        # Drop original label
        chunk = chunk.drop(columns=[LABEL_COLUMN])

        # Write to disk
        chunk.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        total_rows += len(chunk)
        first_chunk = False

        print(f"  Processed {total_rows} rows", end="\r")

    print(f"\n[INFO] Finished {day.upper()} → {total_rows} rows written")

if __name__ == "__main__":
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        preprocess_day(day, chunk_size=100_000)

