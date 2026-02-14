import pandas as pd
from pathlib import Path

RAW_DIR = Path("raw/CICIDS2017_improved")

FILES = {
    "monday": RAW_DIR / "monday.csv",
    "tuesday": RAW_DIR / "tuesday.csv",
    "wednesday": RAW_DIR / "wednesday.csv",
    "thursday": RAW_DIR / "thursday.csv",
    "friday": RAW_DIR / "friday.csv",
}

LABEL_COLUMN = "label_binary"

def inspect_file(name: str, path: Path):
    print(f"\n==============================")
    print(f"DATASET: {name.upper()}")
    print(f"PATH   : {path}")
    print(f"==============================")

    if not path.exists():
        print("❌ File not found")
        return None

    df = pd.read_csv(path)

    print(f"Rows            : {len(df)}")
    print(f"Columns         : {len(df.columns)}")
    print(f"Column names    :")
    print(df.columns.tolist())

    if LABEL_COLUMN in df.columns:
        print("\nLabel distribution:")
        print(df[LABEL_COLUMN].value_counts())
    else:
        print("\n⚠️ label_binary column missing")

    print("\nFirst row:")
    print(df.iloc[0])

    print("\nLast 5 columns:")
    print(df.columns[-5:].tolist())

    return df.columns.tolist()

def main():
    schemas = {}

    for name, path in FILES.items():
        cols = inspect_file(name, path)
        if cols:
            schemas[name] = cols

    print("\n==============================")
    print("SCHEMA CONSISTENCY CHECK")
    print("==============================")

    base_schema = None
    for name, cols in schemas.items():
        if base_schema is None:
            base_schema = cols
            continue

        if cols != base_schema:
            print(f"❌ Schema mismatch in {name}")
        else:
            print(f"✅ Schema matches for {name}")

if __name__ == "__main__":
    main()
