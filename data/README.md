# Dataset Information

This project uses the CICIDS2017 dataset.

Due to size constraints, raw and processed datasets are not included in this repository.

## Directory structure (local only)

- raw/            Original CICIDS CSV files
- processed/      Cleaned feature tables
- temporal/       Sliding window outputs
- tensors/        LSTM tensors (.npy)

## Reproduction
Run scripts in `ml/` in the following order:
1. preprocess.py
2. temporal_windows.py
3. prepare_lstm_tensors.py