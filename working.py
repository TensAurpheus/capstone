"""
working.py
---------------------------------
Quick functional test for normalized/standardized datasets.

Checks:
1. Loading parquet
2. Label generation (min_max or triple_barrier)
3. Dataset creation (CryptoDataset)
4. Scaling consistency (split_scale)
"""

import sys
from pathlib import Path
import pandas as pd
import torch
from src.utils.data_utils import triple_barrier_label, min_max_label, split_scale, CryptoDataset



if __name__ == "__main__":
    # === 1️⃣ Select which dataset to test ===
    symbol = "BTC_USDT"
    timeframe = "15m"

    parquet_path = Path(f"data/processed/{symbol}_{timeframe}_standardized.parquet")
    if not parquet_path.exists():
        parquet_path = Path(f"data/processed/{symbol}_{timeframe}_normalized.parquet")

    print(f"[INFO] Loading dataset → {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"[INFO] Loaded {len(df):,} rows and {len(df.columns)} columns")

    # === 2️⃣ Label generation ===
    print("\n[STEP] Generating regression labels (min_max)...")
    df_labeled = min_max_label(df, horizon=8)
    print(f"[OK] Added columns: {set(df_labeled.columns) - set(df.columns)}")

    # === 3️⃣ Create PyTorch Dataset ===
    print("\n[STEP] Creating CryptoDataset...")
    target_cols = ["y_high", "y_low"]
    dataset = CryptoDataset(df_labeled, window_size=32, target="y_high")

    X, y = dataset[0]
    print(f"[OK] Dataset created — total windows: {len(dataset)}")
    print(f"Example window shape: {X.shape}, target: {y}")

    # === 4️⃣ Check split & scaling ===
    print("\n[STEP] Splitting and scaling dataset...")
    df_train, df_val, df_test, scaler = split_scale(
        df_labeled, target_cols=target_cols, test_size=0.2, val_size=0.1, scale=True
    )

    print(f"[OK] Split complete:")
    print(f"  Train: {len(df_train)} rows")
    print(f"  Val:   {len(df_val)} rows")
    print(f"  Test:  {len(df_test)} rows")

    if scaler:
        print(f"[INFO] Scaler mean (first 5): {scaler.scaler.mean_[:5]}")
        print(f"[INFO] Scaler std  (first 5): {scaler.scaler.scale_[:5]}")

    print("\n✅ All tests completed successfully.")
