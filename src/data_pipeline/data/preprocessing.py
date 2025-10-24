"""
preprocessing.py
---------------------------------
Performs secondary preprocessing on raw OHLCV + funding data:
 - Validates time continuity
 - Cleans duplicates and NaNs
 - Ensures correct timestamp type
 - Optionally recalculates log returns and directions

Usage:
  python src/data_pipeline/data/preprocessing.py --input data/raw/BTC_USDT_15m_futures_features.parquet --output data/processed/BTC_USDT_15m_features.parquet
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def preprocess(input_path: str, output_path: str):
    print("[INFO] Starting preprocessing...")
    df = pd.read_parquet(input_path)
    print(f"[INFO] Loaded {len(df):,} rows from {input_path}")

    # --- Timestamp validation ---
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        print("[WARN] Converting timestamp to datetime (UTC)...")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if not df["timestamp"].dt.tz:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    # --- Remove duplicates & sort ---
    dup_count = df.duplicated(subset="timestamp").sum()
    if dup_count > 0:
        print(f"[WARN] Found and removed {dup_count} duplicate timestamps.")
        df = df.drop_duplicates(subset="timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Check missing intervals ---
    expected_interval = pd.Timedelta(minutes=15)
    diffs = df["timestamp"].diff().dropna()
    missing_intervals = diffs[diffs != expected_interval]
    if not missing_intervals.empty:
        print(f"[WARN] Found {len(missing_intervals)} irregular intervals.")
        print(missing_intervals.value_counts().head())
    else:
        print("[INFO] No missing intervals detected.")

    # --- Check NaN values ---
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print("[WARN] Found NaN values:")
        print(nan_counts[nan_counts > 0])
        df = df.dropna().reset_index(drop=True)
        print("[INFO] Dropped rows with NaN values.")
    else:
        print("[INFO] No NaN values found.")

    # --- Ensure all numeric columns are correct dtype ---
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # --- Save processed file ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[OK] Saved cleaned dataset -> {output_path}")
    print(f"[INFO] Final shape: {df.shape}")

    print("\n[INFO] Columns:")
    for c in df.columns:
        print("  -", c)


def main():
    parser = argparse.ArgumentParser(description="Clean and preprocess raw OHLCV + funding data")
    parser.add_argument("--input", type=str, required=True, help="Path to input parquet file")
    parser.add_argument("--output", type=str, required=True, help="Path to save cleaned parquet file")
    args = parser.parse_args()

    preprocess(args.input, args.output)


if __name__ == "__main__":
    main()
