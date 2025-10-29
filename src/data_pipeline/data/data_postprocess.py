"""
data_postprocess.py
---------------------------------
Final stage of the data pipeline.
Prepares the dataset into a fully numeric, model-ready format
and removes intermediate parquet files to keep storage clean.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os


def prepare_dataframe_for_torch(df: pd.DataFrame) -> pd.DataFrame:
    """Transform non-numeric columns into numeric format suitable for ML or Torch models."""
    print("[INFO] Preparing DataFrame for modeling...")

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    # Add cyclical encodings
    if pd.api.types.is_datetime64_any_dtype(df.index):
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # One-hot encode categorical columns
    for col in ["session", "pda"]:
        if col in df.columns and df[col].nunique() > 1:
            df = pd.get_dummies(df, columns=[col], prefix=col)
        elif col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Convert boolean columns
    for col in ["fvg_present", "pattern_active"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Drop constant symbol
    if "symbol" in df.columns:
        if df["symbol"].nunique() <= 1:
            df.drop(columns=["symbol"], inplace=True)
        else:
            df = pd.get_dummies(df, columns=["symbol"], prefix="symbol")

    # Drop remaining non-numeric
    non_numeric_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric_cols:
        print(f"[WARN] Dropping non-numeric columns: {non_numeric_cols}")
        df.drop(columns=non_numeric_cols, inplace=True)

    # Drop NaN (for early indicator warmup)
    df.dropna(inplace=True)

    df = df.astype(float)
    print(f"[OK] Final numeric dataset shape: {df.shape}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Final preprocessing for numeric model-ready dataset")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--market", type=str, default="futures")
    parser.add_argument("--timeframe", type=str, default="15m")
    args = parser.parse_args()

    symbol_clean = args.symbol.replace("/", "_")

    # --- Determine input file automatically (patterns > technical > features)
    stages_priority = ["patterns", "technical", "features"]
    input_path = None

    for stage_name in stages_priority:
        candidate = Path("data/processed") / f"{symbol_clean}_{args.timeframe}_{stage_name}.parquet"
        if candidate.exists():
            input_path = candidate
            print(f"[INFO] Using available dataset: {candidate}")
            break

    if input_path is None:
        print("[ERROR] No processed dataset found (patterns / technical / features). Exiting.")
        sys.exit(1)

    # Output path
    output_path = Path("data/processed") / f"{symbol_clean}_{args.timeframe}_numeric.parquet"

    print(f"[INFO] Loading {input_path}")
    df = pd.read_parquet(input_path)

    df_prepared = prepare_dataframe_for_torch(df)
    df_prepared.to_parquet(output_path)
    print(f"[OK] Saved final numeric dataset: {output_path}")

    # --- Remove intermediate parquet files
    print("[INFO] Cleaning up intermediate datasets...")
    for stage_name in stages_priority:
        file_path = Path("data/processed") / f"{symbol_clean}_{args.timeframe}_{stage_name}.parquet"
        if file_path.exists():
            os.remove(file_path)
            print(f"   - Removed {file_path.name}")

    print("[OK] Intermediate files removed. Data pipeline complete.")
    print(f"[OK] Final dataset ready: {output_path}")


if __name__ == "__main__":
    main()



