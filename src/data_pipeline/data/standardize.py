"""
standardize.py
---------------------------------
Applies z-score standardization (StandardScaler) 
to the normalized dataset for DL models and saves both
the standardized .parquet and the scaler .pkl.
"""

import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib


def standardize_features(df: pd.DataFrame, scaler_path: str) -> pd.DataFrame:
    """
    Apply z-score standardization using StandardScaler and save scaler.
    """
    print("\n=== STANDARDIZATION STARTED ===")

    # --- Select numeric columns ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # --- Detect binary (exclude from scaling) ---
    binary_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) <= 2]
    cont_cols = [c for c in numeric_cols if c not in binary_cols]

    print(f"[INFO] Continuous features to be standardized: {len(cont_cols)} columns")
    print(f"[INFO] Binary/excluded features: {len(binary_cols)} columns")

    # --- Apply StandardScaler only to continuous numeric features ---
    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])

    # --- Save fitted scaler ---
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[OK] Saved scaler to {scaler_path}")

    # --- Diagnostic summary ---
    summary = df[cont_cols].describe().T[["mean", "std"]]
    large_scale = summary[(summary["std"] > 5) | (summary["mean"].abs() > 5)]
    if not large_scale.empty:
        print("[WARN] Some features may be unscaled:")
        print(large_scale)
    else:
        print("[OK] All features standardized properly.")

    print("=== STANDARDIZATION COMPLETED ===")
    return df


def main():
    parser = argparse.ArgumentParser(description="Standardize normalized dataset for DL models")
    parser.add_argument("--input", type=str, required=True, help="Path to normalized .parquet file")
    parser.add_argument("--output", type=str, required=True, help="Path to save standardized .parquet file")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol name (e.g. BTC/USDT)")
    args = parser.parse_args()

    print(f"[INFO] Loading dataset -> {args.input}")
    df = pd.read_parquet(args.input)
    print(f"[INFO] Loaded {len(df):,} rows, {len(df.columns)} columns")

    # --- Construct scaler path ---
    symbol_clean = args.symbol.replace("/", "_")
    scaler_path = os.path.join("data", "model", "scalers", f"standard_scaler_{symbol_clean}.pkl")

    # --- Apply standardization ---
    df_std = standardize_features(df, scaler_path)

    # --- Save standardized dataset ---
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_std.to_parquet(args.output, index=False)
    print(f"[OK] Saved standardized dataset -> {args.output}")
    print(f"[COLUMNS] {len(df_std.columns)} features")


if __name__ == "__main__":
    main()
