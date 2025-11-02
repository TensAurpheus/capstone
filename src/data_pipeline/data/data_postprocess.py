"""
data_postprocess.py
---------------------------------
Final stage of the data pipeline.
1. Transforms mixed-type dataframe (categorical, boolean, timestamps) into numeric.
2. Creates two model-ready datasets:
   - normalized.parquet (for ML models)
   - standardized.parquet + .pkl (for DL models)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


# ============================================================
# === 1️⃣ PREPARE BASE NUMERIC DATAFRAME ======================
# ============================================================

def prepare_dataframe_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """Transform mixed dataframe into fully numeric format suitable for ML/DL models."""
    print("[INFO] Preparing DataFrame for modeling...")

    # --- Timestamp handling ---
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # Add cyclical encodings (useful for both ML & DL)
        df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
        df.drop(columns=["timestamp"], inplace=True)

    # --- One-hot encode categorical ---
    for col in ["session", "pda"]:
        if col in df.columns and df[col].nunique() > 1:
            df = pd.get_dummies(df, columns=[col], prefix=col)
        elif col in df.columns:
            df.drop(columns=[col], inplace=True)

    # --- Boolean to int ---
    for col in ["fvg_present", "pattern_active"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # --- Symbol column ---
    if "symbol" in df.columns:
        if df["symbol"].nunique() <= 1:
            df.drop(columns=["symbol"], inplace=True)
        else:
            df = pd.get_dummies(df, columns=["symbol"], prefix="symbol")

    # --- Drop remaining non-numeric ---
    non_numeric_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric_cols:
        print(f"[WARN] Dropping non-numeric columns: {non_numeric_cols}")
        df.drop(columns=non_numeric_cols, inplace=True)

    df = df.dropna().astype(float)
    print(f"[OK] Final numeric dataset shape: {df.shape}")
    return df


# ============================================================
# === 2️⃣ MAIN PIPELINE ENTRY ================================
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Final postprocessing for ML & DL model-ready datasets")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--market", type=str, default="futures")
    parser.add_argument("--timeframe", type=str, default="15m")
    args = parser.parse_args()

    symbol_clean = args.symbol.replace("/", "_")

    # --- Find last available processed stage ---
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

    # --- Step 1: Load base dataframe ---
    print(f"[INFO] Loading {input_path}")
    df = pd.read_parquet(input_path)

    # --- Step 2: Make fully numeric ---
    df = prepare_dataframe_for_model(df)

    os.makedirs("data/processed", exist_ok=True)

    # --- Step 3: Save base numeric dataset ---
    base_path = Path("data/processed") / f"{symbol_clean}_{args.timeframe}_numeric.parquet"
    df.to_parquet(base_path, index=False)
    print(f"[OK] Saved numeric dataset to {base_path}")

    # ============================================================
    # === Step 4: Run normalization (ATR scale-free) ============
    # ============================================================
    normalized_path = Path("data/processed") / f"{symbol_clean}_{args.timeframe}_normalized.parquet"
    print(f"[INFO] Running normalization -> {normalized_path.name}")

    subprocess.run([
        sys.executable, "src/data_pipeline/data/normalize.py",
        "--input", str(base_path),
        "--output", str(normalized_path),
        "--symbol", args.symbol
    ], check=True)

    # ============================================================
    # === Step 5: Run standardization (z-score + .pkl) ===========
    # ============================================================
    standardized_path = Path("data/processed") / f"{symbol_clean}_{args.timeframe}_standardized.parquet"
    print(f"[INFO] Running standardization -> {standardized_path.name}")

    subprocess.run([
        sys.executable, "src/data_pipeline/data/standardize.py",
        "--input", str(normalized_path),
        "--output", str(standardized_path),
        "--symbol", args.symbol
    ], check=True)

    print(f"[OK] Final ML dataset -> {normalized_path}")
    print(f"[OK] Final DL dataset -> {standardized_path}")
    print(f"[OK] Scaler saved -> data/model/scalers/standard_scaler_{symbol_clean}.pkl")

    # --- Step 6: Clean up intermediate files ---
    print("[INFO] Cleaning up intermediate files...")
    for stage_name in stages_priority:
        file_path = Path("data/processed") / f"{symbol_clean}_{args.timeframe}_{stage_name}.parquet"
        if file_path.exists():
            os.remove(file_path)
            print(f"   - Removed {file_path.name}")

    # Optionally remove numeric base file if you want
    if base_path.exists():
        os.remove(base_path)
        print(f"   - Removed {base_path.name}")

    print("\n[OK] Data pipeline complete!")
    print(f" ML dataset: {normalized_path}")
    print(f" DL dataset: {standardized_path}")


if __name__ == "__main__":
    main()
