"""
data_postprocess.py
---------------------------------
Final stage of the data pipeline.

Converts normalized crypto dataframe
into a fully numeric, model-ready dataset:
BTC_USDT_15m_futures.parquet
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


def prepare_dataframe_for_model(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Preparing DataFrame for modeling...")

    df = df.copy()

    # --- Timestamp handling ---
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
        df.drop(columns=["timestamp"], inplace=True)

    # --- One-hot encode categorical columns ---
    for col in ["session", "pda"]:
        if col in df.columns and df[col].nunique() > 1:
            df = pd.get_dummies(df, columns=[col], prefix=col)
        elif col in df.columns:
            df.drop(columns=[col], inplace=True)

    #  Boolean conversion
    bool_cols = df.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # --- Drop text or symbol columns ---
    for col in ["symbol"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # --- Drop remaining non-numeric ---
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        print(f"[WARN] Dropping non-numeric columns: {non_numeric}")
        df.drop(columns=non_numeric, inplace=True)

    df = df.dropna().astype(float)
    print(f"[OK] Final numeric dataset shape: {df.shape}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Finalize normalized crypto dataset")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--market", type=str, default="futures")
    parser.add_argument("--timeframe", type=str, default="15m")
    args = parser.parse_args()

    symbol_clean = args.symbol.replace("/", "_")

    # --- Load normalized dataset ---
    normalized_path = Path("data/processed") / f"{symbol_clean}_{args.timeframe}_normalized.parquet"
    if not normalized_path.exists():
        print(f"[ERROR] Normalized dataset not found at {normalized_path}. Run normalize.py first!")
        sys.exit(1)

    print(f"[INFO] Loading {normalized_path}")
    df = pd.read_parquet(normalized_path)

    # --- Clean and encode ---
    df = prepare_dataframe_for_model(df)

    # --- Save final dataset ---
    final_path = Path("data/processed") / f"{symbol_clean}_{args.timeframe}_{args.market}.parquet"
    os.makedirs(final_path.parent, exist_ok=True)
    df.to_parquet(final_path, index=False)

    print(f"[OK] Final dataset saved to {final_path}")
    print(f"[COLUMNS COUNT] {len(df.columns)}")
    print("[DONE] Data postprocessing complete.")

    # ---Save also to Excel ---
    excel_path = final_path.with_suffix(".xlsx")
    try:
        df_final = pd.read_parquet(final_path)

        # --- Fix timezone-aware datetime columns for Excel ---
        for col in df_final.select_dtypes(include=["datetimetz"]).columns:
            df_final[col] = df_final[col].dt.tz_localize(None)

        df_final.to_excel(excel_path, index=False)

        print(f"[OK] Excel file saved to {excel_path}")
    except Exception as e:
        print(f"[WARN] Could not save Excel file: {e}")

    # --- Remove intermediate parquet files ---
    stages = ["features", "technical", "patterns", "macro", "normalized"]
    for stage in stages:
        temp_file = Path("data/processed") / f"{symbol_clean}_{args.timeframe}_{stage}.parquet"
        if temp_file.exists() and temp_file != final_path:
            temp_file.unlink()

if __name__ == "__main__":
    main()
