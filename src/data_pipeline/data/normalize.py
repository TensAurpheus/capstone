"""
normalize.py
---------------------------------
Feature normalization and standardization for DL/ML models.
Converts OHLC-based indicators into scale-free features (ATR-relative),
then applies StandardScaler via DataScaler class.
"""

import os
import numpy as np
import pandas as pd
from src.data_pipeline.data.data_utils import DataScaler
import argparse

def normalize_features(df: pd.DataFrame, scale_for_dl=True, save_scaler_path=None):
    """
    Convert dataset to scale-free features and optionally standardize continuous columns
    using DataScaler class (shared across pipeline, training, inference).
    

    Args:
        df : pandas DataFrame containing full feature set
        scale_for_dl : whether to apply standard scaling (True for DL, False for trees)
        save_scaler_path : path to save fitted StandardScaler (used in DL pipeline)
    Returns:
        normalized DataFrame
    """
    print("\n=== NORMALIZATION STARTED ===")

    # --- Ensure ATR is available ---
    if "atr_14" not in df.columns:
        print("[WARN] 'atr_14' not found — using rolling std(14) as proxy.")
        df["atr_14"] = df["close"].rolling(14).std()
    atr = df["atr_14"].replace(0, np.nan)

    # =====================================================
    # === SCALE-FREE TRANSFORMATIONS ======================
    # =====================================================

    # atr_14 → normalized by close (relative % volatility)
    # other features → normalized by ATR (scale-free vs volatility)
    mappings = {
        # Moving averages (relative to ATR)
        "ema_20": (df["close"] - df["ema_20"]) / atr,
        "ema_50": (df["close"] - df["ema_50"]) / atr,
        "ema_200": (df["close"] - df["ema_200"]) / atr,

        # MACD-related (normalize by ATR)
        "macd": df["macd"] / atr,
        "macd_signal": df["macd_signal"] / atr,
        "macd_hist": df["macd_hist"] / atr,

        # --- Volatility ---
        "atr_14": df["atr_14"] / df["close"],           # relative volatility (% of price)
        "bb_bbm": (df["close"] - df["bb_bbm"]) / atr,   # BB middle vs price
        "bb_bbh": (df["close"] - df["bb_bbh"]) / atr,   # BB high vs price
        "bb_bbl": (df["close"] - df["bb_bbl"]) / atr,  

        # Rolling highs/lows normalized by ATR
        "rolling_high": (df["rolling_high"] - df["close"]) / atr,
        "rolling_low": (df["close"] - df["rolling_low"]) / atr,

        # --- Rolling volatility ---
        "roll_std_16": df["roll_std_16"],   # already scale-free
        "roll_std_32": df["roll_std_32"],

        # Equilibrium & VWAP normalized
        "equilibrium": (df["close"] - df["equilibrium"]) / atr,
        "vwap_session": (df["close"] - df["vwap_session"]) / atr,
        "vwap": (df["close"] - df["vwap"]) / atr,

        # Volume
        "volume": np.log1p(df["volume"]),             
        "z_volume": df["z_volume"],
    }

    for col, expr in mappings.items():
        if col in df.columns:
            df[col] = expr

    # --- FVG gap (only if continuous, not binary) ---
    if "fvg_gap" in df.columns and df["fvg_gap"].nunique() > 2:
        df["fvg_gap"] = df["fvg_gap"] / atr

    # --- bb_width already scale-free (skip) ---
    if "bb_width" in df.columns:
        pass  # skip normalization, already scale-free

    # --- VWAP distance already scale-free (skip)
    if "vwap_distance" in df.columns:
        pass   # skip normalization, already scale-free

    # =====================================================
    # === STANDARDIZATION FOR DEEP LEARNING ===============
    # =====================================================
    if scale_for_dl:
        print("\n[INFO] Detecting continuous vs binary features...")

        # Numeric-only columns
        numeric_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns.tolist()

        # Binary columns (≤2 unique values)
        binary_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) <= 2]

        # Continuous columns (everything else)
        continuous_cols = [c for c in numeric_cols if c not in binary_cols]

        print(f"[INFO] Continuous features to be standardized ({len(continuous_cols)}):")
        print(", ".join(continuous_cols))
        print(f"\n[INFO] Binary / excluded features ({len(binary_cols)}):")
        print(", ".join(binary_cols))

        # Standardize continuous columns
        scaler = DataScaler(save_scaler_path)
        df = scaler.fit_transform(df, continuous_cols)

        # Optional: save fitted scaler for reuse
        if save_scaler_path:
            print(f"[OK] StandardScaler saved at: {save_scaler_path}")

        # =====================================================
        # === DIAGNOSTICS =====================================
        # =====================================================
        print("\n[CHECK] Verifying scaling consistency...")
        non_binary = df.select_dtypes(include=[np.number]).nunique() > 2
        non_binary_cols = df.columns[non_binary.values]
        summary = df[non_binary_cols].describe().T[["mean", "std", "min", "max"]]
        large_scale = summary[(summary["std"] > 5) | (summary["max"].abs() > 10)]
        if not large_scale.empty:
            print("[WARN] Potential unstandardized columns detected:")
            print(large_scale)
        else:
            print("[OK] All continuous features appear standardized.")

    print("\n=== NORMALIZATION COMPLETED ===")
    return df


# =====================================================
# === MAIN EXECUTION BLOCK (OPTIONAL) ================
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="Normalize and optionally scale features")
    parser.add_argument("--input", type=str, required=True, help="Input .parquet file (after patterns.py)")
    parser.add_argument("--output", type=str, required=True, help="Output .parquet file")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol name (e.g. BTC/USDT)")
    args = parser.parse_args()

    print(f"\n[INFO] Loading dataset → {args.input}")
    df = pd.read_parquet(args.input)
    print(f"[INFO] Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Ask user if scaling for DL models should be applied
    choice = input("\nApply StandardScaler (for ML/DL models)? [Y/n]: ").strip().lower()
    scale_for_dl = choice in ["", "y", "yes"]

    # Construct scaler path
    symbol_clean = args.symbol.replace("/", "_")
    os.makedirs(os.path.join("data", "model", "scalers"), exist_ok=True)
    scaler_path = os.path.join("data", "model", "scalers", f"standard_scaler_{symbol_clean}.pkl")

    print(f"\n[INFO] Normalizing features (scale_for_dl={scale_for_dl}) ...")
    df = normalize_features(df, scale_for_dl=scale_for_dl, save_scaler_path=scaler_path)
    print(f"[OK] Normalization complete. Scaler path: {scaler_path}")

    df.to_parquet(args.output, index=False)
    print(f"[OK] Saved normalized dataset → {args.output}")
    print(f"[COLUMNS] {df.columns.tolist()}")

if __name__ == "__main__":
    main()