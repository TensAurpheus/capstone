"""
normalize.py
---------------------------------
Applies ATR-based (scale-free) normalization for ML models.
Converts OHLC-based indicators into volatility-relative features (ATR-relative).
"""

import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataset to scale-free features (ATR-relative).
    """
    print("\n=== SCALE-FREE NORMALIZATION STARTED ===")

    # --- Ensure ATR is available ---
    if "atr_14" not in df.columns:
        print("[WARN] 'atr_14' not found — using rolling std(14) as proxy.")
        df["atr_14"] = df["close"].rolling(14).std()

    atr = df["atr_14"].replace(0, np.nan)

    # =====================================================
    # === SCALE-FREE TRANSFORMATIONS ======================
    # =====================================================
    mappings = {
        "ema_20": (df["close"] - df["ema_20"]) / atr,
        "ema_50": (df["close"] - df["ema_50"]) / atr,
        "ema_200": (df["close"] - df["ema_200"]) / atr,
        "macd": df["macd"] / atr,
        "macd_signal": df["macd_signal"] / atr,
        "macd_hist": df["macd_hist"] / atr,
        "bb_bbm": (df["close"] - df["bb_bbm"]) / atr,
        "bb_bbh": (df["close"] - df["bb_bbh"]) / atr,
        "bb_bbl": (df["close"] - df["bb_bbl"]) / atr,
        "rolling_high": (df["rolling_high"] - df["close"]) / atr,
        "rolling_low": (df["close"] - df["rolling_low"]) / atr,
        "equilibrium": (df["close"] - df["equilibrium"]) / atr,
        "vwap_session": (df["close"] - df["vwap_session"]) / atr,
        "vwap": (df["close"] - df["vwap"]) / atr,
        "volume": np.log1p(df["volume"]),
        "z_volume": df["z_volume"],
    }

    for col, expr in mappings.items():
        if col in df.columns:
            df[col] = expr

 
    # ATR-relative volatility metrics
    df["atr_pct"] = df["atr_14"] / df["close"]
    df["range_atr"] = (df["high"] - df["low"]) / df["atr_14"]
    df["body_atr"] = (df["close"] - df["open"]) / df["atr_14"]           

    # FVG gap normalization
    if "fvg_gap" in df.columns and df["fvg_gap"].nunique() > 2:
        df["fvg_gap"] = df["fvg_gap"] / atr

    # Handle inf/nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Drop unstable EMA region
    min_period = 199  
    df = df.iloc[min_period:].reset_index(drop=True)
    print(f"[INFO] Dropped first {min_period} rows (unstable EMA region).")

    print("[OK] Scale-free normalization complete.")
    print("\n=== NORMALIZATION COMPLETED ===")
    return df


def main():
    parser = argparse.ArgumentParser(description="ATR-based normalization for ML models")
    parser.add_argument("--input", type=str, required=True, help="Input .parquet file (after patterns.py)")
    parser.add_argument("--output", type=str, required=True, help="Output .parquet file")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol name (e.g. BTC/USDT)")
    args = parser.parse_args()

    print(f"\n[INFO] Loading dataset -> {args.input}")
    df = pd.read_parquet(args.input)
    print(f"[INFO] Loaded {len(df):,} rows, {len(df.columns)} columns")

    # --- Save raw (pre-normalized) dataset to Excel ---
    excel_draft = Path(args.input).with_name(Path(args.input).stem + "_draft.xlsx")
    try:
        # Видаляємо timezone для Excel
        for col in df.select_dtypes(include=["datetimetz"]).columns:
            df[col] = df[col].dt.tz_localize(None)

        df.to_excel(excel_draft, index=False)
        print(f"[OK] Excel draft saved to {excel_draft}")
    except Exception as e:
        print(f"[WARN] Could not save Excel draft file: {e}")
        

    df = normalize_features(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"[OK] Saved normalized dataset -> {args.output}")
    print(f"[COLUMNS] {df.columns.tolist()}")


if __name__ == "__main__":
    main()
