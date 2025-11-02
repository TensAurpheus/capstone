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
        "atr_14": df["atr_14"] / df["close"],  # relative volatility
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

    # FVG gap normalization
    if "fvg_gap" in df.columns and df["fvg_gap"].nunique() > 2:
        df["fvg_gap"] = df["fvg_gap"] / atr

    print("[OK] Scale-free normalization complete.")
    print("\n=== NORMALIZATION COMPLETED ===")
    return df


def main():
    parser = argparse.ArgumentParser(description="ATR-based normalization for ML models")
    parser.add_argument("--input", type=str, required=True, help="Input .parquet file (after patterns.py)")
    parser.add_argument("--output", type=str, required=True, help="Output .parquet file")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol name (e.g. BTC/USDT)")
    args = parser.parse_args()

    print(f"\n[INFO] Loading dataset → {args.input}")
    df = pd.read_parquet(args.input)
    print(f"[INFO] Loaded {len(df):,} rows, {len(df.columns)} columns")

    df = normalize_features(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"[OK] Saved normalized dataset → {args.output}")
    print(f"[COLUMNS] {df.columns.tolist()}")


if __name__ == "__main__":
    main()
