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
    Convert dataset to scale-free features (ATR-relative) using long-term ATR (atr_200).
    """
    print("\n=== SCALE-FREE NORMALIZATION STARTED ===")

    # =====================================================
    # === ATR — MAIN NORMALIZER (USE ATR-200) ============
    # =====================================================
    if "atr_200" not in df.columns:
        raise ValueError("atr_200 is required but missing. Compute it in technical.py.")

    atr = df["atr_200"].replace(0, np.nan).clip(lower=1e-8)

    # =====================================================
    # === SCALE-FREE TRANSFORMATIONS ======================
    # =====================================================
    mappings = {
        "ema_20":      (df["close"] - df["ema_20"]) / atr,
        "ema_50":      (df["close"] - df["ema_50"]) / atr,
        "ema_200":     (df["close"] - df["ema_200"]) / atr,
        "rolling_high": (df["rolling_high"] - df["close"]) / atr,
        "rolling_low": (df["close"] - df["rolling_low"]) / atr,
        "equilibrium": (df["close"] - df["equilibrium"]) / atr,
        "vwap_session": (df["close"] - df["vwap_session"]) / atr,
        "vwap":         (df["close"] - df["vwap"]) / atr,
        "volume":       np.log1p(df["volume"]),
        "z_volume":     df["z_volume"],
    }

    for col, expr in mappings.items():
        if col in df.columns:
            df[col] = expr

    # ATR-relative volatility metrics
    df["atr_pct"] = df["atr_200"] / df["close"]
    df["range_atr"] = (df["high"] - df["low"]) / df["atr_200"]
    df["body_atr"] = (df["close"] - df["open"]) / df["atr_200"]

    # =====================================================
    # === Distance to Daily/Weekly High/Low (ATR200 norm)
    # =====================================================
    df["dist_daily_high"] = (df["close"] - df["daily_high"]) / atr
    df["dist_daily_low"]  = (df["daily_low"] - df["close"]) / atr

    df["dist_weekly_high"] = (df["close"] - df["weekly_high"]) / atr
    df["dist_weekly_low"]  = (df["weekly_low"] - df["close"]) / atr

    df.drop(columns=["daily_high", "daily_low", "weekly_high", "weekly_low"], inplace=True)

    # =====================================================
    # === FVG gap normalization ===========================
    # =====================================================
    if "fvg_gap" in df.columns and df["fvg_gap"].nunique() > 2:
        df["fvg_gap"] = df["fvg_gap"] / atr

    # =====================================================
    # === ATR Volatility Regime ===========================
    # =====================================================
    if "atr_vol_regime" in df.columns:
        df["atr_vol_regime"] = df["atr_vol_regime"] / atr

    if "atr_vol_regime" in df.columns:
        win = 24 * 10
        eps = 1e-8

        roll_mean = df["atr_vol_regime"].rolling(win, min_periods=win//2).mean()
        roll_std  = df["atr_vol_regime"].rolling(win, min_periods=win//2).std()

        df["atr_vol_regime_z"] = (
            (df["atr_vol_regime"] - roll_mean) /
            roll_std.replace(0, eps)
        ).clip(-5, 5)

    # =====================================================
    # === PPO instead of MACD =============================
    # =====================================================
    if {"macd", "macd_signal", "macd_hist"}.issubset(df.columns):
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()

        df["ppo"] = (ema_fast - ema_slow) / ema_slow.clip(lower=1e-12)
        df["ppo_signal"] = df["ppo"].ewm(span=9, adjust=False).mean()
        df["ppo_hist"] = df["ppo"] - df["ppo_signal"]

        df.drop(columns=["macd", "macd_signal", "macd_hist"], inplace=True)
        print("[INFO] Replaced MACD with PPO.")

    # =====================================================
    # === Replace BB with Z-score form ====================
    # =====================================================
    if {"bb_bbm", "bb_bbh", "bb_bbl"}.issubset(df.columns):
        k = 2.0
        bb_sigma = ((df["bb_bbh"] - df["bb_bbl"]) / (2 * k)).clip(lower=1e-12)

        df["bb_z"] = (df["close"] - df["bb_bbm"]) / bb_sigma
        df["bb_percB"] = (df["close"] - df["bb_bbl"]) / \
            (df["bb_bbh"] - df["bb_bbl"]).clip(lower=1e-12)

        df.drop(columns=["bb_bbm", "bb_bbh", "bb_bbl", "atr_14"], inplace=True)
        print("[INFO] Replaced BB values with bb_z and %B.")

    # =====================================================
    # === NaN handling (NO FD BACKFILL) ===================
    # =====================================================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    non_fd_cols = [col for col in df.columns if not col.startswith("fd_")]
    df[non_fd_cols] = df[non_fd_cols].ffill()

    # Drop unstable region
    df = df.iloc[200:].reset_index(drop=True)
    print("[INFO] Dropped first 200 rows (ATR200 warm-up).")

    print("[OK] Scale-free normalization complete.\n")
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
