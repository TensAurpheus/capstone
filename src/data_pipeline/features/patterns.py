# patterns.py
"""
Detects key price-action patterns and market structure features without leakage.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Try TA-Lib
try:
    import talib
    HAS_TALIB = True
except:
    HAS_TALIB = False
    print("[WARN] TA-Lib not installed — candlestick patterns limited.")


# =====================================================================
# A) Candlestick patterns
# =====================================================================
def detect_candlestick_patterns(df):
    print("[INFO] Detecting candlestick patterns...")

    if HAS_TALIB:
        df["pattern_bullish_engulf"] = (talib.CDLENGULFING(df["open"], df["high"], df["low"], df["close"]) > 0)
        df["pattern_bearish_engulf"] = (talib.CDLENGULFING(df["open"], df["high"], df["low"], df["close"]) < 0)
        df["pattern_harami"] = talib.CDLHARAMI(df["open"], df["high"], df["low"], df["close"]) != 0
        df["pattern_hammer"] = talib.CDLHAMMER(df["open"], df["high"], df["low"], df["close"]) != 0
        df["pattern_inverted_hammer"] = talib.CDLINVERTEDHAMMER(df["open"], df["high"], df["low"], df["close"]) != 0
    else:
        for col in [
            "pattern_bullish_engulf", "pattern_bearish_engulf",
            "pattern_harami", "pattern_hammer", "pattern_inverted_hammer"
        ]:
            df[col] = False

    return df


# =====================================================================
# B) Fractals (causal) + protected swings (ICT)
# =====================================================================
def detect_fractals_and_market_structure(df):
    print("[INFO] Detecting fractals, protected swings, BOS/CHoCH/MSS...")

    # ===== 1) Causal fractals =====
    df["fract_high"] = (
        (df["high"].shift(1) > df["high"].shift(2)) &   # t-1 > t-2
        (df["high"] > df["high"].shift(1))              # t > t-1
    ).fillna(0).astype(bool)

    df["fract_low"] = (
        (df["low"].shift(1) < df["low"].shift(2)) &     # t-1 < t-2
        (df["low"] < df["low"].shift(1))                # t < t-1
    ).fillna(0).astype(bool)

    # ===== 2) Protected swings =====
    df["protected_high"] = (
        df["fract_high"] &
        (df["high"] > df["high"].shift(2))
    ).fillna(0).astype(bool)

    df["protected_low"] = (
        df["fract_low"] &
        (df["low"] < df["low"].shift(2))
    ).fillna(0).astype(bool)

    # ===== 3) BOS (break of structure) =====
    df["bos_bullish"] = (
        df["protected_high"].shift(1).fillna(0).astype(bool) &
        (df["close"] > df["high"].shift(1))
    ).fillna(0).astype(bool)

    df["bos_bearish"] = (
        df["protected_low"].shift(1).fillna(0).astype(bool) &
        (df["close"] < df["low"].shift(1))
    ).fillna(0).astype(bool)

    # ===== 4) CHoCH =====
    prev_up = df["bos_bullish"].shift(1).fillna(0).astype(bool)
    prev_down = df["bos_bearish"].shift(1).fillna(0).astype(bool)

    df["choch_bullish"] = (df["bos_bullish"] & prev_down).fillna(0).astype(bool)
    df["choch_bearish"] = (df["bos_bearish"] & prev_up).fillna(0).astype(bool)

    # ===== 5) MSS (internal structure shift) =====
    df["mss_bullish"] = (
        df["fract_low"].shift(1).fillna(0).astype(bool) &
        (df["close"] > df["high"].shift(2))
    ).fillna(0).astype(bool)

    df["mss_bearish"] = (
        df["fract_high"].shift(1).fillna(0).astype(bool) &
        (df["close"] < df["low"].shift(2))
    ).fillna(0).astype(bool)

    return df


# =====================================================================
# C) Fair Value Gaps (causal)
# =====================================================================
def detect_fvg(df):
    print("[INFO] Detecting causal Fair Value Gaps...")

    bullish = (df["low"].shift(2) > df["high"])
    bearish = (df["high"].shift(2) < df["low"])

    gap = pd.Series(
        np.where(
            bullish,
            df["low"].shift(2) - df["high"],
            np.where(bearish, df["high"] - df["low"].shift(2), 0)
        ),
        index=df.index
    )

    df["bullish_fvg"] = bullish.fillna(0).astype(int)
    df["bearish_fvg"] = bearish.fillna(0).astype(int)
    df["fvg_gap"] = gap.fillna(0)

    return df


# =====================================================================
# D) PDA zones
# =====================================================================
def compute_pda_zones(df):
    print("[INFO] Computing PDA zones...")

    df["rolling_high"] = df["high"].rolling(100, min_periods=1).max()
    df["rolling_low"] = df["low"].rolling(100, min_periods=1).min()
    df["equilibrium"] = (df["rolling_high"] + df["rolling_low"]) / 2
    df["pda"] = np.where(df["close"] >= df["equilibrium"], "Premium", "Discount")

    return df


# =====================================================================
# E) Volume + volatility breakouts
# =====================================================================
def detect_breakouts(df):
    print("[INFO] Detecting breakouts...")

    vol_ma = df["volume"].rolling(10).mean()

    df["breakout_bullish"] = (df["close"] > df["rolling_high"].shift(1)) & (df["volume"] > vol_ma)
    df["breakout_bearish"] = (df["close"] < df["rolling_low"].shift(1)) & (df["volume"] > vol_ma)

    return df


# =====================================================================
# MASTER WRAPPER
# =====================================================================
def generate_patterns(df):
    print("[INFO] Generating all pattern features...")

    assert "timestamp" in df.columns, "❌ Missing timestamp BEFORE patterns.py!"

    df = detect_candlestick_patterns(df)
    df = detect_fractals_and_market_structure(df)
    df = detect_fvg(df)
    df = compute_pda_zones(df)
    df = detect_breakouts(df)

    assert "timestamp" in df.columns, "❌ Timestamp LOST inside patterns.py!"

    # Binary pattern count
    pattern_cols = [
        "pattern_bullish_engulf", "pattern_bearish_engulf",
        "pattern_harami", "pattern_hammer", "pattern_inverted_hammer",
        "protected_high", "protected_low",
        "bos_bullish", "bos_bearish",
        "choch_bullish", "choch_bearish",
        "mss_bullish", "mss_bearish",
        "bullish_fvg", "bearish_fvg",
        "breakout_bullish", "breakout_bearish"
    ]

    for c in pattern_cols:
        df[c] = df[c].astype(int)

    df["pattern_count"] = df[pattern_cols].sum(axis=1)
    df["pattern_active"] = (df["pattern_count"] > 0).astype(int)

    return df


# =====================================================================
# CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df = generate_patterns(df)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print("[OK] Saved:", args.output)


if __name__ == "__main__":
    main()
