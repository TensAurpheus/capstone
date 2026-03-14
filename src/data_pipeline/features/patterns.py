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
        df["pattern_bullish_engulf"] = (
            talib.CDLENGULFING(df["open"], df["high"], df["low"], df["close"]) > 0
        )
        df["pattern_bearish_engulf"] = (
            talib.CDLENGULFING(df["open"], df["high"], df["low"], df["close"]) < 0
        )
        df["pattern_harami"] = (
            talib.CDLHARAMI(df["open"], df["high"], df["low"], df["close"]) != 0
        )
        df["pattern_hammer"] = (
            talib.CDLHAMMER(df["open"], df["high"], df["low"], df["close"]) != 0
        )
        df["pattern_inverted_hammer"] = (
            talib.CDLINVERTEDHAMMER(df["open"], df["high"], df["low"], df["close"]) != 0
        )
    else:
        for col in [
            "pattern_bullish_engulf",
            "pattern_bearish_engulf",
            "pattern_harami",
            "pattern_hammer",
            "pattern_inverted_hammer",
        ]:
            df[col] = False

    return df


# =====================================================================
# B) ICT MARKET STRUCTURE (pivot → swing → BOS → CHoCH → MSS)
# =====================================================================
def apply_ict_market_structure(
    df,
):  # <<< NEW FUNCTION (extracted from generate_patterns)
    print("[INFO] Applying ICT market structure...")

    # ---------------------------------------------------------------
    # 1) TRUE CAUSAL PIVOTS
    # ---------------------------------------------------------------
    df["swing_high"] = (
        (df["high"].shift(1) > df["high"].shift(2)) & (df["high"].shift(1) > df["high"])
    ).fillna(False)

    df["swing_low"] = (
        (df["low"].shift(1) < df["low"].shift(2)) & (df["low"].shift(1) < df["low"])
    ).fillna(False)

    # ---------------------------------------------------------------
    # OPTIONAL IMPROVEMENT — clean leading garbage swings
    # ---------------------------------------------------------------  # <<< ADDED
    first_high = df["swing_high"].idxmax()
    first_low = df["swing_low"].idxmax()
    first_real = min(first_high, first_low)
    df.loc[:first_real, ["swing_high", "swing_low"]] = False

    # ---------------------------------------------------------------
    # 2) PROTECTED SWINGS
    # ---------------------------------------------------------------
    df["last_swing_high"] = np.where(df["swing_high"], df["high"].shift(1), np.nan)
    df["last_swing_low"] = np.where(df["swing_low"], df["low"].shift(1), np.nan)

    df["last_swing_high"] = df["last_swing_high"].ffill()
    df["last_swing_low"] = df["last_swing_low"].ffill()

    # ---------------------------------------------------------------
    # 3) BOS
    # ---------------------------------------------------------------

    df["bos_bullish"] = (df["close"] > df["last_swing_high"].shift(1)).fillna(False)
    df["bos_bearish"] = (df["close"] < df["last_swing_low"].shift(1)).fillna(False)

    df["bos_bullish"] = df["bos_bullish"].astype(bool)
    df["bos_bearish"] = df["bos_bearish"].astype(bool)
    df = df.infer_objects(copy=False)

    df["bos_bullish"] = df["bos_bullish"].astype("bool")
    df["bos_bearish"] = df["bos_bearish"].astype("bool")

    # ---------------------------------------------------------------
    # 4) CHoCH
    # ---------------------------------------------------------------
    prev_bull = df["bos_bullish"].shift(1)
    prev_bull = prev_bull.where(prev_bull.notna(), False)

    prev_bear = df["bos_bearish"].shift(1)
    prev_bear = prev_bear.where(prev_bear.notna(), False)

    df["choch_bullish"] = df["bos_bullish"] & prev_bear
    df["choch_bearish"] = df["bos_bearish"] & prev_bull

    # ---------------------------------------------------------------
    # 5) MSS — internal HH/LL
    # ---------------------------------------------------------------
    int_high = df["high"].rolling(5, min_periods=2).max()
    int_low = df["low"].rolling(5, min_periods=2).min()

    df["mss_bullish"] = (df["close"] > int_high.shift(1)).fillna(False)
    df["mss_bearish"] = (df["close"] < int_low.shift(1)).fillna(False)

    return df


# =====================================================================
# C) Fair Value Gaps (ICT-style causal)
# =====================================================================
def detect_fvg(df):
    print("[INFO] Detecting causal Fair Value Gaps...")

    # ICT FVG convention:
    # bullish: low[t] > high[t-2]
    # causal version uses the same logic reversed in time
    # (your version is equivalent but slightly different ordering)

    bullish = df["low"].shift(2) > df["high"]
    bearish = df["high"].shift(2) < df["low"]

    gap = pd.Series(
        np.where(
            bullish,
            df["low"].shift(2) - df["high"],
            np.where(bearish, df["high"] - df["low"].shift(2), 0),
        ),
        index=df.index,
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

    df["rolling_high"] = df["high"].rolling(25, min_periods=1).max()
    df["rolling_low"] = df["low"].rolling(25, min_periods=1).min()
    df["equilibrium"] = (df["rolling_high"] + df["rolling_low"]) / 2
    df["pda"] = np.where(df["close"] >= df["equilibrium"], "Premium", "Discount")

    return df


# =====================================================================
# E) Volume + volatility breakouts
# =====================================================================
def detect_breakouts(df):
    print("[INFO] Detecting breakouts...")

    vol_ma = df["volume"].rolling(10).mean()

    rolling_high = df["high"].rolling(25).max()
    rolling_low = df["low"].rolling(25).min()

    df["breakout_bullish"] = (df["close"] > rolling_high.shift(1)) & (
        df["volume"] > vol_ma
    )
    df["breakout_bearish"] = (df["close"] < rolling_low.shift(1)) & (
        df["volume"] > vol_ma
    )

    return df


# =====================================================================
# MASTER WRAPPER
# =====================================================================
def generate_patterns(df):
    print("[INFO] Generating all pattern features...")

    assert "timestamp" in df.columns, "❌ Missing timestamp BEFORE patterns.py!"

    df = detect_candlestick_patterns(df)
    df = apply_ict_market_structure(df)
    df = detect_fvg(df)
    df = compute_pda_zones(df)
    df = detect_breakouts(df)

    # Final checks
    assert "timestamp" in df.columns, "❌ Timestamp LOST inside patterns.py!"

    # Convert booleans → ints
    pattern_cols = [
        "pattern_bullish_engulf",
        "pattern_bearish_engulf",
        "pattern_harami",
        "pattern_hammer",
        "pattern_inverted_hammer",
        "swing_high",
        "swing_low",
        "bos_bullish",
        "bos_bearish",
        "choch_bullish",
        "choch_bearish",
        "mss_bullish",
        "mss_bearish",
        "bullish_fvg",
        "bearish_fvg",
        "breakout_bullish",
        "breakout_bearish",
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
