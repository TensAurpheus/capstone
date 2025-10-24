"""
patterns.py
---------------------------------
Detects key price-action patterns and market structure features.

Usage:
  python src/data_pipeline/features/patterns.py --input data/processed/BTC_USDT_15m_technical.parquet --output data/processed/BTC_USDT_15m_patterns.parquet
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure UTF-8 on Windows
sys.stdout.reconfigure(encoding="utf-8")

# Optional TA-Lib
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("[WARN] TA-Lib not installed — candlestick pattern detection limited.")


# === A. CANDLESTICK PATTERNS ===
def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect key reversal candlestick patterns using TA-Lib."""
    print("[INFO] Detecting candlestick patterns...")

    if HAS_TALIB:
        df["pattern_bullish_engulf"] = talib.CDLENGULFING(df["open"], df["high"], df["low"], df["close"]) > 0
        df["pattern_bearish_engulf"] = talib.CDLENGULFING(df["open"], df["high"], df["low"], df["close"]) < 0
        df["pattern_harami"] = talib.CDLHARAMI(df["open"], df["high"], df["low"], df["close"]) != 0
        df["pattern_hammer"] = talib.CDLHAMMER(df["open"], df["high"], df["low"], df["close"]) != 0
        df["pattern_inverted_hammer"] = talib.CDLINVERTEDHAMMER(df["open"], df["high"], df["low"], df["close"]) != 0
    else:
        for col in ["pattern_bullish_engulf", "pattern_bearish_engulf",
                    "pattern_harami", "pattern_hammer", "pattern_inverted_hammer"]:
            df[col] = False

    return df


# === B. FRACTALS & STRONG LEVELS ===
def detect_fractals_and_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Detect fractal points and confirm strong highs/lows."""
    print("[INFO] Detecting fractals and structure levels...")

    df["fract_high"] = (df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))
    df["fract_low"] = (df["low"] < df["low"].shift(1)) & (df["low"] < df["low"].shift(-1))

    df["strong_high"] = df["fract_high"] & (df["close"].shift(-2) < df["high"])
    df["strong_low"] = df["fract_low"] & (df["close"].shift(-2) > df["low"])

    return df


# === C. FAIR VALUE GAPS (FVG) ===
def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """Detect fair value gaps (bullish/bearish imbalance)."""
    print("[INFO] Detecting Fair Value Gaps (FVG)...")

    df["bullish_fvg"] = df["low"].shift(1) > df["high"].shift(-1)
    df["bearish_fvg"] = df["high"].shift(1) < df["low"].shift(-1)

    df["fvg_gap"] = np.where(df["bullish_fvg"],
                             df["low"].shift(1) - df["high"].shift(-1),
                             np.where(df["bearish_fvg"],
                                      df["high"].shift(-1) - df["low"].shift(1),
                                      0))

    return df


# === D. PDA / TDA ZONES ===
def compute_pda_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Compute premium/discount arrays relative to equilibrium."""
    print("[INFO] Calculating PDA (Premium / Discount Arrays)...")

    df["rolling_high"] = df["high"].rolling(100, min_periods=1).max()
    df["rolling_low"] = df["low"].rolling(100, min_periods=1).min()
    df["equilibrium"] = (df["rolling_high"] + df["rolling_low"]) / 2
    df["pda"] = np.where(df["close"] > df["equilibrium"], "Premium", "Discount")

    return df


# === E. BREAKOUTS ===
def detect_breakouts(df: pd.DataFrame) -> pd.DataFrame:
    """Detect bullish/bearish breakouts with volume confirmation."""
    print("[INFO] Detecting breakouts (GG-Shot)...")

    df["breakout_bullish"] = (df["close"] > df["rolling_high"].shift(1)) & (
        df["volume"] > df["volume"].rolling(10).mean()
    )
    df["breakout_bearish"] = (df["close"] < df["rolling_low"].shift(1)) & (
        df["volume"] > df["volume"].rolling(10).mean()
    )

    return df


# === F. MAIN FUNCTION ===
def generate_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all pattern detections into one enriched dataset."""
    print("[INFO] Detecting price-action and structure patterns...")

    df = detect_candlestick_patterns(df)
    df = detect_fractals_and_levels(df)
    df = detect_fvg(df)
    df = compute_pda_zones(df)
    df = detect_breakouts(df)

    drop_cols = ["bullish_fvg", "bearish_fvg", "fract_high", "fract_low"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Aggregate info
    pattern_cols = [
        "pattern_bullish_engulf", "pattern_bearish_engulf", "pattern_harami",
        "pattern_hammer", "pattern_inverted_hammer","strong_high", "strong_low",
        "breakout_bullish", "breakout_bearish"
    ]

    # Convert True/False to int for summing
    for c in pattern_cols:
        df[c] = df[c].astype(int)

    df["pattern_count"] = df[pattern_cols].sum(axis=1)
    df["pattern_active"] = df["pattern_count"] > 0

    print("[OK] Pattern detection completed.")
    return df


# === CLI EXECUTION ===
def main():
    parser = argparse.ArgumentParser(description="Detect patterns in crypto price data.")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[INFO] Reading input: {input_path}")
    df = pd.read_parquet(input_path)

    df = generate_patterns(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"[OK] Saved -> {output_path}")
    print("[COLUMNS]", df.columns.tolist())


if __name__ == "__main__":
    main()

