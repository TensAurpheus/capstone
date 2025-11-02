"""
technical.py
---------------------------------
Generates technical indicators and session-level statistics for crypto OHLCV data.
Integrates TA-Lib-style indicators via pandas_ta and adds trading session features.

Usage:
  python src/data_pipeline/features/technical.py \
    --input data/processed/SOL_USDT_15m_features.parquet \
    --output data/processed/SOL_USDT_15m_technical.parquet
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys

# --- Clean console output encoding ---
sys.stdout.reconfigure(encoding='utf-8')

# --- Suppress unnecessary warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", message="DataFrameGroupBy.apply operated on the grouping columns")

# --- Try importing pandas_ta ---
try:
    import pandas_ta as pta
    HAS_PTA = True
except ImportError:
    HAS_PTA = False
    print("[WARN] pandas_ta not installed — skipping some indicators (MFI, VWAP, etc.)")


# === Session Assignment Function ===
def assign_session(hour):
    """
    Assigns trading session by UTC hour:
      - Asia: 01–06
      - Frankfurt: 07
      - London: 08–14
      - NewYork: 15–21
      - OffHours: others
    """
    if 1 <= hour <= 6:
        return "Asia"
    elif hour == 7:
        return "Frankfurt"
    elif 8 <= hour <= 14:
        return "London"
    elif 15 <= hour <= 21:
        return "NewYork"
    else:
        return "OffHours"


# === Core Function ===
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Adding technical indicators...")

    # Ensure datetime index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").set_index("timestamp", drop=True)
    elif not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()

    # === Trend indicators ===
    df["ema_20"] = pta.ema(df["close"], length=20)
    df["ema_50"] = pta.ema(df["close"], length=50)
    df["ema_200"] = pta.ema(df["close"], length=200)

    macd = pta.macd(df["close"])
    if macd is not None:
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"] = macd["MACDh_12_26_9"]

    adx = pta.adx(df["high"], df["low"], df["close"], length=14)
    if adx is not None:
        df["adx"] = adx["ADX_14"]

    # === Momentum ===
    df["rsi_14"] = pta.rsi(df["close"], length=14)

    # === Volatility ===
    bb = pta.bbands(df["close"], length=20)
    if bb is not None:
        df["bb_bbm"] = bb["BBM_20_2.0"]
        df["bb_bbh"] = bb["BBU_20_2.0"]
        df["bb_bbl"] = bb["BBL_20_2.0"]
        df["bb_percent"] = (df["close"] - df["bb_bbl"]) / (df["bb_bbh"] - df["bb_bbl"])
        df["bb_width"] = (df["bb_bbh"] - df["bb_bbl"]) / df["bb_bbm"]

    df["atr_14"] = pta.atr(df["high"], df["low"], df["close"], length=14)

    # === Volume-based ===
    if "volume" in df.columns:
        if HAS_PTA:
            df["mfi_14"] = pta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
        df["z_volume"] = (df["volume"] - df["volume"].rolling(32).mean()) / df["volume"].rolling(32).std()

    # === VWAP and distance ===
    if HAS_PTA and all(c in df.columns for c in ["high", "low", "close", "volume"]):
        df["vwap"] = pta.vwap(df["high"], df["low"], df["close"], df["volume"])
        df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]

    # === Log returns ===
    df["log_return_15m"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_1h"] = np.log(df["close"] / df["close"].shift(4))
    df["log_return_4h"] = np.log(df["close"] / df["close"].shift(16))
    df["log_return_1d"] = np.log(df["close"] / df["close"].shift(96))

    # === Rolling volatility ===
    df["roll_std_16"] = df["log_return_15m"].rolling(16).std()
    df["roll_std_32"] = df["log_return_15m"].rolling(32).std()
    
    # === Funding Rate Context ===
    if "funding_rate" in df.columns:
        df["funding_bias"] = np.sign(df["funding_rate"]).astype(int)
    else:
        df["funding_bias"] = 0

    # === Assign session ===
    df["hour"] = df.index.hour
    df["session"] = df["hour"].map(assign_session)

    # === Session-level statistics (by day + session) ===
    df["date"] = df.index.date
    session_stats = (
        df.groupby(["date", "session"])
        .agg({
            "volume": "mean",
            "log_return_15m": "mean",
            "roll_std_16": "mean"
        })
        .rename(columns={
            "volume": "session_vol_mean",
            "log_return_15m": "session_return_mean",
            "roll_std_16": "session_volatility"
        })
        .reset_index()
    )
    df = df.reset_index().merge(session_stats, on=["date", "session"], how="left")

    df.drop(columns=["hour"], inplace=True, errors="ignore")

    print("[OK] Technical indicators and session statistics added.")
    return df


# === CLI entrypoint ===
def main():
    parser = argparse.ArgumentParser(description="Add technical indicators and session-level stats to dataset.")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file path")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[INFO] Reading data from {input_path}")
    df = pd.read_parquet(input_path)

    df = add_technical_indicators(df)

    # === Save ===
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[OK] Saved enriched dataset to {output_path}")


if __name__ == "__main__":
    main()
