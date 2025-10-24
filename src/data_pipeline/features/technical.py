"""
technical.py
---------------------------------
Generates technical indicators for crypto price data.
Integrates TA-Lib-style indicators via pandas_ta, plus session-based metrics and PDA zones.

Usage:
  python src/data_pipeline/features/technical.py --input data/processed/BTC_USDT_15m_features.parquet --output data/processed/BTC_USDT_15m_technical.parquet
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# --- Global Configuration and Warnings ---
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)

# --- Додаємо фільтр для попередження про groupby().apply() ---
warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns",
    category=FutureWarning
)
# --- Кінець доданого фільтра ---

sys.stdout.reconfigure(encoding='utf-8')

# Try to import pandas_ta if available
try:
    import pandas_ta as pta
    HAS_PTA = True
except ImportError:
    HAS_PTA = False
    print("[WARN] pandas_ta not installed — skipping MFI and some volume-based indicators.")

# === Helper: assign session name (From third code) ===
def assign_session(hour):
    """Return trading session name based on UTC hour."""
    if 0 <= hour < 6:
        return "Asia"
    elif 6 <= hour < 8:
        return "Frankfurt"
    elif 8 <= hour < 16:
        return "London"
    else:  # 16 <= hour < 24
        return "NewYork"

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Adding technical indicators...")

    # Ensure timestamp is datetime and set as index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").set_index("timestamp", drop=True)
    elif not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()

    # --- Trend indicators ---
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

    # --- Momentum ---
    df["rsi_14"] = pta.rsi(df["close"], length=14)

    # --- Volatility ---
    bb = pta.bbands(df["close"], length=20)
    if bb is not None:
        df["bb_bbm"] = bb["BBM_20_2.0"]
        df["bb_bbh"] = bb["BBU_20_2.0"]
        df["bb_bbl"] = bb["BBL_20_2.0"]
        df["bb_percent"] = (df["close"] - df["bb_bbl"]) / (df["bb_bbh"] - df["bb_bbl"])
        df["bb_width"] = (df["bb_bbh"] - df["bb_bbl"]) / df["bb_bbm"]

    df["atr_14"] = pta.atr(df["high"], df["low"], df["close"], length=14)

    # --- Volume-based ---
    if "volume" in df.columns:
        df["mfi_14"] = pta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
        df["z_volume"] = (df["volume"] - df["volume"].rolling(32).mean()
                         ) / df["volume"].rolling(32).std()

    # --- VWAP and distance ---
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Converting to PeriodArray/Index representation will drop timezone information",
                category=UserWarning
            )
            df["vwap"] = pta.vwap(df["high"], df["low"], df["close"], df["volume"])
        df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]
    except Exception as e:
        print(f"[WARN] VWAP skipped: {e}")

    # --- Rolling volatility ---
    df["roll_std_16"] = df["close"].rolling(16).std()
    df["roll_std_32"] = df["close"].rolling(32).std()

    # --- Returns ---
    df["log_return_15m"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_1h"] = np.log(df["close"] / df["close"].shift(4))
    df["log_return_4h"] = np.log(df["close"] / df["close"].shift(16))
    df["log_return_1d"] = np.log(df["close"] / df["close"].shift(96))

    # --- Session assignment (from third code) ---
    if pd.api.types.is_datetime64_any_dtype(df.index):
        df["session"] = df.index.hour.map(assign_session)

        # === Session VWAP (reset within each session) (from third code) ===
        def compute_session_vwap_group(group):
            required_vwap_cols = ["high", "low", "close", "volume"]
            if not all(col in group.columns for col in required_vwap_cols) or group['volume'].isnull().all():
                return pd.Series(np.nan, index=group.index)

            price = (group["high"] + group["low"] + group["close"]) / 3
            cum_pv = (price * group["volume"]).cumsum()
            cum_vol = group["volume"].cumsum().replace(0, np.nan)
            return cum_pv / cum_vol

        required_vwap_cols_for_check = ["high", "low", "close", "volume"]
        if all(col in df.columns for col in required_vwap_cols_for_check) and "session" in df.columns:
            # Рядок повернено до початкового вигляду, попередження ігнорується фільтром
            df["vwap_session"] = df.groupby("session", group_keys=False).apply(compute_session_vwap_group)
        else:
            print(f"[WARN] Session VWAP skipped: Missing 'session' column or one of required price/volume columns ({', '.join(required_vwap_cols_for_check)}).")
    else:
        print("[WARN] Session-based indicators (session, vwap_session) skipped: DataFrame index is not datetime.")

    df = df.reset_index(names='timestamp')

    print("[OK] Technical indicators added successfully.")
    return df


def main():
    parser = argparse.ArgumentParser(description="Add technical indicators to dataset.")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file path")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file path")
    args = parser.parse_args()

    # --- Видаляємо тимчасовий рядок для налагодження версії pandas ---
    # print(f"DEBUG: pandas version is {pd.__version__}")

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[INFO] Reading data from {input_path}")
    df = pd.read_parquet(input_path)

    df = add_technical_indicators(df)

    # === Save ===
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[OK] Saved enriched dataset to {output_path}")
    print("[COLUMNS]", df.columns.tolist())


if __name__ == "__main__":
    main()
