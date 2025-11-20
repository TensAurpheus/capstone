"""
technical.py
---------------------------------
Generates technical indicators and session-level statistics for crypto OHLCV data.
Integrates TA-Lib-style indicators via pandas_ta and adds trading session features.

Usage:
  python src/data_pipeline/features/technical.py \
    --input data/processed/BTC_USDT_15m_features.parquet \
    --output data/processed/BTC_USDT_15m_technical.parquet
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
    df["atr_200"] = pta.atr(df["high"], df["low"], df["close"], length=200)
    df["atr_vol_regime"] = df["atr_14"] / df["atr_200"].clip(lower=1e-8)

    # === Volume-based ===
    if "volume" in df.columns:
        if HAS_PTA:
            df["mfi_14"] = pta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
        df["z_volume"] = (df["volume"] - df["volume"].rolling(32).mean()) / df["volume"].rolling(32).std()

    # === VWAP and distance ===
    if HAS_PTA:
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

    # === Assign trading sessions ===
    df["hour"] = df.index.hour
    df["session"] = df["hour"].map(assign_session)
    df["date"] = df.index.date

    # === CAUSAL Session VWAP ===
    print("[INFO] Computing causal session VWAP...")
    price = (df["high"] + df["low"] + df["close"]) / 3

    df["vwap_session"] = (
        df.groupby(["date", "session"])
          .apply(lambda g: (price.loc[g.index] * g["volume"]).cumsum() /
                           g["volume"].cumsum().replace(0, np.nan))
          .droplevel([0, 1])
    )

    # === CAUSAL Session Statistics (expanding) ===
    print("[INFO] Computing causal session statistics...")
    g = df.groupby(["date", "session"])

    df["session_vol_mean"] = (
        g["volume"].expanding().mean().reset_index(level=[0,1], drop=True)
    )

    df["session_return_mean"] = (
        g["log_return_15m"].expanding().mean().reset_index(level=[0,1], drop=True)
    )

    df["session_volatility"] = (
        g["roll_std_16"].expanding().mean().reset_index(level=[0,1], drop=True)
    )

    df.drop(columns=["hour"], inplace=True, errors="ignore")

    print("[OK] Technical indicators + causal session features added.")

    # ============================================================
    # === Higuchi Fractal Dimension (FD) — CAUSAL, NO LEAK =======
    # ============================================================

    print("[INFO] Computing Fractal Dimension (Higuchi FD)...")

    def higuchi_fd(series, kmax=5):
        y = np.array(series, dtype=float)
        N = len(y)
        Lk = []
        for k in range(1, kmax + 1):
            Lm = []
            for m in range(k):
                idx = np.arange(m, N, k)
                if len(idx) < 2:
                    continue
                Lm_val = np.sum(np.abs(np.diff(y[idx]))) * (N - 1) / (len(idx) * k)
                Lm.append(Lm_val)
            if Lm:
                Lk.append(np.mean(Lm))
        if len(Lk) < 2:
            return np.nan
        logs = np.log(1.0 / np.arange(1, len(Lk) + 1))
        return np.polyfit(logs, np.log(Lk), 1)[0]

    fd_window = 96  # 1 day at 15m
    close_vals = df["close"].values
    fd_vals = [np.nan] * len(df)

    for i in range(fd_window, len(df)):
        fd_vals[i] = higuchi_fd(close_vals[i - fd_window:i])

    df["fd_96"] = fd_vals

    # === FD slope ======================================================
    df["fd_slope"] = df["fd_96"].diff()

    print("[INFO] FD computed.")


    # ============================================================
    # === EMA smoothing ==========================================
    # ============================================================
    df["fd_ema_12"] = df["fd_96"].ewm(span=12, adjust=False, min_periods=12).mean()
    df["fd_ema_24"] = df["fd_96"].ewm(span=24, adjust=False, min_periods=24).mean()
    df["fd_trend_strength"] = df["fd_96"] - df["fd_ema_24"]


    # ============================================================
    # === Adaptive FD Regime (CAUSAL QUANTILE) ==================
    # ============================================================

    print("[INFO] Computing causal FD regime with expanding quantile...")

    df["fd_threshold_causal"] = (
        df["fd_ema_24"]
        .expanding(min_periods=96)
        .quantile(0.70)
    )

    df["fd_regime"] = (df["fd_ema_24"] >= df["fd_threshold_causal"]).astype(int)

    # regime transitions (very strong feature)
    df["fd_regime_switch"] = df["fd_regime"].diff().fillna(0)

    last_thr = df["fd_threshold_causal"].iloc[-1]
    print(f"[INFO] Latest causal FD regime threshold = {last_thr:.4f}")


    # ============================================================
    # === FD × Volatility interaction (using ATR-200) ============
    # ============================================================

    atr = df["atr_200"].replace(0, np.nan).clip(lower=1e-8)

    df["fd_volatility"] = df["fd_96"] * atr
    df["fd_vol_ratio"] = df["fd_96"] / atr
    df["fd_vol_slope"] = df["fd_slope"] * atr

    # normalize slope by ATR (strong feature)
    df["fd_slope_atr_norm"] = df["fd_slope"] / atr


    # ============================================================
    # === FD Entropy (local FD uncertainty) =======================
    # ============================================================

    df["fd_entropy"] = (
        df["fd_slope"]
        .rolling(96, min_periods=48)
        .std()
    )


    # ============================================================
    # === Volatility-adjusted FD (Fractal Market Hypothesis) =====
    # ============================================================

    df["fd_vol_adjusted"] = df["fd_96"] / (1 + df["atr_vol_regime"])


    # ============================================================
    # === Robust FD Normalization (MAD Z-score) ==================
    # ============================================================

    print("[INFO] Computing robust FD normalization...")

    fd_cols = [
        "fd_96",
        "fd_ema_12",
        "fd_ema_24",
        "fd_trend_strength",
        "fd_slope",
        "fd_slope_atr_norm",
        "fd_volatility",
        "fd_vol_ratio",
        "fd_vol_slope",
        "fd_entropy",
        "fd_vol_adjusted",
    ]

    win = 96 * 10  # 10 days
    for col in fd_cols:
        if col in df.columns:
            median = df[col].rolling(win, min_periods=win//2).median()
            mad = (np.abs(df[col] - median)).rolling(win, min_periods=win//2).median()
            mad = mad.replace(0, np.nan)

            df[col + "_robust_z"] = (
                (df[col] - median) / (1.4826 * mad)
            ).clip(-5, 5)

    print("[OK] FD block enhanced and normalized.")
    
    # === Daily & Weekly High/Low (CAUSAL: previous day/week) ============
    print("[INFO] Computing Daily and Weekly High/Low (previous)...")

    # ----- Previous Daily High/Low -----
    daily = df.resample("1D").agg({"high": "max", "low": "min"})
    daily.columns = ["daily_high", "daily_low"]

    # робимо ці рівні "рівнями вчорашнього дня"
    daily[["daily_high", "daily_low"]] = daily[["daily_high", "daily_low"]].shift(1)

    df["date"] = df.index.normalize()
    df = df.merge(daily, left_on="date", right_index=True, how="left")

    df["daily_high"] = df["daily_high"].ffill()
    df["daily_low"] = df["daily_low"].ffill()

    # ----- Previous Weekly High/Low -----
    # групуємо по тижнях (PeriodIndex), щоб було стабільно
    weekly = df.groupby(df.index.to_period("W")).agg({"high": "max", "low": "min"})
    weekly.columns = ["weekly_high", "weekly_low"]

    # знову ж робимо "минулого тижня"
    weekly = weekly.shift(1)

    df["week_period"] = df.index.to_period("W")
    df = df.merge(weekly, left_on="week_period", right_index=True, how="left")

    df["weekly_high"] = df["weekly_high"].ffill()
    df["weekly_low"] = df["weekly_low"].ffill()
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

    # Ensure timestamp column exists (parquet does not preserve index)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "timestamp"})


    # === Save ===
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[OK] Saved enriched dataset to {output_path}")


if __name__ == "__main__":
    main()
