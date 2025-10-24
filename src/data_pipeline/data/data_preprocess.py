"""
data_preprocess.py
---------------------------------
Fetches OHLCV (and optionally funding rate) data from Binance Spot or Futures,
performs preprocessing ( direction, QA),
and saves ready-to-use dataset.

Usage:
  python src/data_pipeline/data/data_preprocess.py --symbol BTCUSDT --market futures --start 2018-01-01 --end 2025-01-01 --timeframe 15m
"""

import sys
import argparse
import time
from pathlib import Path
import warnings

import ccxt
import numpy as np
import pandas as pd

# Ensure UTF-8 output (especially for Windows)
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------- OHLCV ----------------
def fetch_ohlcv(symbol: str, timeframe: str, start_utc: pd.Timestamp,
                end_utc: pd.Timestamp, market_type: str) -> pd.DataFrame:
    """Fetch OHLCV from Binance Spot or USDT-M Futures (ccxt)."""

    # Select correct exchange
    if market_type == "futures":
        exchange = ccxt.binance({
            "options": {"defaultType": "future"}  # ✅ more reliable for USDT-M futures
        })
    else:
        exchange = ccxt.binance()

    exchange.load_markets()

    # ✅ Ensure symbol in Binance format (add "/" if missing)
    if "/" not in symbol and len(symbol) > 4:
        if symbol.endswith("USDT"):
            symbol = symbol[:-4] + "/USDT"
        elif symbol.endswith("USDC"):
            symbol = symbol[:-4] + "/USDC"

    # ✅ Try to reload markets and fallback if not found
    if symbol not in exchange.markets:
        print(f"[WARN] {symbol} not found on first load — retrying...")
        time.sleep(1)
        exchange.load_markets(True)
        if symbol not in exchange.markets:
            print("[WARN] Reload failed, trying alternative exchange (binanceusdm)...")
            exchange = ccxt.binanceusdm()
            exchange.load_markets()
            if symbol not in exchange.markets:
                raise ValueError(f"Symbol {symbol} not found on Binance {market_type} even after retry.")

    since_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    step_ms = exchange.parse_timeframe(timeframe) * 1000
    limit = 1500

    print(f"[INFO] Fetching {symbol} ({market_type}) {timeframe} data...")

    all_rows = []
    while since_ms < end_ms:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not candles:
            break
        all_rows += candles
        since_ms = candles[-1][0] + step_ms
        if len(all_rows) % 50000 == 0:
            print(f"[INFO] Fetched {len(all_rows)} rows...")
        time.sleep(0.15)

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    print(f"[OK] OHLCV done: {len(df):,} rows")
    return df



# ------------- FUNDING RATE -------------
def _call_funding_endpoint(exchange, params):
    """
    Try multiple ccxt method names for Binance Futures funding endpoint,
    because names vary across ccxt versions.
    """
    candidates = [
        "fapiPublicGetFundingRate",
        "publicGetFapiV1FundingRate",
        "fapiPublic_get_fundingrate",
        "public_linear_get_funding_rate",
    ]
    for name in candidates:
        if hasattr(exchange, name):
            return getattr(exchange, name)(params)
    raise AttributeError("No funding endpoint available in this ccxt version.")


def fetch_funding_rate(symbol: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    """Fetch Binance USDT-M Futures funding rate (8h)."""
    exchange = ccxt.binanceusdm()
    exchange.load_markets()

    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)

    params = {
        "symbol": symbol.replace("/", ""),  # e.g. BTCUSDT
        "startTime": start_ms,
        "limit": 1000,
    }

    print(f"[INFO] Fetching funding rates for {symbol} ...")
    all_rows = []
    while True:
        try:
            resp = _call_funding_endpoint(exchange, params)
        except Exception as e:
            print(f"[WARN] Funding endpoint error: {e}")
            break

        if not resp:
            break

        all_rows += resp
        last_time = int(resp[-1]["fundingTime"])
        if last_time >= end_ms:
            break
        params["startTime"] = last_time + 1
        if len(all_rows) % 4000 == 0:
            print(f"[INFO] Fetched {len(all_rows)} funding rows...")
        time.sleep(0.15)

    if not all_rows:
        print("[WARN] No funding data found.")
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df = pd.DataFrame(all_rows)
    df["fundingTime"] = pd.to_datetime(pd.to_numeric(df["fundingTime"]), unit="ms", utc=True)
    df.rename(columns={"fundingTime": "timestamp", "fundingRate": "funding_rate"}, inplace=True)
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    print(f"[OK] Funding data done: {len(df):,} rows")
    return df[["timestamp", "funding_rate"]]


# --------------- PREPROCESS ---------------
def preprocess_data(df_ohlcv: pd.DataFrame, df_funding: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Merge OHLCV with (optional) funding, basic QA."""
    print("[INFO] Preprocessing data...")

    df_ohlcv = df_ohlcv.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    if df_funding is not None and not df_funding.empty:
        df_funding = df_funding.drop_duplicates("timestamp").sort_values("timestamp")
        df = pd.merge_asof(df_ohlcv, df_funding, on="timestamp", direction="backward")
        df["funding_rate"] = df["funding_rate"].ffill()
    else:
        df = df_ohlcv.copy()
        df["funding_rate"] = np.nan

    df = df.dropna().reset_index(drop=True)

    print("[INFO] Time diff check:")
    print(df["timestamp"].diff().value_counts().head())

    df["symbol"] = symbol.replace("/", "_")
    print(f"[OK] Final dataset: {df.shape}")
    return df


# ------------------- CLI -------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch and preprocess Binance Spot/Futures data.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Pair, e.g. BTCUSDT")
    parser.add_argument("--market", type=str, choices=["spot", "futures"], default="futures",
                        help="Market type")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe")
    parser.add_argument("--outdir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    start_utc = pd.to_datetime(args.start, utc=True)
    end_utc = pd.to_datetime(args.end, utc=True)

    df_ohlcv = fetch_ohlcv(args.symbol, args.timeframe, start_utc, end_utc, args.market)
    df_funding = fetch_funding_rate(args.symbol, start_utc, end_utc) if args.market == "futures" else pd.DataFrame()

    df_prep = preprocess_data(df_ohlcv, df_funding, args.symbol)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{args.symbol.replace('/', '_')}_{args.timeframe}_{args.market}_features.parquet"
    df_prep.to_parquet(out_path, index=False)

    print(f"[OK] Saved -> {out_path}")
    print("[COLUMNS]", df_prep.columns.tolist())


if __name__ == "__main__":
    main()

