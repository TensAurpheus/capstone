#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import requests
import investpy
from tzlocal import get_localzone
from dateutil import tz

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------


def extract_asset_name(symbol: str):
    """
    Перетворює 'BTC/USDT' → 'btc', 'eth-usdt' → 'eth', 'SOLUSD' → 'sol'
    """
    symbol = symbol.upper().replace("-", "/")
    base = symbol.split("/")[0]
    return base.lower()


# ----------------------------------------------------------
# Data Loaders
# ----------------------------------------------------------


def load_investing_macro_news(start_date, end_date):
    """
    Load macro news from Investing.com, convert to UTC, build causal sentiment features.
    """

    # 1) Raw calendar
    df = investpy.economic_calendar(
        countries=["united states"],
        from_date=pd.to_datetime(start_date).strftime("%d/%m/%Y"),
        to_date=pd.to_datetime(end_date).strftime("%d/%m/%Y"),
    )

    # 2) Parse date & time as naive
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df[df["time"] != "All Day"].copy()
    df["time"] = df["time"].fillna("00:00")

    naive_ts = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"], errors="coerce"
    )

    # 3) local time -> UTC
    local_tz = tz.tzlocal()
    df["release_ts"] = naive_ts.dt.tz_localize(
        local_tz, nonexistent="NaT", ambiguous="NaT"
    ).dt.tz_convert("UTC")

    df = df.dropna(subset=["release_ts"]).copy()

    # 4) Impact score (importance -> 1/2/3)
    df["impact"] = (
        df["importance"]
        .astype(str)
        .str.lower()
        .map({"low": 1, "medium": 2, "high": 3})
        .fillna(1)
    )

    # 5) actual / previous -> float
    def to_float(x):
        x = str(x).replace("%", "").replace(",", "").strip()
        if x.replace(".", "", 1).lstrip("-").isdigit():
            return float(x)
        return np.nan

    df["actual"] = df["actual"].apply(to_float)
    df["previous"] = df["previous"].apply(to_float)

    # 6) Causal surprise: actual vs previous
    df["surprise"] = np.where(
        df["previous"].notna() & df["actual"].notna(),
        (df["actual"] - df["previous"]) / df["previous"].abs().replace(0, np.nan),
        0.0,
    )

    # 7) Sentiment + flag
    df["macro_event_sentiment"] = df["surprise"] * df["impact"]
    df["macro_event_flag"] = 1

    # 8) Select columns
    macro = df[["release_ts", "macro_event_sentiment", "macro_event_flag"]].copy()

    macro = macro.dropna(subset=["release_ts"]).sort_values("release_ts")
    macro = macro.set_index("release_ts")
    macro = macro[~macro.index.isna()]
    macro = macro.sort_index()

    macro["macro_event_intensity"] = (
        macro["macro_event_sentiment"].rolling("5D", min_periods=1).mean()
    )

    macro["macro_event_intensity_smooth"] = (
        macro["macro_event_intensity"].ewm(span=5, adjust=False).mean()
    )

    # 10) Reset index
    macro = macro.reset_index()

    return macro


def load_fear_greed(start_date, end_date):
    url = "https://api.alternative.me/fng/?limit=0"
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data["data"])

    # Convert UNIX timestamp to datetime at midnight UTC
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.floor("D")

    # Convert value to numeric
    df["fear_greed"] = df["value"].astype(float)

    # Ensure data is within requested range
    df = df[df["date"] >= pd.to_datetime(start_date)]

    # No backward fill allowed (avoids leakage)
    df["fear_greed"] = df["fear_greed"].replace(0, np.nan).ffill()

    # Assign actual publish time (next day)
    df["publish_ts"] = df["date"] + pd.Timedelta(days=1)

    # Keep only required columns
    return df[["publish_ts", "fear_greed"]].sort_values("publish_ts")


def load_onchain_coinmetrics(start_date, end_date, symbol):
    """
    Loads on-chain metrics and computes a causal rolling Z-score index.
    Returns only one feature: onchain_index.
    Ensures no future leakage (1-day publish lag + backward rolling window).
    """

    asset = extract_asset_name(symbol)
    url = f"https://raw.githubusercontent.com/coinmetrics/data/master/csv/{asset}.csv"

    r = requests.get(url)
    if r.status_code != 200:
        print(f"[WARN] No on-chain data for {asset.upper()} — fallback zeros.")
        dates = pd.date_range(start_date, end_date)
        return pd.DataFrame({"publish_ts": dates, "onchain_index": 0.0})

    df = pd.read_csv(url)

    df = df.rename(
        columns={
            "time": "date",
            "AdrActCnt": "active_addresses",
            "TxCnt": "tx_count",
            "volume_reported_spot_usd_1d": "volume_usd",
            "CapMrktCurUSD": "market_cap",
        }
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    # Publish delay → ensures no future data is available on the same day
    df["publish_ts"] = df["date"] + pd.Timedelta(days=1)

    features = ["active_addresses", "tx_count", "volume_usd", "market_cap"]
    df = df.set_index("date").sort_index()

    # Compute Z-score for each metric (rolling causal window)
    for col in features:
        roll_mean = df[col].rolling("90D", min_periods=5).mean()
        roll_std = df[col].rolling("90D", min_periods=5).std()
        df[f"{col}_z"] = (df[col] - roll_mean) / roll_std

    # Combine into a single index
    z_cols = [f"{c}_z" for c in features]
    df["onchain_activity_index"] = df[z_cols].mean(axis=1)
    df = df.reset_index()

    return df[["publish_ts", "onchain_activity_index"]].sort_values("publish_ts")


# ----------------------------------------------------------
# Main Merge Logic
# ----------------------------------------------------------


def main(input_path, output_path, start_date, end_date, symbol):

    # -----------------------------
    # 1. Load base dataset (intraday OHLCV)
    # -----------------------------
    base = pd.read_parquet(input_path)

    # Ensure timestamp is datetime without shifting real time
    base["timestamp"] = pd.to_datetime(base["timestamp"], errors="coerce")

    # Make it timezone-aware (UTC) WITHOUT modifying the hour
    if base["timestamp"].dt.tz is None:
        base["timestamp"] = base["timestamp"].dt.tz_localize("UTC")

    # Extract date column at daily granularity (kept naive)
    base["date"] = base["timestamp"].dt.date
    base = base.sort_values("timestamp")

    # -----------------------------
    # 2. Load external datasets
    # -----------------------------
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    macro = load_investing_macro_news(start, end)  # has release_ts
    fg = load_fear_greed(start, end)  # has publish_ts
    onchain = load_onchain_coinmetrics(start, end, symbol)  # has publish_ts

    # -----------------------------
    # 3. Normalize timestamps of external datasets
    # -----------------------------
    # ONLY convert to datetime
    macro["release_ts"] = pd.to_datetime(macro["release_ts"], errors="coerce")

    # Fear & Greed: UNIX timestamps → always UTC
    fg["publish_ts"] = pd.to_datetime(fg["publish_ts"], utc=True)

    # On-chain: CoinMetrics → also UTC
    onchain["publish_ts"] = pd.to_datetime(onchain["publish_ts"], utc=True)

    # Sort for merge_asof
    macro = macro.sort_values("release_ts")
    fg = fg.sort_values("publish_ts")
    onchain = onchain.sort_values("publish_ts")

    # -----------------------------
    # 4. Merge macro by event release time
    # -----------------------------
    merged = pd.merge_asof(
        base, macro, left_on="timestamp", right_on="release_ts", direction="backward"
    )

    # -----------------------------
    # 5. Merge Fear & Greed
    # -----------------------------
    merged = pd.merge_asof(
        merged, fg, left_on="timestamp", right_on="publish_ts", direction="backward"
    )

    # -----------------------------
    # 6. Merge On-chain metrics
    # -----------------------------
    merged = pd.merge_asof(
        merged,
        onchain,
        left_on="timestamp",
        right_on="publish_ts",
        direction="backward",
    )

    # -----------------------------
    # 7. Final cleanup (STRICT no-leak)
    # -----------------------------

    # Only forward fill allowed (no bfill!)
    merged["fear_greed"] = merged["fear_greed"].ffill()

    # Macro missing = 0
    for col in [
        "macro_event_sentiment",
        "macro_event_intensity",
        "macro_event_intensity_smooth",
        "macro_event_flag",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # On-chain missing: forward fill only
    if "onchain_activity_index" in merged.columns:
        merged["onchain_activity_index"] = (
            merged["onchain_activity_index"].replace([np.inf, -np.inf], np.nan).ffill()
        )

    merged = merged.sort_values("timestamp")

    # -----------------------------
    # 8. Save
    # -----------------------------
    print(f"[OK] Saving merged macro/F&G/onchain -> {output_path}")
    merged.to_parquet(output_path, index=False)


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()

    main(args.input, args.output, args.start, args.end, args.symbol)
