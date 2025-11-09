#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import requests
import investpy


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def extract_asset_name(symbol: str):
    """
    Перетворює 'BTC/USDT' → 'btc', 'eth-usdt' → 'eth', 'SOLUSD' → 'sol'
    """
    symbol = symbol.upper().replace('-', '/')
    base = symbol.split('/')[0]
    return base.lower()


# ----------------------------------------------------------
# Data Loaders
# ----------------------------------------------------------

def load_investing_macro_news(start_date, end_date):
    df = investpy.economic_calendar(
        countries=["united states"],
        from_date=pd.to_datetime(start_date).strftime("%d/%m/%Y"),
        to_date=pd.to_datetime(end_date).strftime("%d/%m/%Y")
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    df["impact"] = df["importance"].str.lower().map({"low":1, "medium":2, "high":3}).fillna(1)

    for col in ["actual", "forecast"]:
        df[col] = (
            df[col].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .apply(lambda x: float(x) if x.replace(".", "", 1).isdigit() else np.nan)
        )

    df["surprise"] = (df["actual"] - df["forecast"]) / df["forecast"].replace(0, np.nan)
    df["event_score"] = df["surprise"].fillna(0) * df["impact"]

    daily = df.groupby("date")["event_score"].mean().reset_index()
    daily.rename(columns={"event_score": "macro_event_sentiment"}, inplace=True)

    daily["macro_event_intensity"] = daily["macro_event_sentiment"].rolling(5).mean()
    daily["macro_event_intensity_smooth"] = daily["macro_event_intensity"].ewm(span=5).mean()
    daily["macro_event_flag"] = (daily["macro_event_sentiment"] != 0).astype(int)

    return daily


def load_fear_greed(start_date, end_date):
    url = "https://api.alternative.me/fng/?limit=0"
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data["data"])
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df["date"] = df["date"].dt.floor("D") 
    df["fear_greed"] = df["value"].astype(float)
    
    df = df[(df["date"] >= pd.to_datetime(start_date))]

    df["fear_greed"] = df["fear_greed"].replace(0, np.nan)

    df["fear_greed"] = df["fear_greed"].ffill().bfill()

    return df[["date", "fear_greed"]]

def load_onchain_coinmetrics(start_date, end_date, symbol):
    asset = extract_asset_name(symbol)
    url = f"https://raw.githubusercontent.com/coinmetrics/data/master/csv/{asset}.csv"

    r = requests.get(url)
    if r.status_code != 200:
        print(f"[WARN] No on-chain data for {asset.upper()} — filling zeros.")
        return pd.DataFrame({"date": pd.date_range(start_date, end_date), "onchain_activity_index": 0})

    df = pd.read_csv(url)

    df = df.rename(columns={
        "time": "date",
        "AdrActCnt": "active_addresses",
        "TxCnt": "tx_count",
        "volume_reported_spot_usd_1d": "volume_usd",
        "CapMrktCurUSD": "market_cap",
        "CapMVRVCur": "mvrv_ratio"
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    features = ["active_addresses", "tx_count", "volume_usd", "market_cap"]
    df_norm = (df[features] - df[features].min()) / (df[features].max() - df[features].min())
    df["onchain_activity_index"] = df_norm.mean(axis=1)

    return df[["date", "onchain_activity_index"]]


# ----------------------------------------------------------
# Main Merge Logic
# ----------------------------------------------------------

def main(input_path, output_path, start_date, end_date, symbol):
    #print(f"[1/4] Loading dataset: {input_path}")
    base = pd.read_parquet(input_path)
    base["date"] = pd.to_datetime(base["date"])

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    #print("[2/4] Loading external datasets...")
    macro = load_investing_macro_news(start, end)
    fg = load_fear_greed(start, end)
    onchain = load_onchain_coinmetrics(start, end, symbol)

    #print("[3/4] Merging datasets...")
    # --- Align date formats (floor to day) ---
    base["date"] = pd.to_datetime(base["date"]).dt.floor("D")
    macro["date"] = pd.to_datetime(macro["date"]).dt.floor("D")
    fg["date"]   = pd.to_datetime(fg["date"]).dt.floor("D")
    onchain["date"] = pd.to_datetime(onchain["date"]).dt.floor("D")

    # --- Sort for merge_asof ---
    base = base.sort_values("date")
    macro = macro.sort_values("date")
    fg = fg.sort_values("date")
    onchain = onchain.sort_values("date")

    # --- Merge macro & sentiment ---
    ext = pd.merge_asof(base, macro, on="date", direction="backward")
    ext = pd.merge_asof(ext, fg, on="date", direction="backward")
    ext = pd.merge_asof(ext, onchain, on="date", direction="backward")

    # --- Final cleanup ---
    ext["fear_greed"] = ext["fear_greed"].replace(0, np.nan).ffill().bfill()
    ext["macro_event_sentiment"] = ext["macro_event_sentiment"].fillna(0)
    ext["macro_event_intensity"] = ext["macro_event_intensity"].fillna(0)
    ext["macro_event_intensity_smooth"] = ext["macro_event_intensity_smooth"].fillna(0)
    ext["macro_event_flag"] = ext["macro_event_flag"].fillna(0)
    ext["onchain_activity_index"] = ext["onchain_activity_index"].replace([np.inf, -np.inf], np.nan).ffill().bfill()

    merged = ext
    merged = merged.sort_values("timestamp")

    print(f"Saving macro features -> {output_path}")
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