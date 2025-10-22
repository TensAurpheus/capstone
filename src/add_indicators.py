import pandas as pd
import numpy as np
import os
import pandas_ta as ta
from pathlib import Path

# directory containing feather files
data_dir = Path("data/binance/futures/data/")

# list of tickers
coins = ["ADA", "BNB", "BTC", "ETH", "SOL", "XRP"]

merged_list = []
def resave_data(data_dir, coins):
    for coin in coins:
        # read price and funding rate data
        price_file = data_dir / f"{coin}_USDT_USDT-15m-futures.feather"
        funding_file = data_dir / f"{coin}_USDT_USDT-8h-funding_rate.feather"

        df_price = pd.read_feather(price_file)
        df_funding = pd.read_feather(funding_file)
        

        # ensure datetime columns are consistent
        df_price["timestamp"] = pd.to_datetime(df_price["date"])
        df_funding["timestamp"] = pd.to_datetime(df_funding["date"])
        df_funding["funding_rate"] = df_funding["open"].astype("float64")
        print(df_funding.head())

        # merge funding rate into price data by nearest timestamp
        df_merged = pd.merge_asof(
            df_price.drop("date", axis=1).sort_values("timestamp"),
            df_funding[['timestamp', 'funding_rate']].sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        ).set_index("timestamp")
        # save as feather or parquet
        df_merged.to_parquet(data_dir / f"data{coin}_merged_15m_8h.parquet")
        print(df_merged.head())

def add_tech_indicators(g):
        
    g["ema20"] = ta.ema(g["close"], length=20)
    g["ema50"] = ta.ema(g["close"], length=50)
    g["ema200"] = ta.ema(g["close"], length=200)

    # Momentum
    g["rsi"] = ta.rsi(g["close"], length=14)
    macd = ta.macd(g["close"])
    g["macd"] = macd["MACD_12_26_9"]
    g["macd_signal"] = macd["MACDs_12_26_9"]

    # Volatility and trend strength
    bb = ta.bbands(g["close"], length=20)
    g["bb_width"] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]
    g["atr"] = ta.atr(g["high"], g["low"], g["close"], length=14)
    g["adx"] = ta.adx(g["high"], g["low"], g["close"], length=14)["ADX_14"]

    # ---- Volume-based ----
    g["mfi"] = ta.mfi(g["high"], g["low"], g["close"], g["volume"], length=14)
    g["z_volume"] = (g["volume"] - g["volume"].rolling(32).mean()
                     ) / g["volume"].rolling(32).std()
    
    # ---- VWAP and distance ----
    g["vwap"] = ta.vwap(g["high"], g["low"], g["close"], g["volume"])
    g["vwap_distance"] = (g["close"] - g["vwap"]) / g["vwap"]

    # ---- Rolling volatility ----
    g["roll_std_16"] = g["close"].rolling(16).std()
    g["roll_std_32"] = g["close"].rolling(32).std()

    # Returns
    g["return_15m"] = np.log(g["close"] / g["close"].shift(1))
    g["return_1h"] = np.log(g["close"] / g["close"].shift(4))
    g["return_4h"] = np.log(g["close"] / g["close"].shift(16))
    g["return_1d"] = np.log(g["close"] / g["close"].shift(96))

    return g

if __name__ == "__main__":
    df = pd.read_parquet(data_dir/os.listdir(data_dir)[0])
    print(df.head(100))
    # bb = ta.bbands(df["close"], length=20)
    # print(bb.columns)

    # for file in data_dir.iterdir():
    #     if file.suffix == ".parquet":
    #         df = pd.read_parquet(file)
    #         print(df.tail(20))
    #         print(df.columns)
    #         df.to_feather(file.with_suffix(".feather"))
            # df_ind = add_tech_indicators(df)
            # df_ind.to_parquet(file)





