# Crypto Data Pipeline

This repository provides a modular and extensible data pipeline for cryptocurrency time series analysis.
It automates data collection, cleaning, feature engineering, and transformation into a PyTorch-ready dataset for deep learning models (e.g., LSTM, Transformer, Attention-based architectures).

## Pipeline Overview

Each stage builds upon the previous one, progressively enriching the dataset.

## Data Fetching & Primary Preprocessing

File: src/data_pipeline/data/data_preprocess.py
- Fetches OHLCV data from Binance Spot or Futures via ccxt.
- Optionally fetches funding rate (for USDT-M futures).
- Merges OHLCV + funding data and performs basic cleaning.

Saves as:

data/raw/{symbol}_{timeframe}_{market}_features.parquet

## Data Cleaning & Validation

File: src/data_pipeline/data/preprocessing.py
- Ensures consistent timestamps (UTC).
- Removes duplicates, NaNs, and irregular time intervals.
- Validates numeric types and prepares for further processing.

Output:

data/processed/{symbol}_{timeframe}_features.parquet

## Technical Indicator Generation

File: src/data_pipeline/features/technical.py

Adds a rich set of technical analysis indicators via pandas_ta:

- Trend:	EMA (20, 50, 200), MACD, ADX
- Momentum:	RSI(14)
- Volatility:	Bollinger Bands, ATR(14)
- Volume:	MFI(14), Volume Z-score
- Price Structure:	VWAP, VWAP Distance, Rolling Std
- Returns:	Log Returns (15m, 1h, 4h, 1d)
- Session Info:	Trading sessions (Asia, London, NY), Session VWAP

Output:

data/processed/{symbol}_{timeframe}_technical.parquet

## Price Action & Market Structure Patterns

File: src/data_pipeline/features/patterns.py

Detects market structure and candlestick-based patterns using TA-Lib and custom logic:
- Candlestick: Engulfing, Harami, Hammer, Inverted Hammer
- Fractals & Levels: Detects local highs/lows and strong reversal levels
- Fair Value Gaps (FVG): Identifies price imbalances
- Premium / Discount Arrays (PDA): Defines price relative to equilibrium
- Breakouts: Detects bullish/bearish breakouts confirmed by volume

Output:

data/processed/{symbol}_{timeframe}_patterns.parquet

## PyTorch Dataset Creation

File: src/data_pipeline/data/data_postprocess_torch.py

Transforms processed data into PyTorch-ready datasets for training, validation, and testing.

Features:
- Converts categorical columns (session, pda) into one-hot encoding.
- Drops non-numeric columns and fills missing values.
- Splits data into Train/Validation/Test subsets.
- Scales features via MinMaxScaler.
- Creates windowed sequences for time-series learning.
- Generates .pt datasets and metadata in JSON.

Output directory:

data/model/{symbol}_{timeframe}_{stage}_{target}_{window}/

├── train_dataset.pt

├── val_dataset.pt

├── test_dataset.pt

├── scaler.pkl

└── dataset_meta.json


Example metadata (dataset_meta.json):

{"symbol": "BTC/USDT",

  "market": "futures",
  
  "timeframe": "15m",
    
  "stage": "patterns",
    
  "target": "close",
    
  "task_type": "regression",
    
  "window_size": 64,
    
  "num_features": 128,
    
  "train_len": 50000,
    
  "val_len": 10000,
    
  "test_len": 8000}

## Pipeline Orchestrator

File: run_pipeline.py

An interactive command-line tool that automates all the above stages with live logging and optional confirmation before each step.

Example:

python run_pipeline.py \
  --symbol BTC/USDT \
  --market futures \
  --start 2020-01-01 \
  --end 2025-01-01 \
  --timeframe 15m

## Dependencies
- ccxt -	Binance data fetching
- pandas, numpy	- Core data manipulation
- pandas_ta	- Technical indicators
- TA-Lib (optional)	- Candlestick pattern detection
- torch	- Dataset creation
- sklearn	- Data scaling
- pyarrow	- Parquet format support
