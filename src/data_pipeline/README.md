# Crypto Data Pipeline

This repository contains a full data-processing pipeline for building a
**model-ready dataset** for crypto forecasting.\
The pipeline fetches raw market data, performs multi-stage
preprocessing, adds technical, pattern-based, macroeconomic, behavioral,
and on-chain features, applies scale-free normalization, and outputs a
final numeric dataset ready for ML/DL models.

The system is modular --- each stage is a separate script --- and
`build_pipeline.py` orchestrates them.

------------------------------------------------------------------------

## Pipeline Overview

### 1. Raw Data Fetching (`data_preprocess.py`)

-   Downloads OHLCV from Binance Spot or Futures.
-   Fetches funding rates.
-   Merges OHLCV + funding.
-   Saves raw dataset.

### 2. Cleaning & QA (`preprocessing.py`)

-   Validates timestamps.
-   Removes duplicates and NaNs.
-   Ensures correct datatypes.

### 3. Technical Indicators (`technical.py`)

-   Adds EMA, RSI, ATR, MACD/PPO, Bollinger Bands.
-   VWAP and causal session statistics.
-   Fractal Dimension (Higuchi FD) + regime features.
-   Daily/weekly high-low levels.

### 4. Price--Action Patterns (`patterns.py`)

-   Candlestick patterns.
-   ICT market structure: pivots, BOS, CHoCH, MSS.
-   Fair Value Gaps.
-   PDA zones.
-   Breakouts.

### 5. Macro / Behavioral / On‑Chain (`macro_behavior_onchain.py`)

-   Macroeconomic events (Investing.com).
-   Fear & Greed Index.
-   On-chain metrics (CoinMetrics).
-   Strictly causal merging.

### 6. ATR‑Based Normalization (`normalize.py`)

-   ATR‑relative scaling.
-   Converts all price features to volatility‑normalized form.
-   PPO instead of MACD.
-   Bollinger to z‑scores.

### 7. Final Modeling Dataset (`data_postprocess.py`)

-   One‑hot encoding.
-   Cyclical timestamp features.
-   Convert everything to numeric.

### 8. Master Runner (`build_pipeline.py`)

Runs all steps in sequence.

------------------------------------------------------------------------

## Running the Pipeline

    python src/data_pipeline/build_pipeline.py     --symbol BTC/USDT     --market futures     --start 2018-01-01     --end 2025-11-01     --timeframe 1h

------------------------------------------------------------------------

## Output

The final dataset is:
- strictly numeric
- normalized
- leak-free
- chronologically sorted
- aligned across market, macro, sentiment, and on-chain signals
- ready for ML/DL models 

Final dataset:

    data/processed/BTC_USDT_1h_futures.parquet

Fully numeric, leak‑free, normalized, ML‑ready.

## Notes on Data Leakage Prevention

All feature engineering is explicitly causal:
- Funding, macro, F&G, and on-chain data are merged using backward merge_asof.
- Daily/weekly high/low use previous periods only.
- Fractal features use rolling windows with no forward looking.
- No backfilling (only forward fill allowed).
- Publish delays are respected (macro & on-chain release timestamps).