# Adaptive Price Forecasting for Crypto Asset Trading: Tree-Based, Recurrent and Hybrid Models with Multimodal Features

This repository contains the research, models, and backtesting framework developed for the Master of Science in Computer Science capstone project (December 2025). The focus is on a deep learning and machine learning based hybrid trading strategy for BTC/USDT futures.

## Project Overview

### Motivation & Problem Statement
With the rapid growth and high volatility of cryptocurrencies, there is a critical need for robust forecasting and trading methods. A common issue in financial modeling is that models are typically evaluated purely on statistical metrics, lacking insight into their real-world utility. This project focuses on short-term price direction forecasting using **multimodal features** and a **unified protocol for evaluating financial performance** in realistic trading conditions, comparing traditional ML, DL, and novel Hybrid models.

### Goal
To create and validate an adaptive crypto price forecasting and trading system.

### Key Results & Conclusions
- **Statistical accuracy ≠ financial result**: Models must be evaluated via rigorous backtesting.
- **Hybrid Models** (DL-encoder → ML-classifier) demonstrated the best classification performance.
- **Pure Recurrent Models** (GRU, BiGRU) achieved the best return/risk ratio in trading.
- Combining multimodal features (market, on-chain, macro, behavioral) significantly improves results.
- Models trained on BTC showed successful cross-asset transferability to ETH.

## Multimodal Data & Target Variable
The models utilize a diverse, multimodal dataset spanning from Oct 2019 to Nov 2025:
- **Market data** (Binance)
- **Technical and structural indicators**
- **On-chain data** (CoinMetrics)
- **Macroeconomic events** (Investing)
- **Behavioral signals** (Fear & Greed Index)

**Target Labeling**: The project employs the **Triple Barrier Method** for labeling (upper profit-take, lower stop-loss, and time limit boundaries set dynamically via ATR).

## Project Structure

- `notebooks/`: Contains the main research and implementation notebooks.
  - `DL_hybrid_trade.ipynb`: The primary research notebook.
  - `ML_working.ipynb`: Implementations of ML models.
- `src/`: Core logic and utilities.
  - `data_pipeline/`: Data fetching, processing, and feature engineering (dedicated readme provided within).
  - `models/`: Implementations of Deep Learning architectures (LSTM, GRU, BiLSTM, BiGRU) and Hybrid models.
  - `utils/`: Data processing, Triple Barrier labeling, classification metrics, and the trading/backtesting engine.
- `requirements.txt`: Project dependencies for environment setup.

## Key Features

1. **Modular Architecture**: Reusable components for data piping, model training, and backtesting.
2. **Hybrid Models**: Combines Deep Learning sequence feature extraction with Gradient Boosting classification (CatBoost, XGBoost, LightGBM).
3. **Professional Backtesting**: Includes slippage, transaction costs, and detailed performance metrics (Sharpe, Martin, Calmar, MDD, Total Return).

## Getting Started

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `notebooks/DL_hybrid_trade.ipynb` to explore the strategy and view the complete research pipeline.
