import subprocess
import sys
import argparse
from pathlib import Path


def run_script(script, args=None):
    """Run another Python script with subprocess and stream output live"""
    cmd = [sys.executable, script]
    if args:
        cmd += args
    print(f"\nRunning: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()


def ask_step(prompt):
    """Ask user if a pipeline step should be executed"""
    choice = input(f"\nStep: {prompt}? [Y/n]: ").strip().lower()
    return choice in ["", "y", "yes"]


def main():
    parser = argparse.ArgumentParser(description="Build full crypto dataset pipeline")
    parser.add_argument("--symbol", type=str, required=True, help="Trading pair, e.g. BTC/USDT or SOL/USDT")
    parser.add_argument("--market", type=str, choices=["spot", "futures"], default="futures", help="Market type")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (UTC)")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date (UTC)")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe (e.g., 1m, 15m, 1h)")
    args = parser.parse_args()

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    symbol_clean = args.symbol.replace("/", "_")

    print("\n=== FULL PIPELINE STARTED ===")
    print(f"Symbol: {args.symbol} | Market: {args.market} | TF: {args.timeframe}")
    print(f"Period: {args.start} → {args.end}")

    # Step 1: Fetch + preprocess Binance data
    if ask_step("Run data_preprocess.py (fetch OHLCV and funding data)"):
        run_script("src/data_pipeline/data/data_preprocess.py", [
            "--symbol", args.symbol,
            "--market", args.market,
            "--start", args.start,
            "--end", args.end,
            "--timeframe", args.timeframe
        ])

    # Step 2: Pre-clean data
    if ask_step("Run preprocessing.py (data cleaning and QA)"):
        raw_path = f"data/raw/{symbol_clean}_{args.timeframe}_{args.market}_features.parquet"
        processed_features_path = f"data/processed/{symbol_clean}_{args.timeframe}_features.parquet"

        run_script("src/data_pipeline/data/preprocessing.py", [
            "--input", raw_path,
            "--output", processed_features_path
        ])

    # Step 3: Add technical indicators
    if ask_step("Run technical.py (technical indicators)"):
        run_script("src/data_pipeline/features/technical.py", [
            "--input", f"data/processed/{symbol_clean}_{args.timeframe}_features.parquet",
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_technical.parquet"
        ])

    # Step 4: Add pattern-based features
    if ask_step("Run patterns.py (price-action and structural patterns)"):
        run_script("src/data_pipeline/features/patterns.py", [
            "--input", f"data/processed/{symbol_clean}_{args.timeframe}_technical.parquet",
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_patterns.parquet"
        ])

    # Step 5: PyTorch-ready dataset
    if ask_step("Create PyTorch dataset (data_postprocess_torch.py)"):
        stage = input("Select data stage [raw/features/technical/patterns] (default=patterns): ").strip() or "patterns"
        window = input("Enter window size [64/128/256] (default=64): ").strip() or "64"
        target = input("Enter target column [close/log_return_15m] (default=close): ").strip() or "close"

        run_script("src/data_pipeline/data/data_postprocess_torch.py", [
            "--symbol", args.symbol,
            "--market", args.market,
            "--timeframe", args.timeframe,
            "--stage", stage,
            "--window", window,
            "--target", target
        ])

    print("\n=== FULL PIPELINE ENDED ===")
    print("Saved datasets:")
    print(f" ├─ data/raw/{symbol_clean}_{args.timeframe}_{args.market}_features.parquet")
    print(f" ├─ data/processed/{symbol_clean}_{args.timeframe}_features.parquet")
    print(f" ├─ data/processed/{symbol_clean}_{args.timeframe}_technical.parquet")
    print(f" └─ data/processed/{symbol_clean}_{args.timeframe}_patterns.parquet")


if __name__ == "__main__":
    main()



