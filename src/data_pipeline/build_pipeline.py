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

    # ensure dirs
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/model/scalers").mkdir(parents=True, exist_ok=True)

    symbol_clean = args.symbol.replace("/", "_")

    print("\n=== FULL PIPELINE STARTED ===")
    print(f"Symbol: {args.symbol} | Market: {args.market} | TF: {args.timeframe}")
    print(f"Period: {args.start} → {args.end}")

    # 1️ Step: Fetch + raw data
    if ask_step("Run data_preprocess.py (fetch OHLCV and funding data)"):
        run_script("src/data_pipeline/data/data_preprocess.py", [
            "--symbol", args.symbol,
            "--market", args.market,
            "--start", args.start,
            "--end", args.end,
            "--timeframe", args.timeframe
        ])

    # 2️ Step: Minimal preprocessing / QA
    if ask_step("Run preprocessing.py (clean + basic QA)"):
        raw_path = f"data/raw/{symbol_clean}_{args.timeframe}_{args.market}_raw.parquet"
        cleaned_path = f"data/processed/{symbol_clean}_{args.timeframe}_features.parquet"
        run_script("src/data_pipeline/data/preprocessing.py", [
            "--input", raw_path,
            "--output", cleaned_path
        ])

    # 3️ Step: Technical indicators
    if ask_step("Run technical.py (technical indicators)"):
        run_script("src/data_pipeline/features/technical.py", [
            "--input", f"data/processed/{symbol_clean}_{args.timeframe}_features.parquet",
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_technical.parquet"
        ])

    # 4️ Step: Pattern-based features
    if ask_step("Run patterns.py (price-action & structural patterns)"):
        run_script("src/data_pipeline/features/patterns.py", [
            "--input", f"data/processed/{symbol_clean}_{args.timeframe}_technical.parquet",
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_patterns.parquet"
        ])

    # 5️ Step: Feature normalization (+ scaling option)
    if ask_step("Run normalized.py (normalize features + optional StandardScaler)"):
        run_script("src/data_pipeline/data/normalized.py", [
            "--input", f"data/processed/{symbol_clean}_{args.timeframe}_patterns.parquet",
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_normalized.parquet",
            "--symbol", args.symbol
        ])

    # 6️ Step: Final postprocessing (e.g., labeling)
    if ask_step("Run data_postprocess.py (final labeling and export)"):
        run_script("src/data_pipeline/data/data_postprocess.py", [
            "--symbol", args.symbol,
            "--market", args.market,
            "--timeframe", args.timeframe
        ])

    # ✅ Summary
    print("\n=== FULL PIPELINE COMPLETED ===")
    print("✅ Final saved datasets:")
    print(f" ├─ data/raw/{symbol_clean}_{args.timeframe}_{args.market}_raw.parquet")
    print(f" └─ data/processed/{symbol_clean}_{args.timeframe}_postprocessed.parquet")
    print("🧱 All intermediate files are ready for model training.")


if __name__ == "__main__":
    main()