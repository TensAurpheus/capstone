"""
build_pipeline.py
---------------------------------
Master pipeline runner for crypto data processing.

Executes all data preparation stages in sequence:
1. data_preprocess.py       – fetch OHLCV + funding data
2. preprocessing.py         – basic QA, cleanup
3. technical.py             – add technical indicators
4. patterns.py              – pattern-based feature extraction
5. normalize.py             – ATR-based normalization
6. standardize.py           – z-score scaling + save StandardScaler.pkl
7. data_postprocess.py      – categorical + cyclical encoding (final dataset)
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_script(script, args=None):
    """Run another Python script with subprocess and stream output live. 
    Returns True if success, False if non-zero exit code."""
    cmd = [sys.executable, script]
    if args:
        cmd += args
    print(f"\nRunning: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line, end="")

    process.wait()
    if process.returncode != 0:
        print(f"\n❌ ERROR: Script {script} failed with code {process.returncode}.")
        print("🛑 Pipeline stopped. Please fix the issue.")
        return False

    print(f"✅ {script} completed successfully.\n")
    return True


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

    print("\n=== FULL DATA PIPELINE STARTED ===")
    print(f"Symbol: {args.symbol} | Market: {args.market} | TF: {args.timeframe}")
    print(f"Period: {args.start} → {args.end}\n")

    # 1️ Step: Fetch + raw data
    if ask_step("Run data_preprocess.py (fetch OHLCV and funding data)"):
        if not run_script("src/data_pipeline/data/data_preprocess.py", [
            "--symbol", args.symbol,
            "--market", args.market,
            "--start", args.start,
            "--end", args.end,
            "--timeframe", args.timeframe
        ]):
            return

    # Step 2: Cleaning / QA
    raw_path = f"data/raw/{symbol_clean}_{args.timeframe}_{args.market}_raw.parquet"
    if ask_step("Run preprocessing.py (cleanup + feature prep)"):
        if not run_script("src/data_pipeline/data/preprocessing.py", [
            "--input", raw_path,
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_features.parquet"
        ]):
            return

    # 3️ Step: Technical indicators
    if ask_step("Run technical.py (technical indicators)"):
        if not run_script("src/data_pipeline/features/technical.py", [
            "--input", f"data/processed/{symbol_clean}_{args.timeframe}_features.parquet",
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_technical.parquet"
        ]):
            return

    # Step 4: Pattern-based features
    if ask_step("Run patterns.py (price-action patterns)"):
        if not run_script("src/data_pipeline/features/patterns.py", [
            "--input", f"data/processed/{symbol_clean}_{args.timeframe}_technical.parquet",
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_patterns.parquet"
        ]):
            return

    # Step 5: Normalization
    if ask_step("Run normalize.py (ATR normalization)"):
        if not run_script("src/data_pipeline/data/normalize.py", [
            "--input", f"data/processed/{symbol_clean}_{args.timeframe}_patterns.parquet",
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_normalized.parquet",
            "--symbol", args.symbol
        ]):
            return

    # Step 6: Standardization
    if ask_step("Run standardize.py (z-score scaling + save .pkl)"):
        if not run_script("src/data_pipeline/data/standardize.py", [
            "--input", f"data/processed/{symbol_clean}_{args.timeframe}_normalized.parquet",
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_standardized.parquet",
            "--symbol", args.symbol
        ]):
            return

    # Step 7: Final postprocessing (encoding + cleanup)
    if ask_step("Run data_postprocess.py (final feature cleanup & encoding)"):
        if not run_script("src/data_pipeline/data/data_postprocess.py", [
            "--symbol", args.symbol,
            "--market", args.market,
            "--timeframe", args.timeframe
        ]):
            return

    print("\n=== ✅ FULL PIPELINE COMPLETED SUCCESSFULLY ===")
    print("📦 Final datasets generated:")
    print(f" ├─ data/processed/{symbol_clean}_{args.timeframe}_normalized.parquet")
    print(f" ├─ data/processed/{symbol_clean}_{args.timeframe}_standardized.parquet")
    print(f" └─ data/processed/{symbol_clean}_{args.timeframe}_postprocessed.parquet")
    print("\n🧱 You can now train ML/DL models on these datasets.")


if __name__ == "__main__":
    main()



def ask_step(prompt):
    """Ask user if a pipeline step should be executed"""
    choice = input(f"\nStep: {prompt}? [Y/n]: ").strip().lower()
    return choice in ["", "y", "yes"]


def main():
    parser = argparse.ArgumentParser(description="Full crypto data preparation pipeline")
    parser.add_argument("--symbol", type=str, required=True, help="Trading pair, e.g. BTC/USDT")
    parser.add_argument("--market", type=str, choices=["spot", "futures"], default="futures")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--timeframe", type=str, default="15m")
    args = parser.parse_args()

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/model/scalers").mkdir(parents=True, exist_ok=True)

    symbol_clean = args.symbol.replace("/", "_")

    print("\n=== FULL DATA PIPELINE STARTED ===")
    print(f"Symbol: {args.symbol} | Market: {args.market} | TF: {args.timeframe}")
    print(f"Period: {args.start} → {args.end}\n")

    # Step 1: Fetch + Preprocess
    if ask_step("Run data_preprocess.py (fetch OHLCV + funding)"):
        if not run_script("src/data_pipeline/data/data_preprocess.py", [
            "--symbol", args.symbol,
            "--market", args.market,
            "--start", args.start,
            "--end", args.end,
            "--timeframe", args.timeframe
        ]):
            return

    # Step 2: Cleaning / QA
    raw_path = f"data/raw/{symbol_clean}_{args.timeframe}_{args.market}_raw.parquet"
    if ask_step("Run preprocessing.py (basic QA + feature prep)"):
        if not run_script("src/data_pipeline/data/preprocessing.py", [
            "--input", raw_path,
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_features.parquet"
        ]):
            return

    # Step 3: Technical indicators
    if ask_step("Run technical.py (technical indicators)"):
        if not run_script("src/data_pipeline/features/technical.py", [
            "--input", f"data/processed/{symbol_clean}_{args.timeframe}_features.parquet",
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_technical.parquet"
        ]):
            return

    # Step 4: Pattern-based features
    if ask_step("Run patterns.py (price-action patterns)"):
        if not run_script("src/data_pipeline/features/patterns.py", [
            "--input", f"data/processed/{symbol_clean}_{args.timeframe}_technical.parquet",
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_patterns.parquet"
        ]):
            return

    # Step 5: Normalization
    if ask_step("Run normalize.py (ATR normalization)"):
        if not run_script("src/data_pipeline/data/normalize.py", [
            "--input", f"data/processed/{symbol_clean}_{args.timeframe}_patterns.parquet",
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_normalized.parquet",
            "--symbol", args.symbol
        ]):
            return

    # Step 6: Standardization
    if ask_step("Run standardize.py (z-score scaling + save .pkl)"):
        if not run_script("src/data_pipeline/data/standardize.py", [
            "--input", f"data/processed/{symbol_clean}_{args.timeframe}_normalized.parquet",
            "--output", f"data/processed/{symbol_clean}_{args.timeframe}_standardized.parquet",
            "--symbol", args.symbol
        ]):
            return

    # Step 7: Final postprocessing (encoding + cleanup)
    if ask_step("Run data_postprocess.py (final feature cleanup & encoding)"):
        if not run_script("src/data_pipeline/data/data_postprocess.py", [
            "--symbol", args.symbol,
            "--market", args.market,
            "--timeframe", args.timeframe
        ]):
            return

    print("\n=== ✅ FULL PIPELINE COMPLETED SUCCESSFULLY ===")
    print(" Final datasets generated:")
    print(f" ├─ data/processed/{symbol_clean}_{args.timeframe}_normalized.parquet")
    print(f" └─ data/processed/{symbol_clean}_{args.timeframe}_standardized.parquet")

    print("\n🧱 You can now train ML/DL models on these datasets.")


if __name__ == "__main__":
    main()