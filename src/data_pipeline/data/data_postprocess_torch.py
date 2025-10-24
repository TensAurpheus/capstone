"""
data_postprocess_torch.py
-------------------------
Converts processed crypto datasets into PyTorch-ready format,
including scaling, windowing, and saving metadata per experiment.
"""

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
import sys


# === PyTorch Dataset ===
class CryptoDataset(Dataset):
    def __init__(self, data, window_size=64):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.features = self.data[:, :-1]
        self.targets = self.data[:, -1]

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        X = self.features[idx: idx + self.window_size]
        y = self.targets[idx + self.window_size - 1]
        return X, y


# === Helper Functions ===
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert non-numeric columns and prepare for scaling."""
    df = df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp')
        df = df.set_index('timestamp')

    for col in ['session', 'pda']:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col)
    for col in ['fvg_present', 'pattern_active']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    if 'symbol' in df.columns and df['symbol'].nunique() == 1:
        df = df.drop(columns=['symbol'])

    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        print(f"[WARN] Dropping non-numeric columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    return df.fillna(0).astype(float)


def split_scale(df, feature_cols, target_col='close', test_size=0.2, val_size=0.1, scale=True):
    """Split into train/val/test and scale."""
    df = df.copy()
    df['target'] = np.where(df[target_col].shift(-1) > df[target_col], 1, -1)
    df = df.dropna().reset_index(drop=True)

    scaler = MinMaxScaler() if scale else None
    n = len(df)
    test_start = int(n * (1 - test_size - val_size))
    val_start = int(n * (1 - val_size))

    if scale:
        scaler.fit(df.loc[:test_start - 1, feature_cols])
        df[feature_cols] = scaler.transform(df[feature_cols])

    return (
        df.iloc[:test_start].reset_index(drop=True),
        df.iloc[test_start:val_start].reset_index(drop=True),
        df.iloc[val_start:].reset_index(drop=True),
        scaler
    )


# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--market", default="futures")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--stage", default="patterns", choices=["raw", "features", "technical", "patterns"])
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--target", default="close", help="Target column: close or log_return_15m")
    args = parser.parse_args()

    symbol_clean = args.symbol.replace("/", "_")
    input_path = Path(f"data/processed/{symbol_clean}_{args.timeframe}_{args.stage}.parquet")

    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)

    # ✅ Unique model directory (includes target + window)
    output_dir = Path(f"data/model/{symbol_clean}_{args.timeframe}_{args.stage}_{args.target}_{args.window}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Model output directory: {output_dir}")

    df = pd.read_parquet(input_path)
    df = prepare_dataframe(df)

    feature_cols = [c for c in df.columns if c != "timestamp"]
    train_df, val_df, test_df, scaler = split_scale(df, feature_cols, target_col=args.target)

    train_ds = CryptoDataset(train_df, args.window)
    val_ds = CryptoDataset(val_df, args.window)
    test_ds = CryptoDataset(test_df, args.window)

    torch.save(train_ds, output_dir / "train_dataset.pt")
    torch.save(val_ds, output_dir / "val_dataset.pt")
    torch.save(test_ds, output_dir / "test_dataset.pt")
    print(f"[OK] Saved datasets in {output_dir}")

    if scaler:
        joblib.dump(scaler, output_dir / "scaler.pkl")

    # ✅ Added feature list to metadata
    meta = {
        "symbol": args.symbol,
        "market": args.market,
        "timeframe": args.timeframe,
        "stage": args.stage,
        "target": args.target,
        "task_type": "classification" if args.target == "log_return_15m" else "regression",
        "window_size": args.window,
        "num_features": len(feature_cols),
        "feature_cols": feature_cols,   # 👈 added feature list here
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "test_len": len(test_ds),
    }

    with open(output_dir / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(f"[OK] Saved metadata to {output_dir / 'dataset_meta.json'}")
    print(f"Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print("=== PYTORCH DATASET CREATION ENDED ===")


if __name__ == "__main__":
    main()

