import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch


# ============================================================
# === UNIVERSAL SCALER CLASS ===
# ============================================================

class DataScaler:
    """
    Wrapper class for StandardScaler that supports fit/save/load/transform
    across different stages (pipeline, training, inference).
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.scaler = None

    def fit(self, df: pd.DataFrame, cols: List[str]):
        """Fit StandardScaler on specified columns and optionally save."""
        self.scaler = StandardScaler()
        self.scaler.fit(df[cols])

        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            joblib.dump(self.scaler, self.path)
            print(f"[OK] StandardScaler fitted and saved to {self.path}")
        else:
            print("[INFO] StandardScaler fitted (not saved).")
        return self

    def load(self):
        """Load pre-fitted scaler from file."""
        if not self.path or not os.path.exists(self.path):
            raise FileNotFoundError(f"[ERROR] Scaler file not found: {self.path}")
        self.scaler = joblib.load(self.path)
        print(f"[INFO] Loaded StandardScaler from {self.path}")
        return self

    def transform(self, df: pd.DataFrame, cols: List[str]):
        """Apply transformation using fitted or loaded scaler."""
        if self.scaler is None:
            raise RuntimeError("[ERROR] Scaler not fitted or loaded before transform().")
        df[cols] = self.scaler.transform(df[cols])
        return df

    def fit_transform(self, df: pd.DataFrame, cols: List[str]):
        """Fit then transform in one call."""
        self.fit(df, cols)
        return self.transform(df, cols)


# ============================================================
# === SPLIT + SCALE FUNCTION ===
# ============================================================

def split_scale(
    df: pd.DataFrame,
    target_cols: str = 'y',
    test_size: float = 0.2,
    val_size: float = 0.1,
    scale: bool = True,
    continuous_cols: Optional[List[str]] = None,
    scaler_path: Optional[str] = None
):
    """
    Split DataFrame into train/val/test and apply DataScaler if requested.

    Args:
        df : full DataFrame
        target_cols : target column name
        test_size : proportion for test set
        val_size : proportion for validation set
        scale : whether to scale continuous columns
        continuous_cols : columns to scale (auto-detected if None)
        scaler_path : path for StandardScaler file

    Returns:
        train_df, val_df, test_df, fitted DataScaler (or None)
    """
    n = len(df)
    test_start = int(n * (1 - test_size - val_size))
    val_start = int(n * (1 - val_size))

    if continuous_cols is None:
        numeric_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
        binary_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) <= 2]
        continuous_cols = [c for c in numeric_cols if c not in binary_cols]

    scaler = None
    if scale:
        scaler = DataScaler(scaler_path)
        if scaler_path and os.path.exists(scaler_path):
            scaler.load()
        else:
            scaler.fit(df.iloc[:test_start], continuous_cols)
        df = scaler.transform(df, continuous_cols)

    return (
        df.iloc[:test_start].reset_index(drop=True),
        df.iloc[test_start:val_start].reset_index(drop=True),
        df.iloc[val_start:].reset_index(drop=True),
        scaler
    )


# ============================================================
# === TORCH DATASET CLASS ===
# ============================================================

class CryptoDataset(Dataset):
    """
    PyTorch Dataset for windowed crypto data.
    Converts tabular features into sequential windows.
    """
    def __init__(self, df: pd.DataFrame, window_size=64, target='target'):
        self.window_size = window_size
        self.features = torch.as_tensor(df.drop(columns=target).to_numpy(), dtype=torch.float32)
        self.targets = torch.as_tensor(df[target].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx):
        X = self.features[idx: idx + self.window_size]
        y = self.targets[idx]
        return X, y


# ============================================================
# === LABELING UTILITIES ===
# ============================================================

def triple_barrier_label(df, close='close', high='high', low='low',
                         volatility='atr_14', ku=1.5, kd=1.5,
                         hold=16, labels=None, debug=False):
    """
    Triple barrier labeling method (based on López de Prado).
    Detects direction (higher/lower/neutral) within hold-period window.
    """
    closes, highs, lows, vol = df[close].values, df[high].values, df[low].values, df[volatility].values
    if labels is None:
        labels = {'higher': 1, 'lower': -1, 'none': 0}

    n = len(df)
    y = np.zeros(n, dtype=int)
    b_up = closes + ku * vol
    b_dn = closes - kd * vol

    for i in range(n - hold):
        hit = False
        for j in range(1, hold + 1):
            if highs[i + j] >= b_up[i]:
                y[i] = labels['higher']
                hit = True
                break
            elif lows[i + j] <= b_dn[i]:
                y[i] = labels['lower']
                hit = True
                break
        if not hit:
            y[i] = labels['none']

    out = df.copy()
    out['y'] = y
    if debug:
        out['b_up'] = b_up
        out['b_dn'] = b_dn

    return out


def min_max_label(df, close='close', high='high', low='low', horizon=16):
    """
    Compute future maximum/minimum log returns over a lookahead horizon.
    Useful for regression-based targets or probabilistic labeling.
    """
    closes, highs, lows = df[close].values, df[high].values, df[low].values
    n = len(df)
    y_high, y_low = np.zeros(n), np.zeros(n)

    for i in range(n - horizon):
        y_high[i] = np.max(highs[i + 1:i + 1 + horizon])
        y_low[i] = np.min(lows[i + 1:i + 1 + horizon])

    out = df.copy()
    out['y_high'] = np.log(y_high / closes)
    out['y_low'] = np.log(y_low / closes)
    return out
