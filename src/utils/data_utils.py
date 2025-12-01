# data utils
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np

#data utils

def split_scale(df, target_cols='y', test_size=0.09, val_size=0.09, scale=True, for_test=False, volatility='atr_14'):
    """Split into train/val/test (or train+val/test for for_test) and selectively scale features."""
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if isinstance(target_cols, str):
        target_cols = [target_cols]

    n = len(df)
    test_start = int(n * (1 - test_size))
    val_start = int(n * (1 - val_size - test_size))

    feature_cols = [c for c in df.columns if c not in target_cols]

    exclude_cols = [
        'open', 'high', 'low', 'close', 'hour_sin', 'hour_cos',
        'dayofweek_sin', 'dayofweek_cos', 'z_volume', 'bb_percent'
    ] + [volatility]
    print(exclude_cols)
    binary_cols = [c for c in feature_cols if df[c].nunique(dropna=True) <= 2]
    scale_cols = [c for c in feature_cols if c not in exclude_cols + binary_cols]

    print(f"[INFO] Using StandardScaler, scaling = {scale}")
    print(f"[INFO] Total features: {len(feature_cols)} | Scaled: {len(scale_cols)} | Excluded: {len(exclude_cols)}")

    scaler = StandardScaler() if scale else None

    if for_test:
        # Train = train+val; fit on train+val
        if scale and len(scale_cols) > 0:
            scaler.fit(df[scale_cols].iloc[:test_start])
            df[scale_cols] = scaler.transform(df[scale_cols])
            print(f"[OK] Standard scaling applied to {len(scale_cols)} columns (fit on train+val).")
        else:
            print("[INFO] Scaling skipped (using raw features).")

        train_df = df.iloc[:test_start].reset_index(drop=True)   # train+val
        test_df  = df.iloc[test_start:].reset_index(drop=True)
        print(f"[OK] Split complete → Train (train+val): {len(train_df)}, Test: {len(test_df)}")
        return train_df, test_df, scaler

    # Regular 3-way split; fit on train only
    if scale and len(scale_cols) > 0:
        scaler.fit(df[scale_cols].iloc[:val_start])  # train only
        df[scale_cols] = scaler.transform(df[scale_cols])
        print(f"[OK] Standard scaling applied to {len(scale_cols)} columns (fit on train only).")
    else:
        print("[INFO] Scaling skipped (using raw features).")

    train_df = df.iloc[:val_start].reset_index(drop=True)
    val_df   = df.iloc[val_start:test_start].reset_index(drop=True)
    test_df  = df.iloc[test_start:].reset_index(drop=True)

    print(f"[OK] Split complete → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df, scaler



class CryptoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size=64, target='target'):

        self.window_size = window_size
        self.features = torch.as_tensor(
            df.drop(columns=target).to_numpy(), dtype=torch.float32)
        self.targets = torch.as_tensor(
            df[target].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx):
        X = self.features[idx: idx + self.window_size]
        y = self.targets[idx + self.window_size - 1]
        return X, y


def triple_barrier_label(df, close='close', high='high', low='low', volatility='atr_200', ku=6.0, kd=2.0, hold=336, labels=None, debug=False):
    """
    df: DataFrame with columns close/high/low/volatility (15m bars)
    ku, kd: upper/lower multipliers 
    hold: max holding period in bars
    returns: labels 
    """
    closes, highs, lows, volatility = df[close].values, df[high].values, df[low].values, df[volatility].values

    if labels is None:
        labels = {'higher': 2, 'lower': 1, 'none': 0}

    n = len(df)
    y = np.zeros(n, dtype=int)
    b_up = closes + ku * volatility
    b_dn = closes - kd * volatility

    hit = False

    for i in range(n - hold):
        j = 1
        # barriers in price space, adapted by side
        while not hit and (j <= hold):
            if highs[i+j] >= b_up[i]:
                hit = True
                y[i] = labels['higher']
            elif lows[i+j] <= b_dn[i]:
                hit = True
                y[i] = labels['lower']
            j += 1

        if not hit:
            # expiry label
            y[i] = labels['none']
        hit = False

    out = df.copy()
    out['y'] = y
    if debug:
        out['b_up'] = b_up
        out['b_dn'] = b_dn

    return out.iloc[:n - hold]