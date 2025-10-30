from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np

def split_scale(df, target_cols='y', test_size=0.2, val_size=0.1, scale=True):
    """Split into train/val/test and scale."""
    
    scaler = MinMaxScaler() if scale else None
    n = len(df)
    test_start = int(n * (1 - test_size - val_size))
    val_start = int(n * (1 - val_size))
    feature_cols = df.columns.difference(target_cols)

    if scale:
        scaler.fit(df[feature_cols].iloc[:test_start])
        df[feature_cols] = scaler.transform(df[feature_cols])

    return (
        df.iloc[:test_start].reset_index(drop=True),
        df.iloc[test_start:val_start].reset_index(drop=True),
        df.iloc[val_start:].reset_index(drop=True),
        scaler
    )


class CryptoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size=64, target='target'):

        self.window_size = window_size
        self.features = torch.as_tensor(
            df.drop(columns=target).to_numpy(), dtype=torch.float32)
        self.targets = torch.as_tensor(
            df[target].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        X = self.features[idx: idx + self.window_size]
        y = self.targets[idx]
        return X, y


def triple_barrier_label(df, close='close', high='high', low='low', volatility='atr_14', ku=1.5, kd=1.5, hold=16, labels=None, debug=False):
    """
    df: DataFrame with columns close/high/low/volatility (15m bars)
    ku, kd: upper/lower multipliers 
    hold: max holding period in bars
    returns: labels 
    """
    closes, highs, lows, volatility = df[close].values, df[high].values, df[low].values, df[volatility].values

    if labels is None:
        labels = {'higher': 1, 'lower': -1, 'none': 0}

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


def min_max_label(df, close='close', high='high', low='low', horizon=16):
    """
    df: DataFrame with columns close/high/low (15m bars)
    horizon: look-ahead horizon in bars
    returns: Max and Min values within horizon    
    """
    closes, highs, lows = df[close].values, df[high].values, df[low].values

    n = len(df)
    y_high = np.zeros(n, dtype=float)
    y_low = np.zeros(n, dtype=float)

    for i in range(n - horizon):
        y_high[i] = np.max(np.append(highs[i+1:i+1+horizon], closes[i]))
        y_low[i] = np.min(np.append(lows[i+1:i+1+horizon], closes[i]))

    out = df.copy()
    out['y_high'] = np.log(y_high/closes)
    out['y_low'] = -np.log(y_low/closes)

    return out.iloc[:n - horizon]
