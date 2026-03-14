# data utils
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np

FEATURE_BLOCKS = {
    "core_trend_mom": [
        "volume",
        "funding_rate",
        "ema_20",
        "ema_50",
        "ema_200",
        "adx",
        "rsi_14",
        "bb_percent",
        "bb_width",
        "log_return_1h",
        "log_return_4h",
        "log_return_1d",
        "roll_std_4h",
        "roll_std_8h",
        "atr_vol_regime",
        "ppo",
        "ppo_signal",
        "ppo_hist",
        "bb_z",
        "bb_percB",
    ],
    "volume_vwap_session": [
        "mfi_14",
        "z_volume",
        "vwap",
        "vwap_distance",
        "vwap_session",
        "session_vol_mean",
        "session_return_mean",
        "session_volatility",
    ],
    "fractal_regime": [
        "fd_24",
        "fd_slope",
        "fd_ema_12",
        "fd_ema_24",
        "fd_trend_strength",
        "fd_threshold_causal",
        "fd_regime",
        "fd_regime_switch",
        "fd_volatility",
        "fd_vol_ratio",
        "fd_vol_slope",
        "fd_slope_atr_norm",
        "fd_entropy",
        "fd_vol_adjusted",
        "fd_24_robust_z",
        "fd_ema_12_robust_z",
        "fd_ema_24_robust_z",
        "fd_trend_strength_robust_z",
        "fd_slope_robust_z",
        "fd_slope_atr_norm_robust_z",
        "fd_volatility_robust_z",
        "fd_vol_ratio_robust_z",
        "fd_vol_slope_robust_z",
        "fd_entropy_robust_z",
        "fd_vol_adjusted_robust_z",
    ],
    "patterns_ict": [
        "pattern_bullish_engulf",
        "pattern_bearish_engulf",
        "pattern_harami",
        "pattern_hammer",
        "pattern_inverted_hammer",
        "swing_high",
        "swing_low",
        "last_swing_high",
        "last_swing_low",
        "bos_bullish",
        "bos_bearish",
        "choch_bullish",
        "choch_bearish",
        "mss_bullish",
        "mss_bearish",
        "bullish_fvg",
        "bearish_fvg",
        "fvg_gap",
        "rolling_high",
        "rolling_low",
        "equilibrium",
        "breakout_bullish",
        "breakout_bearish",
        "pattern_count",
        "pattern_active",
    ],
    "macro_onchain": [
        "macro_event_sentiment",
        "macro_event_flag",
        "macro_event_intensity",
        "macro_event_intensity_smooth",
        "fear_greed",
        "onchain_activity_index",
        "funding_bias",
    ],
    "atr_levels": [
        "atr_pct",
        "range_atr",
        "body_atr",
        "dist_daily_high",
        "dist_daily_low",
        "dist_weekly_high",
        "dist_weekly_low",
        "atr_vol_regime_z",
    ],
    "time_pda": [
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "session_Asia",
        "session_Frankfurt",
        "session_London",
        "session_NewYork",
        "session_OffHours",
        "pda_Discount",
        "pda_Premium",
    ],
}


def split_scale(
    df,
    target_cols="y",
    test_size=0.09,
    val_size=0.09,
    scale=True,
    for_test=False,
    volatility="atr_14",
):
    """Split into train/val/test (or train+val/test for for_test) and selectively scale features."""
    df = df.copy().replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if isinstance(target_cols, str):
        target_cols = [target_cols]

    n = len(df)
    test_start = int(n * (1 - test_size))
    val_start = int(n * (1 - val_size - test_size))

    feature_cols = [c for c in df.columns if c not in target_cols]

    exclude_cols = [
        "open",
        "high",
        "low",
        "close",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "z_volume",
        "bb_percent",
    ] + [volatility]

    binary_cols = [c for c in feature_cols if df[c].nunique(dropna=True) <= 2]
    scale_cols = [c for c in feature_cols if c not in exclude_cols + binary_cols]

    print(f"[INFO] Using StandardScaler, scaling = {scale}")
    print(
        f"[INFO] Total features: {len(feature_cols)} | Scaled: {len(scale_cols)} | Excluded: {len(exclude_cols)}"
    )

    scaler = StandardScaler() if scale else None

    if for_test:
        if scale and len(scale_cols) > 0:
            scaler.fit(df[scale_cols].iloc[:test_start])
            df[scale_cols] = scaler.transform(df[scale_cols])
        train_df = df.iloc[:test_start].reset_index(drop=True)
        test_df = df.iloc[test_start:].reset_index(drop=True)
        return train_df, test_df, scaler

    if scale and len(scale_cols) > 0:
        scaler.fit(df[scale_cols].iloc[:val_start])
        df[scale_cols] = scaler.transform(df[scale_cols])

    train_df = df.iloc[:val_start].reset_index(drop=True)
    val_df = df.iloc[val_start:test_start].reset_index(drop=True)
    test_df = df.iloc[test_start:].reset_index(drop=True)

    return train_df, val_df, test_df, scaler


class CryptoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size=64, target="target"):
        self.window_size = window_size
        self.features = torch.as_tensor(
            df.drop(columns=target).to_numpy(), dtype=torch.float32
        )
        self.targets = torch.as_tensor(df[target].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx):
        X = self.features[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size - 1]
        return X, y


def triple_barrier_label(
    df,
    close="close",
    high="high",
    low="low",
    volatility="atr_200",
    ku=6.0,
    kd=2.0,
    hold=336,
    labels=None,
    debug=False,
):
    """Triple-barrier labeling."""
    closes, highs, lows, vol = (
        df[close].values,
        df[high].values,
        df[low].values,
        df[volatility].values,
    )
    if labels is None:
        labels = {"higher": 2, "lower": 1, "none": 0}
    n = len(df)
    y = np.zeros(n, dtype=int)
    b_up, b_dn = closes + ku * vol, closes - kd * vol
    for i in range(n - hold):
        hit, j = False, 1
        while not hit and (j <= hold):
            if highs[i + j] >= b_up[i]:
                hit = True
                y[i] = labels["higher"]
            elif lows[i + j] <= b_dn[i]:
                hit = True
                y[i] = labels["lower"]
            j += 1
        if not hit:
            y[i] = labels["none"]
    out = df.copy()
    out["y"] = y
    if debug:
        out["b_up"], out["b_dn"] = b_up, b_dn
    return out.iloc[: n - hold]


def min_max_label(df, close="close", high="high", low="low", horizon=16):
    """Look-ahead max/min labels."""
    closes, highs, lows = df[close].values, df[high].values, df[low].values
    n = len(df)
    y_high, y_low = np.zeros(n, dtype=float), np.zeros(n, dtype=float)
    for i in range(n - horizon):
        y_high[i] = np.max(np.append(highs[i + 1 : i + horizon + 1], closes[i]))
        y_low[i] = np.min(np.append(lows[i + 1 : i + horizon + 1], closes[i]))
    out = df.copy()
    out["y_high"] = np.log(y_high / closes)
    out["y_low"] = -np.log(y_low / closes)
    return out.iloc[: n - horizon]


def build_feature_df(
    df: pd.DataFrame,
    blocks: List[str],
    target_col: str = "y",
    volatility_col: str = "atr_200",
) -> pd.DataFrame:
    """Subset df to specific blocks of features."""
    cols = set()
    for b in blocks:
        if b in FEATURE_BLOCKS:
            cols.update(FEATURE_BLOCKS[b])
    always = {"open", "high", "low", "close", target_col, volatility_col}
    all_needed = (cols | always) & set(df.columns)
    return df[list(all_needed)].copy()


def fetch_data(tf: str, input_dir: str = "/kaggle/input") -> pd.DataFrame:
    """Fetch BTC data for a given timeframe."""
    path = Path(input_dir) / f"btc-new{tf}" / f"BTC_USDT_{tf}_futures.parquet"
    if not path.exists():
        path = Path(input_dir) / f"btc-{tf}" / f"BTC_USDT_{tf}_futures.parquet"
    df = pd.read_parquet(path)
    if "atr_14" in df.columns and "atr_200" in df.columns:
        df = df.drop(columns=["atr_14"])
    return df


def data_pipe(
    df, ku, kd, hold, volatility_col="atr_200", test_size=0.09, val_size=0.09
):
    """Generic data pipeline: triple-barrier -> split -> scale -> binarize."""
    df_lab = triple_barrier_label(
        df, ku=ku, kd=kd, hold=hold, volatility=volatility_col
    )
    df_train, df_val, df_test, scaler = split_scale(
        df_lab,
        target_cols=["y"],
        volatility=volatility_col,
        test_size=test_size,
        val_size=val_size,
    )
    for d in (df_train, df_val, df_test):
        d["y"] = (d["y"] == 2).astype(int)
    return df_train, df_val, df_test, scaler


def run_length_stats_for_value(df: pd.DataFrame, col: str = "y", value=2):
    """Consecutive run stats for a label value."""
    s = df[col]
    run_id = (s != s.shift()).cumsum()
    runs = (
        df.groupby(run_id)[col].agg(value="first", length="size").reset_index(drop=True)
    )
    runs_val = runs[runs["value"] == value]
    return {
        "length_counts": runs_val["length"].value_counts().sort_index(),
        "length_stats": runs_val["length"].describe(),
    }
