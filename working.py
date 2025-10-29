import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from src.utils.data_utils import triple_barrier_label, min_max_label, split_scale, CryptoDataset



if __name__ == "__main__":
    # Update the path below to point to the desired preprocessed parquet file.
    default_parquet = Path(
        "data/processed/ETH_USDT_15m_numeric.parquet")

    df = pd.read_parquet(default_parquet)
    print(df.info())
    
    # df_labeled = triple_barrier_label(df, ku=3, kd=1, hold=5, debug=False)
    df_labeled = min_max_label(df, horizon=5)
    print(df_labeled.head(50))
    
    target = ['y_high', 'y_low']
    # target = ['y']
    dataset = CryptoDataset(df_labeled, window_size=5,
                            target=target)
    
    print(dataset[0])
    print(dataset[0][0].shape)  # Features shape

    df_train, df_test, df_val, scaler = split_scale(
        df_labeled, target_cols=target, scale=True)
    
    print(df_train.info())
    print(df_train.head(20))
    print(df_labeled[df_labeled['y_low'] >= 0])

    
