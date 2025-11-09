import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from src.utils.data_utils import triple_barrier_label, min_max_label, split_scale, CryptoDataset
from src.models.DL_models import *



if __name__ == "__main__":
    # Update the path below to point to the desired preprocessed parquet file.
    default_parquet = Path(
        "data/processed/ETH_USDT_15m_numeric.parquet")

    df = pd.read_parquet(default_parquet)
    print(df.info())
    
    df_labeled = triple_barrier_label(df, ku=3, kd=1, hold=48, debug=False)
    # df_labeled = min_max_label(df, horizon=48)
    
    # target = ['y_high', 'y_low']
    target = ['y']
    df_w = df_labeled[['log_return_15m'] + target].copy().iloc[-10000:]
    window_size = 96
    # dataset = CryptoDataset(df_w, window_size=window_size,
                            # target=target)
    
    # print(dataset[0])
    # print(dataset[0][0].shape)  # Features shape

    df_train, df_test, df_val, scaler = split_scale(
        df_w, target_cols=target, scale=False)
    
    print(df_train.info())
    # print(df_train[target].value_counts())

    n_features = df_train.drop(columns=target).shape[1]
    
    input_size = (window_size, n_features)

    train_data = CryptoDataset(df_train, window_size=window_size, target=target)
    test_data = CryptoDataset(df_test, window_size=window_size, target=target)

    model = LSTMClassifier(input_size=input_size)
    output = model.train(train_data, test_data)
    # print(output)
    # # model_path = "artifacts/sequence_model.pt"
    # # state_dict = torch.load(model_path)
    # # model._load_state_dict(state_dict)
    output = model.predict(test_data, return_proba=True)
    # print(output[1])

    # print(test_data['y'].value_counts())


    
