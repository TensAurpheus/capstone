import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class CryptoDataset(Dataset):
    def __init__(self, data, window_size=100):
        """
        data: 2D array-like (NumPy, Tensor, or pandas DataFrame)
              Last column = target variable
        """
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        elif not isinstance(data, (np.ndarray, torch.Tensor)):
            raise TypeError(
                "Data must be a pandas DataFrame, NumPy array, or torch Tensor.")

        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.window_size = window_size

        self.features = self.data[:, :-1]
        self.targets = self.data[:, -1]

    def __len__(self):
        return len(self.data) - self.window_size +1

    def __getitem__(self, idx):
        X = self.features[idx: idx + self.window_size]
        y = self.targets[idx: idx + self.window_size]
        return X, y
    

def split_scale(df, feature_cols, target_col='close', test_size=0.2, val_size=0.1, scale=True):
    """
    df: pandas DataFrame containing features including 'close'
    feature_cols: list of columns to use as features
    target_col: column to predict direction (default 'close')
    test_size: fraction of data for test set
    val_size: fraction of data for validation set (from remaining train)
    scale: whether to apply MinMaxScaler to features
    
    Returns:
        train_df, val_df, test_df: DataFrames with features + target
    """
    df = df.copy()

    # Create directional target: 1 if next close > current, -1 if lower
    df['target'] = np.where(df[target_col].shift(-1) > df[target_col], 1, -1)

    # Drop last row (target is NaN)
    df = df[:-1]

    # Optional scaling
    if scale:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Train-test split (chronological)
    n = len(df)
    val_start = int(n * (1 - val_size))
    test_end = val_start
    test_start = int(n * (1 - val_size - test_size))

    train_df = df.iloc[:test_start].reset_index(drop=True)
    val_df = df.iloc[val_start:].reset_index(drop=True)
    test_df = df.iloc[test_start:test_end].reset_index(drop=True)

    timestamps = {'train_start': df.index[0], 'test_start': df.index[test_start], 'val_start': df.index[val_start], 'val_end': df.index[-1]}

    return train_df, val_df, test_df, timestamps

if __name__ == "__main__":
    df = pd.read_parquet("data/BTC_merged_15m_8h.parquet")
    feature_cols = df.columns
    train_df, val_df, test_df, timestamps = split_scale(
        df, feature_cols=feature_cols, target_col='close', scale=True)
    print(train_df.tail())
    print(timestamps)

    data = CryptoDataset(train_df, window_size=3)
    last = len(data)
    print(f"Dataset length: {last}")
    print(data[last-1])