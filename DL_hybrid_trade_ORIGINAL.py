import sys, os, json, importlib.util, zipfile, joblib, math, random, numbers
from pathlib import Path
from typing import List, Sequence, Tuple, Union, Optional, Callable, Dict, Any, Literal
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor
import torch

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
def set_deterministic(seed: int = 42):
    import os, random, numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # comment out the next line to avoid CuBLAS error:
    # torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False#run length stats
def run_length_stats_for_value(df: pd.DataFrame, col: str = "y", value=2):
    """
    Return stats on consecutive runs where df[col] == value.
    """
    s = df[col]

    # Identify runs
    run_id = (s != s.shift()).cumsum()
    runs = (
        df.groupby(run_id)[col]
          .agg(value="first", length="size")
          .reset_index(drop=True)
    )

    # Keep only runs with the given value
    runs_val = runs[runs["value"] == value]

    # Stats
    length_counts = runs_val["length"].value_counts().sort_index()
    length_stats = runs_val["length"].describe()

    return {
        "length_counts": length_counts,  # how many runs of length 1,2,3,...
        "length_stats": length_stats     # count, mean, std, min, quartiles, max
    }def periods_per_year(tf: str, days_in_year: float = 365.0) -> float:
    """
    Calculate number of periods in a year for a given timeframe string.
    Examples of tf: '1m', '15m', '1h', '4h', '1d', '1w', '1mo'
    """
    tf = tf.strip().lower()

    # Split numeric part and unit part
    num_str = ''.join(ch for ch in tf if ch.isdigit())
    unit = ''.join(ch for ch in tf if ch.isalpha())

    if not num_str or not unit:
        raise ValueError(f"Invalid timeframe format: {tf}")

    n = float(num_str)

    minutes_per_unit = {
        'm': 1,                 # minute
        'h': 60,                # hour
        'd': 24 * 60,           # day
        'w': 7 * 24 * 60,       # week
        'mo': 30 * 24 * 60,     # month (approx)
    }

    if unit not in minutes_per_unit:
        raise ValueError(f"Unsupported timeframe unit: {unit}")

    minutes_per_period = n * minutes_per_unit[unit]
    minutes_per_year = days_in_year * 24 * 60

    return minutes_per_year / minutes_per_period
def make_utility_class_weights(
    y_train: np.ndarray,
    ku: float,
    kd: float,
    mode: str = "balanced",
) -> np.ndarray:
    """
    Utility-weighted class weights for triple-barrier labels.

    Classes:
        0 = expiry
        1 = SL
        2 = TP

    Idea:
      - start from standard frequency-based class weights
      - multiply by per-class utility factors:
          w_expiry ~ 1
          w_SL     ~ kd  (penalize SL mistakes ~ loss magnitude)
          w_TP     ~ ku  (reward TP detection ~ reward magnitude)
    """

    y_train = np.asarray(y_train, dtype=int).ravel()
    classes = np.sort(np.unique(y_train))
    num_classes = len(classes)

    if mode == "balanced":
        base = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        ).astype(np.float32)
    elif mode == "none":
        base = np.ones(num_classes, dtype=np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # # per-class utility factors
    # util = np.array([
    #     1.0,     # expiry
    #     1.0,
    #     float(ku/kd),
    # ], dtype=np.float32)

    w = base
    # normalize for numerical stability (not strictly required)
    # w = w / w.mean()

    return w.astype(np.float32)
def build_feature_df(
    df: pd.DataFrame,
    blocks: list[str],
    target_col: str = "y",
    volatility_col: str = "atr_200",
) -> pd.DataFrame:
    """
    Subset df to:
      - all features from the specified blocks,
      - OHLC,
      - volatility_col,
      - target_col.
    Only keeps columns that actually exist in df.
    """
    cols = set()
    for b in blocks:
        if b not in FEATURE_BLOCKS:
            raise ValueError(f"Unknown feature block: {b}")
        cols.update(FEATURE_BLOCKS[b])

    always = {"open", "high", "low", "close", target_col, volatility_col}
    all_needed = (cols | always) & set(df.columns)

    return df[list(all_needed)].copy()
def random_preds_from_train_priors(y_train, y_val_len=None, seed=42, return_proba=True):
    """
    Sample random predictions for the validation set using class priors from y_train.
    Works for binary or multiclass labels.
    """
    rng = np.random.default_rng(seed)
    y_train = y_train.to_numpy().ravel()
    classes, counts = np.unique(y_train, return_counts=True)
    priors = counts / counts.sum()

    n = len(y_val_len) if hasattr(y_val_len, "__len__") else int(y_val_len or 0)
    if n == 0:
        raise ValueError("Provide y_val (array-like) or its length via y_val_len.")

    # sample labels according to priors
    y_pred = rng.choice(classes, size=n, p=priors)

    if not return_proba:
        return y_pred

    # constant per-class probabilities = priors (shape [n, C])
    proba = np.tile(priors, (n, 1))
    # reorder columns to match class order
    return y_pred, classes, probadef make_random_prediction_df_from_priors(
    y_train: pd.Series,
    y_true_test: pd.Series,
    seed: int = 42,
    n_classes: int = 3,
) -> pd.DataFrame:
    """
    Use class frequencies from y_train to generate random labels and
    constant per-class probabilities for y_true_test.

    Returns a DataFrame with columns:
      ['true', 'pred', 'p0', 'p1', 'p2']  (for n_classes=3)
    compatible with triple_barrier_metrics.
    """
    # reuse your existing function
    y_pred, classes, proba = random_preds_from_train_priors(
        y_train=y_train,
        y_val_len=len(y_true_test),
        seed=seed,
        return_proba=True,
    )

    y_true_arr = np.asarray(y_true_test, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    # init proba df with zeros for all classes 0..n_classes-1
    proba_df = pd.DataFrame(
        np.zeros((len(y_true_arr), n_classes), dtype=float),
        columns=[f"p{i}" for i in range(n_classes)],
    )

    # fill only the classes that actually appear in y_train
    for j, cls in enumerate(classes):
        col = f"p{int(cls)}"
        if col in proba_df.columns:
            proba_df[col] = proba[:, j]

    pred_df = pd.DataFrame(
        {
            "true": y_true_arr,
            "pred": y_pred_arr,
        }
    )

    pred_df = pd.concat([pred_df, proba_df], axis=1)
    return pred_df
def in_kaggle() -> bool:
    # 1) Kaggle-only module
    if importlib.util.find_spec("kaggle_secrets") is not None:
        return True
    # 2) Kaggle env vars
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE") or os.environ.get("KAGGLE_USER_SECRETS_TOKEN"):
        return True
    # 3) Kaggle filesystem
    if Path("/kaggle/input").exists() or Path("/kaggle/working").exists():
        return True
    return Falsedef in_colab() -> bool:
    # Colab has the google.colab package available
    try:
        import google.colab  # type: ignore
        return True
    except ImportError:
        return Falsedef to_jsonable(x):
    if isinstance(x, pd.DataFrame):
        return x.to_dict(orient="records")
    if isinstance(x, pd.Series):
        return x.to_dict()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if hasattr(x, "isoformat"):  # pd.Timestamp, datetime
        return x.isoformat()
    return x

def walk(obj):
    if isinstance(obj, dict):
        return {k: walk(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [walk(v) for v in obj]
    return to_jsonable(obj)#Delete outputs.zip
def delete_zip():
    zip_path = "/kaggle/working/outputs.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("Deleted:", zip_path)
    else:
        print("Not found:", zip_path)#ZIP OUTPUTS

def zip_res():
    root = "/kaggle/working"
    zip_path = os.path.join(root, "outputs.zip")
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for r, _, files in os.walk(root):
            for f in files:
                full = os.path.join(r, f)
                if os.path.abspath(full) == os.path.abspath(zip_path):  # skip self
                    continue
                z.write(full, arcname=os.path.relpath(full, root))#UTILS FOR PROCESSING MODEL SWEEP RESULTS

def build_model_metrics_df(all_results):
    """
    Flatten all_results into a DF with classification metrics
    for each tf, ES metric, split, model_type, and mask_type (raw/masked).
    """
    rows = []

    for (tf, es_metric), info in all_results.items():
        # DL models: val + test
        for split_name, split_key in [("val", "val_results"), ("test", "test_results")]:
            split_res = info[split_key]
            for model_type, res_model in split_res.items():
                # raw
                mm_raw = res_model.get("model_metrics_raw", {})
                rows.append(
                    {
                        "tf": tf,
                        "es_metric": es_metric,
                        "split": split_name,
                        "model_type": model_type,
                        "mask_type": "raw",
                        **mm_raw,
                    }
                )

                # masked
                mm_masked = res_model.get("model_metrics_masked", {})
                rows.append(
                    {
                        "tf": tf,
                        "es_metric": es_metric,
                        "split": split_name,
                        "model_type": model_type,
                        "mask_type": "masked",
                        **mm_masked,
                    }
                )

        # Baseline (logreg): val + test
        for split_name, base_key in [("val", "baseline_val"), ("test", "baseline_test")]:
            base_res = info[base_key]

            mm_raw = base_res.get("model_metrics_raw", {})
            rows.append(
                {
                    "tf": tf,
                    "es_metric": es_metric,
                    "split": split_name,
                    "model_type": "logreg",
                    "mask_type": "raw",
                    **mm_raw,
                }
            )

            mm_masked = base_res.get("model_metrics_masked", {})
            rows.append(
                {
                    "tf": tf,
                    "es_metric": es_metric,
                    "split": split_name,
                    "model_type": "logreg",
                    "mask_type": "masked",
                    **mm_masked,
                }
            )

    model_metrics_df = pd.DataFrame(rows)
    return model_metrics_df


# ---------------------------------------------------------
# BUILD TRADE-METRICS TABLE
# ---------------------------------------------------------
def _is_mr_key(k):
    return isinstance(k, numbers.Number)

def build_trade_metrics_df(all_results):
    """
    Flatten all_results into a DF with trade metrics
    for each tf, ES metric, split, model_type, and min_return (mr).

    Includes:
      - DL models (val + test, direct runs + sweep bests)
      - Baseline (val + test, direct runs + sweep bests)
      - BnH rows for val/test
    """
    rows = []

    for (tf, es_metric), info in all_results.items():
        # ---------------- BnH ----------------
        for split_name, bnh_key in [("val", "val_bnh"), ("test", "test_bnh")]:
            bnh_ret = info[bnh_key]
            rows.append(
                {
                    "tf": tf,
                    "es_metric": es_metric,
                    "split": split_name,
                    "model_type": "BnH",
                    "mr": None,
                    "selection": "bnh",
                    "total_return": bnh_ret,
                }
            )

        # ---------------- DL models ----------------
        # Direct runs (val/test)
        for model_type in model_types:
            # VAL
            val_res_model = info["val_results"][model_type]
            for k, v in val_res_model.items():
                if not _is_mr_key(k):
                    continue
                tm = v["trade_metrics"]
                rows.append(
                    {
                        "tf": tf,
                        "es_metric": es_metric,
                        "split": "val",
                        "model_type": model_type,
                        "mr": k,
                        "selection": "direct",
                        **tm,
                    }
                )

            # TEST
            test_res_model = info["test_results"][model_type]
            for k, v in test_res_model.items():
                if not _is_mr_key(k):
                    continue
                tm = v["trade_metrics"]
                rows.append(
                    {
                        "tf": tf,
                        "es_metric": es_metric,
                        "split": "test",
                        "model_type": model_type,
                        "mr": k,
                        "selection": "direct",
                        **tm,
                    }
                )

        # Sweep bests (VAL only)
        for model_type, best_dict in info["best_per_model"].items():
            for sel_key in ["best_return", "best_martin"]:
                mr_star = best_dict[sel_key]["mr"]
                metrics_star = best_dict[sel_key]["metrics"]
                if mr_star is None or metrics_star is None:
                    continue
                rows.append(
                    {
                        "tf": tf,
                        "es_metric": es_metric,
                        "split": "val",
                        "model_type": model_type,
                        "mr": mr_star,
                        "selection": sel_key,
                        **metrics_star,
                    }
                )

        # ---------------- Baseline (logreg) ----------------
        # Direct runs: val/test
        for split_name, base_key in [("val", "baseline_val"), ("test", "baseline_test")]:
            base = info[base_key]
            for k, v in base.items():
                if not _is_mr_key(k):
                    continue
                tm = v["trade_metrics"]
                rows.append(
                    {
                        "tf": tf,
                            "es_metric": es_metric,
                        "split": split_name,
                        "model_type": "logreg",
                        "mr": k,
                        "selection": "direct",
                        **tm,
                    }
                )

        # Sweep best for baseline (VAL)
        base_best = info["baseline_best"]
        for sel_key in ["best_return", "best_martin"]:
            mr_star = base_best[sel_key]["mr"]
            metrics_star = base_best[sel_key]["metrics"]
            if mr_star is None or metrics_star is None:
                continue
            rows.append(
                {
                    "tf": tf,
                    "es_metric": es_metric,
                    "split": "val",
                    "model_type": "logreg",
                    "mr": mr_star,
                    "selection": sel_key,
                    **metrics_star,
                }
            )

    trade_metrics_df = pd.DataFrame(rows)
    return trade_metrics_df
#data utils

def split_scale(df, target_cols='y', test_size=0.09, val_size=0.09, scale=True, for_test=False, volatility='atr_14'):
    """Split into train/val/test (or train+val/test for for_test) and selectively scale features."""
    df = df.copy().replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

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


def triple_barrier_label(df, close='close', high='high', low='low', volatility='atr_14', ku=1.5, kd=1.5, hold=16, labels=None, debug=False):
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

    return out.iloc[:n - horizon]# DATA PIPE

def data_pipe(df, ku, kd, hold, window_size, volatility_col='atr_14', test_size = 0.09, val_size = 0.09):
    # --- 1) Triple-barrier labeling: 3-class y in {0,1,2} ---
    df_labeled = triple_barrier_label(
        df,
        ku=ku,
        kd=kd,
        hold=hold,
        debug=False,
        volatility=volatility_col,
    )
    target = ['y']
    # print(df_labeled[target].value_counts(normalize=True))

    # --- 2) Split + scale with 3-class labels (no binarization yet) ---
    df_train, df_val, df_test, scaler = split_scale(
        df_labeled,
        target_cols=target,
        scale=True,
        volatility=volatility_col,
        test_size=test_size,
        val_size=val_size,
    )

    # --- 3) One neat table of 3-class label distribution per split ---
    def _vc(d):
        vc = d['y'].value_counts().sort_index()
        prop = vc / vc.sum()
        return vc, prop

    train_vc, train_prop = _vc(df_train)
    val_vc,   val_prop   = _vc(df_val)
    test_vc,  test_prop  = _vc(df_test)

    classes = sorted(set(train_vc.index) | set(val_vc.index) | set(test_vc.index))

    # reindex to ensure all classes appear in all splits
    train_vc   = train_vc.reindex(classes, fill_value=0)
    val_vc     = val_vc.reindex(classes, fill_value=0)
    test_vc    = test_vc.reindex(classes, fill_value=0)
    train_prop = train_prop.reindex(classes, fill_value=0.0)
    val_prop   = val_prop.reindex(classes, fill_value=0.0)
    test_prop  = test_prop.reindex(classes, fill_value=0.0)

    # build wide table with MultiIndex columns: (split, metric)
    table = pd.DataFrame(
        {
            ("train", "count"): train_vc,
            ("train", "prop"):  train_prop,
            ("val",   "count"): val_vc,
            ("val",   "prop"):  val_prop,
            ("test",  "count"): test_vc,
            ("test",  "prop"):  test_prop,
        },
        index=classes,
    )

    table.columns = pd.MultiIndex.from_tuples(table.columns, names=["split", "metric"])

    print("=== 3-class label distribution per split (before binarization) ===")
    # round only proportions for readability
    table_to_print = table.copy()
    for split in ["train", "val", "test"]:
        table_to_print[(split, "prop")] = table_to_print[(split, "prop")].round(4)
    print(table_to_print)

    # --- 4) Binarize: TP (=2) vs not-TP (0 or 1), keep original 3-class as y_3c ---
    for d in (df_train, df_val, df_test):
        # d['y_3c'] = d['y'].copy()
        d['y'] = (d['y'] == 2).astype(int)

    return df_train, df_val, df_test, scaler
def fetch_data(tf):

    default_parquet = Path(f"/kaggle/input/btc-new{tf}/BTC_USDT_{tf}_futures.parquet")
    # default_parquet = Path(f"/kaggle/input/btc-{tf}-atr200-distances/BTC_USDT_{tf}_futures.parquet")
    # default_parquet = Path("/kaggle/input/btc-1h/BTC_USDT_1h_futures.parquet")
    volatility_col = 'atr_200'
    num_classes = 2
    probability_col = ['p1']
    
    df = pd.read_parquet(default_parquet)
    if (volatility_col != 'atr_14') and ('atr_14' in df.columns):
        df = df.drop(columns=['atr_14'])

    return df#DL models

ArrayLike = Union[Sequence[float], Sequence[Sequence[float]], Tensor]


def _select_device() -> Tuple[torch.device, bool]:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        return torch.device("cuda"), count > 1

    return torch.device("cpu"), False


def _ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class _RNNClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_size,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        bidirectional: bool,
        rnn_type: str,
    ) -> None:
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(
        #     hidden_size * (2 if bidirectional else 1), num_classes*4)
        # self.fc2 = nn.Linear(num_classes*4, num_classes*2)
        self.out = nn.Linear(
            hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, _ = self.rnn(inputs)
        last_timestep = outputs[:, -1, :]
        out = self.dropout(last_timestep)
        # out = self.fc(out)
        # out = torch.relu(out)
        # out = torch.relu(self.fc2(out))
        out = self.out(out)

        return out


class _SequenceClassifier:
    def __init__(
        self,
        *,
        input_size,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        bidirectional: bool,
        rnn_type: str,
        random_state: int = 42
    ) -> None:
        self.random_state = random_state
        device, use_parallel = _select_device()
        self.num_classes = num_classes
        model = _RNNClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
        )
        if use_parallel:
            model = nn.DataParallel(model)
        self.model = model.to(device)
        self.device = device

   
    def _evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> Tuple[float, Optional[float]]:
        """
        Evaluate on a dataloader.
    
        Returns
        -------
        val_loss : float
        val_metric : float or None
            - If metric_fn is None, val_metric is None and the behaviour is
              identical to your old version (loss-only).
            - If metric_fn is not None, val_metric is the metric on the full val set.
        """
        self.model.eval()
        running_loss = 0.0
        sample_count = 0
    
        all_logits = []
        all_labels = []
    
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.view(-1).long().to(self.device)
    
                preds = self.model(features)
                loss = criterion(preds, labels)
    
                batch = labels.size(0)
                running_loss += loss.item() * batch
                sample_count += batch
    
                if metric_fn is not None:
                    all_logits.append(preds.detach().cpu())
                    all_labels.append(labels.detach().cpu())
    
        val_loss = running_loss / max(sample_count, 1)
    
        if metric_fn is None:
            # Backwards-compatible: loss-only behaviour
            return val_loss, None
    
        if not all_logits:
            # Degenerate case: empty loader
            return val_loss, 0.0
    
        logits = torch.cat(all_logits, dim=0)
        y_true = torch.cat(all_labels, dim=0).numpy()
        probs = torch.softmax(logits, dim=1).numpy()
    
        val_metric = metric_fn(y_true, probs)
        return val_loss, val_metric
    
    def train(
        self,
        train_data: Tuple[ArrayLike, Sequence[int]],
        val_data: Tuple[ArrayLike, Sequence[int]],
        *,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 5,
        model_path: Union[str, Path] = "artifacts/sequence_model.pt",
        loss_plot_path: Union[str, Path] = "artifacts/loss_curve.png",
        num_workers: int = 0,
        weights: Optional[ArrayLike] = None,
        early_metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        early_metric_mode: str = "max",  # "min" or "max" when using metric
    ) -> dict:
    
        """
        Train the sequence model with early stopping.
    
        By default (early_metric_fn=None):
            - Early stopping is based on validation loss (same as before).
    
        If early_metric_fn is provided:
            - early_metric_fn(y_true, probs) -> scalar to monitor.
            - early_metric_mode:
                * "max" for metrics where higher is better (F1, EU, etc.).
                * "min" for metrics where lower is better (Brier, etc.).
        """
    
        set_deterministic(self.random_state)
    
        if weights is None:
            weights = np.ones(self.num_classes, dtype=np.float32)
    
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
    
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
        train_losses: List[float] = []
        val_losses: List[float] = []
        val_metrics: List[Optional[float]] = []
    
        best_val_loss = float("inf")      # best loss seen
        best_monitor: Optional[float] = None  # value used for early stopping
        best_state: Optional[dict] = None
        bad_epochs = 0
        tol = 1e-6
    
        for epoch in range(1, epochs + 1):
            # -----------------
            # Train phase
            # -----------------
            self.model.train()
            running_loss = 0.0
            sample_count = 0
    
            for features, labels in tqdm(
                train_loader, desc=f"Epoch {epoch}/{epochs}"
            ):
                features = features.to(self.device)
                labels = labels.view(-1).long().to(self.device)
    
                optimizer.zero_grad()
                preds = self.model(features)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
    
                batch = labels.size(0)
                running_loss += loss.item() * batch
                sample_count += batch
    
            train_loss = running_loss / max(sample_count, 1)
            train_losses.append(train_loss)
    
            # -----------------
            # Validation phase
            # -----------------
            val_loss, val_metric = self._evaluate(
                val_loader,
                criterion,
                metric_fn=early_metric_fn,
            )
            val_losses.append(val_loss)
            val_metrics.append(val_metric)
    
            # Decide what we monitor for early stopping
            if early_metric_fn is None:
                monitor_value = val_loss
                mode = "min"
            else:
                monitor_value = val_metric
                mode = early_metric_mode
    
            
            # Initialize best state
            if best_monitor is None:
                best_monitor = monitor_value
                best_val_loss = val_loss
                bad_epochs = 0
                best_state = self._state_dict()
                torch.save(best_state, _ensure_dir(model_path))
            else:
                if mode == "min":
                    improved = monitor_value < best_monitor - tol
                else:
                    improved = monitor_value > best_monitor + tol
    
                if improved:
                    best_monitor = monitor_value
                    best_val_loss = val_loss
                    bad_epochs = 0
                    best_state = self._state_dict()
                    torch.save(best_state, _ensure_dir(model_path))
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        break
            # print(best_monitor, monitor_value)
    
        # Restore best weights
        state_dict = torch.load(_ensure_dir(model_path), map_location=self.device)
        self._load_state_dict(state_dict)
    
        # Plot (same as before)
        self._plot_losses(train_losses, val_losses, loss_plot_path)
    
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics,
            "best_val_loss": best_val_loss,
            "best_monitor_value": best_monitor,
            "monitor_mode": ("loss" if early_metric_fn is None else mode),
            "epochs_trained": len(train_losses),
            "model_path": str(model_path),
            "loss_plot_path": str(loss_plot_path),
        }

   
    def _state_dict(self) -> dict:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def _load_state_dict(self, state_dict: dict) -> None:
        target = self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model
        target.load_state_dict(state_dict)

    def _plot_losses(
        self,
        train_losses: Sequence[float],
        val_losses: Sequence[float],
        path: Union[str, Path],
    ) -> None:
        path = _ensure_dir(path)
        plt.figure(figsize=(6, 4))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="train", marker="o")
        plt.plot(epochs, val_losses, label="val", marker="s")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    # ---------------------------------------------------------------- inference
    def predict(
        self,
        data: ArrayLike,
        *,
        batch_size: int = 128,
    ):

        loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
        )

        preds, probas, y_true = [], [], []

        self.model.eval()
        
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                y_true.extend(y.view(-1).cpu().tolist())

                logits = self.model(X)                 # [B, C] logits
                probs = torch.softmax(logits, dim=1)   # [B, C] probs

                probas.extend(probs.cpu().tolist())
                preds.extend(probs.argmax(dim=1).cpu().tolist())

        arr = np.asarray(probas)                       # (N, C)
        cols = [f"p{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)
        df.insert(0, "pred", preds)
        df["true"] = np.asarray(y_true).astype(int)

        return df


class LSTMClassifier(_SequenceClassifier):
    def __init__(
        self,
        *,
        input_size,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=False,
            rnn_type="lstm",
        )


class BiLSTMClassifier(_SequenceClassifier):
    def __init__(
        self,
        *,
        input_size,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=True,
            rnn_type="lstm",
        )


class GRUClassifier(_SequenceClassifier):
    def __init__(
        self,
        *,
        input_size,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=False,
            rnn_type="gru",
        )

class BiGRUClassifier(_SequenceClassifier):
    def __init__(
        self,
        *,
        input_size,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=True,
            rnn_type="gru",
        )# hybrid_gru_ml



ArrayLike = Union[Sequence[float], Sequence[Sequence[float]], Tensor]


# ------------------------- utils -------------------------
def _select_device() -> Tuple[torch.device, bool]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.device_count() > 1
    return torch.device("cpu"), False


def _ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _to_loader(data, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# ------------------------- encoder -------------------------
class _GRUEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool):
        super().__init__()
        self.bi = 2 if bidirectional else 1
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

    def forward(self, x: Tensor) -> Tensor:
        _, h_n = self.gru(x)  # (L*d, B, H)
        if self.bi == 2:
            z = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)
        else:
            z = h_n[-1]                                # (B, H)
        return z


class _GRUPretrainHead(nn.Module):
    def __init__(self, encoder: _GRUEncoder, out_classes: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        hdim = encoder.gru.hidden_size * \
            (2 if encoder.gru.bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hdim, out_classes)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        return self.out(self.drop(z))


# ------------------------- base hybrid -------------------------
class _HybridGRUML:
    """
    GRU pretrain on labels -> extract last hidden state -> sklearn ML head.
    Artifacts: encoder.pt, ml_model.pkl, meta.json, loss_curve.png
    """
    ML_KIND = "base"

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        random_state: int = 42,
        
    ):
        self.num_classes = num_classes
        self.random_state = random_state

        device, use_parallel = _select_device()
        enc = _GRUEncoder(input_size, hidden_size,
                          num_layers, dropout, bidirectional)
        head = _GRUPretrainHead(enc, out_classes=num_classes, dropout=dropout)
        if use_parallel:
            head = nn.DataParallel(head)
        self.model = head.to(device)
        self.device = device
        self.use_parallel = use_parallel

        self._ml: Any = None
        self._meta: Dict[str, Any] = dict(
            ml_kind=self.ML_KIND,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
            random_state=random_state,
        )
        self._meta.update({
            "use_pooling": True,          # new
            "pool_last_k": 16,            # new
            "add_gru_proba": True         # new
        }) 

    # ----------------- training -----------------
    def train(
        self,
        train_data,
        val_data,
        *,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 8,
        num_workers: int = 0,
        class_weight_mode: str = "balanced",  # kept for interface; not used directly here
        model_path: Union[str, Path] = "artifacts/hybrid_gru_ml",
        loss_plot_path: Optional[Union[str, Path]] = None,
        use_gpu_in_ml: bool = True,
        weights = None,
        # ---- feature knobs ----
        use_pooling: bool = True,
        pool_last_k: int = 16,
        add_gru_proba: bool = True,
        # ---- early stopping by metric ----
        early_metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        early_metric_mode: str = "min",   # "min" for Brier, "max" e.g. for F1
    ) -> dict:
        set_deterministic(self.random_state)
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        if loss_plot_path is None:
            loss_plot_path = model_dir / "loss_curve.png"

        # persist meta for inference
        self._meta["use_pooling"] = bool(use_pooling)
        self._meta["pool_last_k"] = int(pool_last_k)
        self._meta["add_gru_proba"] = bool(add_gru_proba)

        if early_metric_fn is not None:
            assert early_metric_mode in {"min", "max"}, "early_metric_mode must be 'min' or 'max'"

        Ltr = _to_loader(train_data, batch_size, shuffle=True,  num_workers=num_workers)
        Lva = _to_loader(val_data,   batch_size, shuffle=False, num_workers=num_workers)

        # --- class weights for GRU pretrain (if provided) ---
        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            w = None

        crit = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_metric: Optional[float] = None
        best_state = None
        bad = 0
        tr_losses, va_losses = [], []

        for ep in range(1, epochs + 1):
            # ----------------- train epoch -----------------
            self.model.train()
            tr_loss, n = 0.0, 0
            for Xb, yb in tqdm(Ltr, desc=f"GRU pretrain {ep}/{epochs}", leave=False):
                Xb = Xb.to(self.device)
                yb = yb.view(-1).long().to(self.device)

                opt.zero_grad()
                logits = self.model(Xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()

                bs = yb.size(0)
                tr_loss += loss.item() * bs
                n += bs
            tr_losses.append(tr_loss / max(n, 1))

            # ----------------- validation epoch -----------------
            self.model.eval()
            va_loss, n = 0.0, 0
            all_probs = []
            all_true = []

            with torch.no_grad():
                for Xb, yb in Lva:
                    Xb = Xb.to(self.device)
                    yb = yb.view(-1).long().to(self.device)

                    logits = self.model(Xb)
                    loss = crit(logits, yb)

                    bs = yb.size(0)
                    va_loss += loss.item() * bs
                    n += bs

                    if early_metric_fn is not None:
                        probs = torch.softmax(logits, dim=1)
                        all_probs.append(probs.cpu().numpy())
                        all_true.append(yb.cpu().numpy())

            va_loss = va_loss / max(n, 1)
            va_losses.append(va_loss)
            best_val_loss = min(best_val_loss, va_loss)

            # --- compute early-stop metric (if requested) ---
            if early_metric_fn is not None:
                if all_probs:
                    y_true = np.concatenate(all_true)
                    probs_np = np.vstack(all_probs)
                    current_metric = float(early_metric_fn(y_true, probs_np))
                else:
                    current_metric = float("inf") if early_metric_mode == "min" else float("-inf")

                if best_metric is None:
                    improved = True
                else:
                    if early_metric_mode == "min":
                        improved = current_metric < best_metric - 1e-6
                    else:
                        improved = current_metric > best_metric + 1e-6

                if improved:
                    best_metric = current_metric
                    best_state = self._state_dict()
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            else:
                # fallback: early stop on val loss (old behavior)
                if va_loss < best_val_loss - 1e-6:
                    best_val_loss = va_loss
                    best_state = self._state_dict()
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

        # restore best encoder weights
        if best_state is not None:
            self._load_state_dict(best_state)

        # plot loss curves
        self._plot_losses(tr_losses, va_losses, loss_plot_path)

        # ----------------- feature extraction for ML head -----------------
        Ztr, ytr = self._extract_features(Ltr)
        Zva, yva = self._extract_features(Lva)

        # ML head (sklearn API)
        self._ml = self._fit_ml(Ztr, ytr, Zva, yva, use_gpu_in_ml)

        self.save(model_dir)

        result = {
            "best_val_loss": best_val_loss,
            "epochs_trained": len(tr_losses),
            "loss_plot_path": str(loss_plot_path),
            "model_dir": str(model_dir),
        }
        if early_metric_fn is not None:
            result["best_early_metric"] = best_metric
        return result


    # ----------------- inference -----------------
    def predict(self, data, *, batch_size: int = 1024, num_workers: int = 0) -> pd.DataFrame:
        assert self._ml is not None, "ML head not trained/loaded."
        L = _to_loader(data, batch_size, shuffle=False, num_workers=num_workers)
        Z, y = self._extract_features(L)   # <- was _extract_embeddings
        proba = self._predict_proba_ml(Z)
        yhat = proba.argmax(axis=1)

        cols = [f"p{i}" for i in range(proba.shape[1])]
        df = pd.DataFrame(proba, columns=cols)
        df.insert(0, "pred", yhat)
        df["true"] = y.astype(int)
        return df

    # ----------------- save / load -----------------
    def save(self, model_dir: Union[str, Path]) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._state_dict(), model_dir / "encoder.pt")
        joblib.dump(self._ml, model_dir / "ml_model.pkl")
        with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "_HybridGRUML":
        model_dir = Path(model_dir)
        with open(model_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        obj = cls(
            input_size=meta["input_size"],
            hidden_size=meta["hidden_size"],
            num_layers=meta["num_layers"],
            num_classes=meta["num_classes"],
            dropout=meta["dropout"],
            bidirectional=meta["bidirectional"],
            random_state=meta.get("random_state", 42),
        )
        state = torch.load(model_dir / "encoder.pt", map_location=obj.device)
        obj._load_state_dict(state)
        obj._ml = joblib.load(model_dir / "ml_model.pkl")
        obj._meta = meta
        return obj

    # ----------------- internals -----------------
    def _state_dict(self) -> dict:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def _load_state_dict(self, sd: dict) -> None:
        target = self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model
        target.load_state_dict(sd, strict=True)

    def _plot_losses(self, tr: Sequence[float], va: Sequence[float], path: Union[str, Path]) -> None:
        path = _ensure_dir(path)
        plt.figure(figsize=(6, 4))
        e = range(1, len(tr) + 1)
        plt.plot(e, tr, label="train", marker="o")
        plt.plot(e, va, label="val", marker="s")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _extract_features(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (X_ml, y), where X_ml is the concatenation of:
          - last hidden state (what you had before),
          - optional mean/max pooling over last k steps of top-layer GRU outputs,
          - optional GRU softmax probabilities from the pretrain head.
        """
        self.model.eval()
        use_pool = bool(self._meta.get("use_pooling", True))
        k = int(self._meta.get("pool_last_k", 16))
        add_prob = bool(self._meta.get("add_gru_proba", True))

        # unwrap modules
        if isinstance(self.model, nn.DataParallel):
            enc = self.model.module.encoder
            head = self.model.module
        else:
            enc = self.model.encoder
            head = self.model

        Z_list, Y_list = [], []
        with torch.no_grad():
            gru = enc.gru  # nn.GRU
            bi = 2 if gru.bidirectional else 1
            H = gru.hidden_size

            for Xb, yb in loader:
                Xb = Xb.to(self.device)

                # 1) last hidden state (your original feature)
                out, h_n = gru(Xb)                 # out: [B,T,bi*H], h_n: [L*bi,B,H]
                if bi == 2:
                    z_last = torch.cat([h_n[-2], h_n[-1]], dim=1)   # [B, 2H]
                else:
                    z_last = h_n[-1]                                 # [B, H]

                feats = [z_last]

                # 2) temporal pooling over top-layer outputs (mean & max over last k steps)
                if use_pool and out.size(1) >= 1:
                    kk = min(k, out.size(1))
                    tail = out[:, -kk:, :]             # [B, kk, bi*H]
                    mean_pool = tail.mean(dim=1)       # [B, bi*H]
                    max_pool, _ = tail.max(dim=1)      # [B, bi*H]
                    feats += [mean_pool, max_pool]

                # 3) GRU softmax probabilities from pretrain head
                if add_prob:
                    logits = head(Xb)                  # linear layer over z_last inside head
                    probs = torch.softmax(logits, dim=-1)  # [B, C]
                    feats += [probs]

                Zb = torch.cat(feats, dim=1).cpu().numpy()
                Z_list.append(Zb)
                Y_list.append(yb.view(-1).cpu().numpy())

        Z = np.vstack(Z_list)
        Y = np.concatenate(Y_list)
        return Z, Y

    # hooks to override
    def _fit_ml(self, Ztr, ytr, Zva, yva, use_gpu: bool):
        raise NotImplementedError

    def _predict_proba_ml(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ------------------------- sklearn ML heads -------------------------
class GRUCatBoostClassifier(_HybridGRUML):
    ML_KIND = "catboost"

    def _fit_ml(self, Ztr, ytr, Zva, yva, use_gpu: bool):
        model = CatBoostClassifier(
            loss_function="MultiClass",
            auto_class_weights='Balanced',
            learning_rate=0.05,
            depth=6,
            n_estimators=2000,
            random_state=self.random_state,
            task_type="GPU" if (
                use_gpu and torch.cuda.is_available()) else "CPU",
            od_type="Iter",
            od_wait=100,
            verbose=False,
        )
        model.fit(Ztr, ytr, eval_set=(Zva, yva),
                  verbose=False, use_best_model=True)
        return model

    def _predict_proba_ml(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(self._ml.predict_proba(Z))


class GRULightGBMClassifier(_HybridGRUML):
    ML_KIND = "lightgbm"

    def _fit_ml(self, Ztr, ytr, Zva, yva, use_gpu: bool):
        model = LGBMClassifier(
            objective="multiclass",
            num_class=self.num_classes,
            learning_rate=0.05,
            num_leaves=63,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=1,
            n_estimators=4000,
            random_state=self.random_state,
            device="gpu" if (use_gpu and torch.cuda.is_available()) else "cpu",
        )
        model.fit(
            Ztr, ytr,
            eval_set=[(Zva, yva)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
        return model

    def _predict_proba_ml(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(self._ml.predict_proba(Z))


class GRUXGBClassifier(_HybridGRUML):
    ML_KIND = "xgboost"

    def _fit_ml(self, Ztr, ytr, Zva, yva, use_gpu: bool):
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=self.num_classes,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            n_estimators=4000,
            eval_metric="mlogloss",
            tree_method="gpu_hist" if (
                use_gpu and torch.cuda.is_available()) else "hist",
            random_state=self.random_state,
            n_jobs=0,
        )
        model.fit(
            Ztr, ytr,
            eval_set=[(Zva, yva)],
            early_stopping_rounds=200,
            verbose=False,
        )
        return model

    def _predict_proba_ml(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(self._ml.predict_proba(Z))#  hybrid_lstm_ml


# ------------------------- utils -------------------------
def _select_device() -> Tuple[torch.device, bool]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.device_count() > 1
    return torch.device("cpu"), False


def _ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _to_loader(data, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# ------------------------- encoder -------------------------
class _LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool):
        super().__init__()
        self.bi = 2 if bidirectional else 1
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # <- same convention as your GRU should use
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

    def forward(self, x: Tensor) -> Tensor:
        # LSTM returns: output, (h_n, c_n)
        _, (h_n, c_n) = self.LSTM(x)  # h_n: [L*d, B, H]
        if self.bi == 2:
            z = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)
        else:
            z = h_n[-1]                               # (B, H)
        return z


class _LSTMPretrainHead(nn.Module):
    def __init__(self, encoder: _LSTMEncoder, out_classes: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        hdim = encoder.LSTM.hidden_size * (2 if encoder.LSTM.bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hdim, out_classes)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        return self.out(self.drop(z))


# ------------------------- base hybrid -------------------------
class _HybridLSTMML:
    """
    LSTM pretrain on labels -> extract features -> sklearn ML head.
    Artifacts: encoder.pt, ml_model.pkl, meta.json, loss_curve.png
    """
    ML_KIND = "base"

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        random_state: int = 42,
    ):
        self.num_classes = num_classes
        self.random_state = random_state

        device, use_parallel = _select_device()
        enc = _LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        head = _LSTMPretrainHead(enc, out_classes=num_classes, dropout=dropout)
        if use_parallel:
            head = nn.DataParallel(head)

        self.model = head.to(device)
        self.device = device
        self.use_parallel = use_parallel

        self._ml: Any = None
        self._meta: Dict[str, Any] = dict(
            ml_kind=self.ML_KIND,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
            random_state=random_state,
        )
        # feature knobs
        self._meta.update(
            {
                "use_pooling": True,
                "pool_last_k": 16,
                "add_LSTM_proba": True,
            }
        )

    # ----------------- training -----------------
    def train(
        self,
        train_data,
        val_data,
        *,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 8,
        num_workers: int = 0,
        class_weight_mode: str = "balanced",  # kept for interface; not used directly here
        model_path: Union[str, Path] = "artifacts/hybrid_LSTM_ml",
        loss_plot_path: Optional[Union[str, Path]] = None,
        use_gpu_in_ml: bool = True,
        weights=None,
        # ---- feature knobs ----
        use_pooling: bool = True,
        pool_last_k: int = 16,
        add_LSTM_proba: bool = True,
        # ---- early stopping by metric ----
        early_metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        early_metric_mode: str = "min",   # "min" for Brier, "max" e.g. for F1
    ) -> dict:
        # expects set_deterministic to exist (same as in your GRU hybrid)
        set_deterministic(self.random_state)

        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        if loss_plot_path is None:
            loss_plot_path = model_dir / "loss_curve.png"

        # persist meta for inference
        self._meta["use_pooling"] = bool(use_pooling)
        self._meta["pool_last_k"] = int(pool_last_k)
        self._meta["add_LSTM_proba"] = bool(add_LSTM_proba)

        if early_metric_fn is not None:
            assert early_metric_mode in {"min", "max"}, "early_metric_mode must be 'min' or 'max'"

        Ltr = _to_loader(train_data, batch_size, shuffle=True, num_workers=num_workers)
        Lva = _to_loader(val_data, batch_size, shuffle=False, num_workers=num_workers)

        # class weights for LSTM pretrain (if provided)
        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            w = None

        crit = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_metric: Optional[float] = None
        best_state = None
        bad = 0
        tr_losses, va_losses = [], []

        for ep in range(1, epochs + 1):
            # ----------------- train epoch -----------------
            self.model.train()
            tr_loss, n = 0.0, 0
            for Xb, yb in tqdm(Ltr, desc=f"LSTM pretrain {ep}/{epochs}", leave=False):
                Xb = Xb.to(self.device)
                yb = yb.view(-1).long().to(self.device)

                opt.zero_grad()
                logits = self.model(Xb)  # [B, num_classes]
                loss = crit(logits, yb)
                loss.backward()
                opt.step()

                bs = yb.size(0)
                tr_loss += loss.item() * bs
                n += bs
            tr_losses.append(tr_loss / max(n, 1))

            # ----------------- validation epoch -----------------
            self.model.eval()
            va_loss, n = 0.0, 0
            all_probs = []
            all_true = []

            with torch.no_grad():
                for Xb, yb in Lva:
                    Xb = Xb.to(self.device)
                    yb = yb.view(-1).long().to(self.device)

                    logits = self.model(Xb)
                    loss = crit(logits, yb)

                    bs = yb.size(0)
                    va_loss += loss.item() * bs
                    n += bs

                    if early_metric_fn is not None:
                        probs = torch.softmax(logits, dim=1)
                        all_probs.append(probs.cpu().numpy())
                        all_true.append(yb.cpu().numpy())

            va_loss = va_loss / max(n, 1)
            va_losses.append(va_loss)

            # --- compute early-stop metric (if requested) ---
            if early_metric_fn is not None:
                if all_probs:
                    y_true = np.concatenate(all_true)
                    probs_np = np.vstack(all_probs)
                    current_metric = float(early_metric_fn(y_true, probs_np))
                else:
                    current_metric = float("inf") if early_metric_mode == "min" else float("-inf")

                if best_metric is None:
                    improved = True
                else:
                    if early_metric_mode == "min":
                        improved = current_metric < best_metric - 1e-6
                    else:
                        improved = current_metric > best_metric + 1e-6

                if improved:
                    best_metric = current_metric
                    best_state = self._state_dict()
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            else:
                # fallback: early stop on val loss
                if va_loss < best_val_loss - 1e-6:
                    best_val_loss = va_loss
                    best_state = self._state_dict()
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

        # restore best encoder weights
        if best_state is not None:
            self._load_state_dict(best_state)

        # plot loss curves
        self._plot_losses(tr_losses, va_losses, loss_plot_path)

        # ----------------- feature extraction for ML head -----------------
        Ztr, ytr = self._extract_features(Ltr)
        Zva, yva = self._extract_features(Lva)

        # ML head (sklearn API)
        self._ml = self._fit_ml(Ztr, ytr, Zva, yva, use_gpu_in_ml)

        self.save(model_dir)

        result = {
            "best_val_loss": best_val_loss,
            "epochs_trained": len(tr_losses),
            "loss_plot_path": str(loss_plot_path),
            "model_dir": str(model_dir),
        }
        if early_metric_fn is not None:
            result["best_early_metric"] = best_metric
        return result

    # ----------------- inference -----------------
    def predict(self, data, *, batch_size: int = 1024, num_workers: int = 0) -> pd.DataFrame:
        assert self._ml is not None, "ML head not trained/loaded."
        L = _to_loader(data, batch_size, shuffle=False, num_workers=num_workers)
        Z, y = self._extract_features(L)
        proba = self._predict_proba_ml(Z)
        yhat = proba.argmax(axis=1)

        cols = [f"p{i}" for i in range(proba.shape[1])]
        df = pd.DataFrame(proba, columns=cols)
        df.insert(0, "pred", yhat)
        df["true"] = y.astype(int)
        return df
    
    # ----------------- save / load -----------------
    def save(self, model_dir: Union[str, Path]) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._state_dict(), model_dir / "encoder.pt")
        joblib.dump(self._ml, model_dir / "ml_model.pkl")
        with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "_HybridLSTMML":
        model_dir = Path(model_dir)
        with open(model_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        obj = cls(
            input_size=meta["input_size"],
            hidden_size=meta["hidden_size"],
            num_layers=meta["num_layers"],
            num_classes=meta["num_classes"],
            dropout=meta["dropout"],
            bidirectional=meta["bidirectional"],
            random_state=meta.get("random_state", 42),
        )
        state = torch.load(model_dir / "encoder.pt", map_location=obj.device)
        obj._load_state_dict(state)
        obj._ml = joblib.load(model_dir / "ml_model.pkl")
        obj._meta = meta
        return obj

    # ----------------- internals -----------------
    def _state_dict(self) -> dict:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def _load_state_dict(self, sd: dict) -> None:
        target = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        target.load_state_dict(sd, strict=True)

    def _plot_losses(self, tr: Sequence[float], va: Sequence[float], path: Union[str, Path]) -> None:
        path = _ensure_dir(path)
        plt.figure(figsize=(6, 4))
        e = range(1, len(tr) + 1)
        plt.plot(e, tr, label="train", marker="o")
        plt.plot(e, va, label="val", marker="s")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _extract_features(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (X_ml, y), where X_ml is the concatenation of:
          - last hidden state,
          - optional mean/max pooling over last k steps of LSTM outputs,
          - optional LSTM softmax probabilities from the pretrain head.
        """
        self.model.eval()
        use_pool = bool(self._meta.get("use_pooling", True))
        k = int(self._meta.get("pool_last_k", 16))
        add_prob = bool(self._meta.get("add_LSTM_proba", True))

        # unwrap modules
        if isinstance(self.model, nn.DataParallel):
            head = self.model.module
        else:
            head = self.model
        enc = head.encoder

        Z_list, Y_list = [], []
        with torch.no_grad():
            LSTM = enc.LSTM  # nn.LSTM
            bi = 2 if LSTM.bidirectional else 1

            for Xb, yb in loader:
                Xb = Xb.to(self.device)

                # 1) last hidden state (original feature)
                out, (h_n, c_n) = LSTM(Xb)     # out: [B,T,bi*H] (batch_first=True)
                if bi == 2:
                    z_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, 2H]
                else:
                    z_last = h_n[-1]                               # [B, H]

                feats = [z_last]

                # 2) temporal pooling over top-layer outputs (mean & max over last k steps)
                if use_pool and out.size(1) >= 1:
                    kk = min(k, out.size(1))
                    tail = out[:, -kk:, :]          # [B, kk, bi*H]
                    mean_pool = tail.mean(dim=1)    # [B, bi*H]
                    max_pool, _ = tail.max(dim=1)   # [B, bi*H]
                    feats += [mean_pool, max_pool]

                # 3) LSTM softmax probabilities from pretrain head
                if add_prob:
                    logits = head(Xb)                       # uses encoder + linear
                    probs = torch.softmax(logits, dim=-1)   # [B, C]
                    feats += [probs]

                Zb = torch.cat(feats, dim=1).cpu().numpy()
                Z_list.append(Zb)
                Y_list.append(yb.view(-1).cpu().numpy())

        Z = np.vstack(Z_list)
        Y = np.concatenate(Y_list)
        return Z, Y

    # hooks to override in subclasses
    def _fit_ml(self, Ztr, ytr, Zva, yva, use_gpu: bool):
        raise NotImplementedError

    def _predict_proba_ml(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ------------------------- sklearn ML heads -------------------------
class LSTMCatBoostClassifier(_HybridLSTMML):
    ML_KIND = "catboost"

    def _fit_ml(self, Ztr, ytr, Zva, yva, use_gpu: bool):
        model = CatBoostClassifier(
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            learning_rate=0.05,
            depth=6,
            n_estimators=2000,
            random_state=self.random_state,
            task_type="GPU" if (use_gpu and torch.cuda.is_available()) else "CPU",
            od_type="Iter",
            od_wait=100,
            verbose=False,
        )
        model.fit(Ztr, ytr, eval_set=(Zva, yva), verbose=False, use_best_model=True)
        return model

    def _predict_proba_ml(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(self._ml.predict_proba(Z))


class LSTMLightGBMClassifier(_HybridLSTMML):
    ML_KIND = "lightgbm"

    def _fit_ml(self, Ztr, ytr, Zva, yva, use_gpu: bool):
        model = LGBMClassifier(
            objective="multiclass",
            num_class=self.num_classes,
            learning_rate=0.05,
            num_leaves=63,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=1,
            n_estimators=4000,
            random_state=self.random_state,
            device="gpu" if (use_gpu and torch.cuda.is_available()) else "cpu",
        )
        model.fit(
            Ztr,
            ytr,
            eval_set=[(Zva, yva)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
        return model

    def _predict_proba_ml(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(self._ml.predict_proba(Z))


class LSTMXGBClassifier(_HybridLSTMML):
    ML_KIND = "xgboost"

    def _fit_ml(self, Ztr, ytr, Zva, yva, use_gpu: bool):
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=self.num_classes,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            n_estimators=4000,
            eval_metric="mlogloss",
            tree_method="gpu_hist" if (use_gpu and torch.cuda.is_available()) else "hist",
            random_state=self.random_state,
            n_jobs=0,
        )
        model.fit(
            Ztr,
            ytr,
            eval_set=[(Zva, yva)],
            early_stopping_rounds=200,
            verbose=False,
        )
        return model

    def _predict_proba_ml(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(self._ml.predict_proba(Z))
def f1_metric(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_pred = (probs[:, 1] >= 0.5).astype(int)
    # print('f1 es used')
    return f1_score(y_true, y_pred, average="macro", zero_division=0)def brier_metric(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Brier score for binary classification.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels in {0, 1}.
    probs : array-like of shape (n_samples, 2)
        Predicted class probabilities, probs[:, 1] is P(y=1).

    Returns
    -------
    float
        Brier score (lower is better).
    """
    y_true = y_true.astype(float)
    p_hat = probs[:, 1].astype(float)  # prob of class 1
    # print('brier es used')
    return float(np.mean((p_hat - y_true) ** 2))def bss_metric(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Brier Skill Score for binary TP vs not-TP.

    Compatible with early stopping:
      - y_true: shape (N,), {0,1}
      - probs:  shape (N,) or (N,2)
        * if (N,2), we take column 1 as P(TP)
    """
    y = np.asarray(y_true, dtype=float).ravel()

    p = np.asarray(probs, dtype=float)
    if p.ndim == 2:
        # Expect binary probabilities (N,2) ⇒ use P(class=1)
        if p.shape[1] != 2:
            raise ValueError(f"bss_metric: expected probs shape (N,2) for binary, got {p.shape}")
        p = p[:, 1]
    elif p.ndim == 1:
        # Already P(TP)
        pass
    else:
        raise ValueError(f"bss_metric: probs must be 1D or 2D, got ndim={p.ndim}")

    p = p.ravel()
    if p.shape[0] != y.shape[0]:
        raise ValueError(f"bss_metric: shape mismatch y={y.shape}, p={p.shape}")

    # Clip to [0,1] just in case
    p = np.clip(p, 0.0, 1.0)

    # Model Brier score
    bs_model = np.mean((p - y) ** 2)

    # Climatology baseline: always predict pi = mean(y)
    pi = y.mean()
    bs_clim = np.mean((pi - y) ** 2)

    if bs_clim <= 0:
        # Degenerate case: all y the same
        return 0.0

    # Brier Skill Score
    return 1.0 - bs_model / bs_clim#TRIPLE BARRIER METRICS

#2-CLASSES

def triple_barrier_metrics(
    *,
    y_true,
    y_pred,
    p_all,
    ku: float,
    kd: float,
) -> dict:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score, f1_score

    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    if isinstance(p_all, pd.DataFrame):
        if p_all.shape[1] != 1:
            raise ValueError("For binary metrics, p_all must have exactly one column = P(TP).")
        p_tp = p_all.iloc[:, 0].to_numpy(dtype=float).ravel()
    else:
        p_tp = np.asarray(p_all, dtype=float).ravel()

    if not (len(y_true) == len(y_pred) == len(p_tp)):
        raise ValueError("y_true, y_pred and p_tp must have the same length.")

    tp_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    tp_recall    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    tp_f1        = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)

    brier = np.mean((p_tp - y_true) ** 2)
    bss   = bss_metric(y_true, p_tp)

    return {
        "tp_precision": tp_precision,
        "tp_recall": tp_recall,
        "tp_f1": tp_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "brier": brier,
        "bss": bss,
    }

#======================================================================================================
#3-class
#=====================================================================================================
# def triple_barrier_metrics(
#     y_true,
#     y_pred,
#     p_all,
#     ku: float,
#     kd: float,
# ):
#     """
#     y_true: array-like (n_samples,), true labels in {0,1,2}
#             0 = expiry, 1 = SL, 2 = TP
#     y_pred: array-like (n_samples,), predicted labels in {0,1,2}
#     p_all:  array-like (n_samples, 3), predicted probs [p0, p1, p2]
#     ku:     TP reward in R units
#     kd:     SL loss in R units

#     Returns dict with:
#       - tp_precision, tp_recall, tp_f1
#       - macro_precision_tp_sl, macro_recall_tp_sl, macro_f1_tp_sl
#       - macro_f1  (macro F1 over all 3 classes)
#       - brier_bin, brier_multi, EU
#     """

#     y_true = np.asarray(y_true, dtype=int)
#     y_pred = np.asarray(y_pred, dtype=int)
#     p2 = np.asarray(p_all['p2'], dtype=float)
#     p_all = np.asarray(p_all, dtype=float)

#     if y_true.shape != y_pred.shape:
#         raise ValueError("y_true and y_pred must have the same shape.")
#     if p2.shape[0] != y_true.shape[0]:
#         raise ValueError("p2 must have length equal to y_true.")
#     if p_all.shape[0] != y_true.shape[0] or p_all.shape[1] != 3:
#         raise ValueError("p_all must have shape (n_samples, 3).")

#     def prf_for_class(k: int):
#         tp = np.sum((y_true == k) & (y_pred == k))
#         fp = np.sum((y_true != k) & (y_pred == k))
#         fn = np.sum((y_true == k) & (y_pred != k))

#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         if (precision + recall) > 0:
#             f1 = 2 * precision * recall / (precision + recall)
#         else:
#             f1 = 0.0
#         return precision, recall, f1

#     # TP (class 2) metrics
#     tp_precision, tp_recall, tp_f1 = prf_for_class(2)

#     # SL (class 1) metrics for macro over {SL, TP}
#     sl_precision, sl_recall, sl_f1 = prf_for_class(1)

#     macro_precision_tp_sl = (tp_precision + sl_precision) / 2.0
#     macro_recall_tp_sl    = (tp_recall + sl_recall) / 2.0
#     macro_f1_tp_sl        = (tp_f1 + sl_f1) / 2.0

#     # Macro F1 over all three classes {0,1,2}
#     f1_0 = prf_for_class(0)[2]
#     f1_1 = sl_f1
#     f1_2 = tp_f1
#     macro_f1 = (f1_0 + f1_1 + f1_2) / 3.0

#     # Brier scores and expected utility
#     brier_bin = brier_binary(y_true, p2)
#     brier_multi = brier_multiclass(y_true, p_all)
#     EU, _ = expected_utility_zero_threshold_multibarrier(
#         y_true=y_true,
#         P_hat=p_all,
#         tp_R=ku,
#         sl_R=kd,
#     )

#     return {
#         "tp_precision": tp_precision,
#         "tp_recall": tp_recall,
#         "tp_f1": tp_f1,
#         "macro_precision_tp_sl": macro_precision_tp_sl,
#         "macro_recall_tp_sl": macro_recall_tp_sl,
#         "macro_f1_tp_sl": macro_f1_tp_sl,
#         "macro_f1": macro_f1,
#         "brier_bin": brier_bin,
#         "brier_multi": brier_multi,
#         "EU": EU,
#     }
#Trade module


TRADE_COLUMNS = [
    "open_date",
    "close_date",
    "side",
    "position_size",
    "pnl",
    "return_pct",
    "total_equity",
    "entry_price",
    "exit_price",
    "exit_reason",
    "min_point_atr"
]

def ulcer_index(eq: pd.Series) -> float:
    eq = eq.astype(float)
    rollmax = eq.cummax()
    dd = 1.0 - (eq / rollmax).clip(upper=1.0)
    return float(np.sqrt(np.mean(np.square(dd.values)))) if len(dd) else 0.0

def martin_ratio(eq: pd.Series, c, mar: float = 0.20) -> float:
    ui = ulcer_index(eq)
    if ui == 0.0:
        return math.inf if c > mar else -math.inf
    return (c - mar) / ui

@dataclass
class TradeRecord:
    """Container for a single completed trade."""

    open_date: object
    close_date: object
    side: str
    pnl: float
    return_pct: float
    total_equity: float
    entry_price: float
    exit_price: float
    exit_reason: str
    position_size: float = 1.0
    min_point_atr: float = np.nan


@dataclass
class TripleBarrierConfig:
    """Configuration parameters for the triple-barrier strategy."""

    tp_distance: float
    sl_distance: float
    holding_period: int
    min_return: float = 0.0


@dataclass
class PredictedRangeConfig:
    """Configuration parameters for the predicted range strategy."""

    holding_period: int
    min_rr: float


class TradingStrategy:
    """Long-only strategy supporting multiple signal generation modes."""

    def __init__(
        self,
        mode: str = "triple_barrier",
        periods_per_year: int = 365 * 24 * 4,
        transaction_cost: bool = True,
        slippage: float = 0.0,
        triple_barrier_config: TripleBarrierConfig | None = None,
        predicted_range_config: PredictedRangeConfig | None = None,
        use_limit_entries: bool = False,
        limit_offset: float = 0.0,  # fraction of distance from entry to SL (0–1)
        taker_fee: float = 0.0005,          # per-side taker fee (fraction of notional)
        maker_fee: float = 0.0002,
        # per-side maker fee (fraction of notional)
    ) -> None:
        if mode not in {"triple_barrier", "predicted_range"}:
            raise ValueError("mode must be 'triple_barrier' or 'predicted_range'")

        self.mode = mode
        self.periods_per_year = periods_per_year
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        if self.transaction_cost:
            self.taker_fee = float(taker_fee)
            self.maker_fee = float(maker_fee)
        else:
            self.taker_fee = 0.0
            self.maker_fee = 0.0

        self.use_limit_entries = use_limit_entries
        self.limit_offset = float(limit_offset)

        if self.use_limit_entries:
            if not (0.0 <= self.limit_offset < 1.0):
                raise ValueError(
                    "limit_offset must be in (0, 1) when use_limit_entries is True"
                )

        if mode == "triple_barrier":
            if triple_barrier_config is None:
                raise ValueError(
                    "triple_barrier_config must be provided for triple_barrier mode"
                )
            if triple_barrier_config.holding_period <= 0:
                raise ValueError(
                    "holding_period must be positive for triple_barrier strategy"
                )
            self.triple_barrier_config = triple_barrier_config
        else:
            if predicted_range_config is None:
                raise ValueError(
                    "predicted_range_config must be provided for predicted_range mode"
                )
            if predicted_range_config.holding_period <= 0:
                raise ValueError(
                    "holding_period must be positive for predicted_range strategy"
                )
            if predicted_range_config.min_rr <= 0:
                raise ValueError("min_rr must be positive for predicted_range strategy")
            self.predicted_range_config = predicted_range_config

        self.signals: np.ndarray = np.array([])
        self.positions: List[int] = []
        self.returns: List[float] = []
        self.equity_curve: List[float] = []
        self.trade_log: pd.DataFrame = pd.DataFrame(columns=TRADE_COLUMNS)


    def reset(self) -> None:
        """Reset tracked performance series."""

        self.positions = []
        self.returns = []
        self.equity_curve = []
        self.trade_log = pd.DataFrame(columns=TRADE_COLUMNS)

    def _commission_path(self, entry_type: str, exit_reason: str) -> float:
        """
        Return total commission fraction for a given entry / exit path.

        Assumptions:
        - market entry: taker
        - limit entry: maker
        - take_profit: exit via resting limit (maker)
        - stop_loss / expiry / market_close: exit via market (taker)
        """
        fees = 0.0

        if entry_type == "market":
            fees += self.taker_fee
        elif entry_type == "limit":
            fees += self.maker_fee
        else:
            raise ValueError(f"Unknown entry_type: {entry_type}")

        if exit_reason == "take_profit":
            fees += self.maker_fee
        else:  # "stop_loss", "expiry", "market_close"
            fees += self.taker_fee

        return fees

    
    def generate_signals(
        self,
        predictions: Sequence[float] | pd.DataFrame,
        prices: pd.DataFrame,
        probability_column: str = "p2",
        atr_column = 'atr_14',
        log_up_column: str = "y_up",
        log_down_column: str = "y_down",
    ) -> dict:
        """Generate entry signals and barrier levels for the configured mode."""

        if predictions.shape[0] != prices.shape[0]:
            raise ValueError(
                "Predictions and prices must have the same length.")

        if self.mode == "triple_barrier":
            if isinstance(probability_column, list):
                probability_column = probability_column[0]
            probs = predictions[probability_column].to_numpy(dtype=float)
            config = self.triple_barrier_config
            
            close_prices = prices["close"].to_numpy(dtype=float)
            atrs        = prices[atr_column].to_numpy(dtype=float)
            atr_pct     = atrs / close_prices
            
            # Price moves in return space (fractions)
            tp_move = config.tp_distance * atr_pct
            sl_move = config.sl_distance * atr_pct
            
            # Commissions per path
            fee_tp = self._commission_path(entry_type="market", exit_reason="take_profit")
            fee_sl = self._commission_path(entry_type="market", exit_reason="stop_loss")
            
            # slippage is still aggregate per-trade cost
            cost_tp = self.slippage + fee_tp
            cost_sl = self.slippage + fee_sl
            
            reward = tp_move - cost_tp      # net up-move if TP hits
            risk   = sl_move + cost_sl      # net loss if SL hits
            
            # ---- Breakeven probability for each bar ----
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = reward + risk
                # If denom <= 0 (pathologically bad trade), set p_BE = 1 (never enter)
                p_be = np.where(denom > 0, risk / denom, 1.0)
                p_be = np.clip(p_be, 0.0, 1.0)
            
            # ---- Probability edge over breakeven ----
            edge = probs - p_be
            
            # `config.min_return` now interpreted as "min probability edge"
            # e.g. 0.02 = require p_TP at least 2 pp above breakeven p_BE
            entries = (edge >= config.min_return).astype(int)
            
            # Optional: avoid NaNs opening trades
            entries[np.isnan(edge)] = 0




        # elif self.mode == "predicted_range":
        #     close_prices = prices['close'].to_numpy(dtype=float)
        #     log_up = predictions[log_up_column].to_numpy(dtype=float)
        #     log_down = predictions[log_down_column].to_numpy(dtype=float)
        #     tp_prices = close_prices * np.exp(log_up)
        #     sl_prices = close_prices * np.exp(-log_down)
        #     ranges = tp_prices - sl_prices
        #     sl_prices = sl_prices - ranges * 0.25

        #     rr = (tp_prices - close_prices) / (close_prices - sl_prices + 1e-8)
        #     entries = (
        #         rr >= self.predicted_range_config.min_rr + self.slippage + self.transaction_cost
        #     ).astype(int)

        return entries

    def backtest(
        self,
        predictions: Sequence[float] | pd.DataFrame,
        prices: Sequence[float] | pd.DataFrame,
        timestamps: Sequence[object] | None = None,
        probability_column: str = "TP_prob",
        log_up_column: str = "y_up",
        log_down_column: str = "y_down",
        atr_column: str = "atr_14",
        equity: float = 1000.0,
        position_size: float = 1.0,
        risk_mode: str = "fixed_size",
        compound: bool = False,
    ) -> dict:
        """Run a backtest using pre-generated prediction outputs."""

        self.initial_equity = equity
        self.compound = compound

        if risk_mode not in {"fixed_size", "fixed_risk"}:
            raise ValueError("risk_mode must be 'fixed_size' or 'fixed_risk'")

        close_prices = prices["close"].to_numpy(dtype=float)
        high_prices = prices["high"].to_numpy(dtype=float)
        low_prices = prices["low"].to_numpy(dtype=float)
        atr = prices[atr_column].to_numpy(dtype=float)

        if close_prices.size == 0:
            raise ValueError("Price data cannot be empty.")

        signals = self.generate_signals(
            predictions,
            prices,
            probability_column=probability_column,
            log_up_column=log_up_column,
            log_down_column=log_down_column,
            atr_column=atr_column
        )

        self.signals = signals

        if self.mode == "predicted_range":
            close_arr = close_prices
            log_up = predictions[log_up_column].to_numpy(dtype=float)
            log_down = predictions[log_down_column].to_numpy(dtype=float)
            tp_prices = close_arr * np.exp(log_up)
            sl_prices = close_arr * np.exp(-log_down)
            ranges = tp_prices - sl_prices
            sl_prices = sl_prices - ranges * 0.25
        elif self.mode == "triple_barrier":
            tp_prices = close_prices + self.triple_barrier_config.tp_distance * atr
            sl_prices = close_prices - self.triple_barrier_config.sl_distance * atr
        else:
            raise ValueError("Unsupported mode")

        if timestamps is None:
            timestamps_arr = np.arange(len(close_prices))
        else:
            timestamps_arr = np.asarray(timestamps)
            if timestamps_arr.shape[0] != close_prices.shape[0]:
                raise ValueError("Timestamps must align with predictions/prices.")

        self.reset()
        trade_records: List[TradeRecord] = []
        current_trade: dict | None = None
        pending_order: dict | None = None

        holding_period = (
            self.triple_barrier_config.holding_period
            if self.mode == "triple_barrier"
            else self.predicted_range_config.holding_period
        )

        exit_price: float | None = None
        exit_reason: str | None = None

        for i in range(close_prices.shape[0]):
            timestamp = timestamps_arr[i]
            prev_equity = equity

            # --- 1. Manage open trade exits ---
            if current_trade is not None and i > current_trade["entry_index"]:
                low_i = float(low_prices[i])
                high_i = float(high_prices[i])

                stop_hit = low_i <= current_trade["stop_price"]
                tp_hit = high_i >= current_trade["target_price"]
                expiry_hit = i >= current_trade["expiry_index"]
                last_bar = i == close_prices.shape[0] - 1

                if stop_hit:
                    exit_price = current_trade["stop_price"]
                    exit_reason = "stop_loss"
                elif tp_hit:
                    exit_price = current_trade["target_price"]
                    exit_reason = "take_profit"
                elif expiry_hit or last_bar:
                    exit_price = close_prices[i]
                    exit_reason = "expiry" if expiry_hit else "market_close"
                else:
                    # Only update running minimum when NO exit occurs on this bar
                    current_trade["min_low"] = min(current_trade["min_low"], low_i)

                if exit_reason is not None and exit_price is not None:
                    equity, record = self._finalize_trade(
                        current_trade,
                        exit_price,
                        timestamp,
                        exit_reason,
                        equity,
                        current_trade["position_size"],
                    )
                    trade_records.append(record)
                    current_trade = None
                    exit_price = None
                    exit_reason = None

            # --- 2. Manage pending limit order (no open trade) ---
            if current_trade is None and pending_order is not None:
                # Cancel if order expired
                if i > pending_order["order_expiry_index"]:
                    pending_order = None
                else:
                    # Check fill condition from the signal onwards (from bar after signal)
                    low_i = float(low_prices[i])
                    if low_i < pending_order["limit_price"]:
                        entry_price = pending_order["limit_price"]
                        target_price = pending_order["target_price"]
                        stop_price = pending_order["stop_price"]
                        # expiry index stays at initial (signal) value
                        expiry_index = pending_order["order_expiry_index"]

                        if risk_mode == "fixed_size":
                            cur_position_size = position_size
                        else:  # fixed_risk
                            cur_position_size = position_size / (
                                abs(stop_price / entry_price - 1.0)
                                + self.slippage
                                + self.taker_fee
                                + self.maker_fee# conservative: assume taker on SL
                            )

                        current_trade = {
                            "open_date": timestamp,
                            "entry_price": entry_price,
                            "target_price": target_price,
                            "stop_price": stop_price,
                            "expiry_index": expiry_index,
                            "entry_index": i,
                            "entry_equity": equity,
                            "position_size": cur_position_size,
                            "entry_atr": float(atr[i]),
                            "min_low": entry_price,
                            "entry_type": "limit",   # <--- NEW
                        }
                        pending_order = None


            # --- 3. Create new order at signal bar (if flat and no pending) ---
            if (
                current_trade is None
                and pending_order is None
                and self.signals[i] == 1
                and i < close_prices.shape[0] - 1
                and close_prices[i] < close_prices[i-1]
                and self.signals[i-1] == 1
                and self.signals[i-2] == 1
            ):
                signal_price = close_prices[i]
                target_price = float(tp_prices[i])
                stop_price = float(sl_prices[i])

                if self.use_limit_entries:
                    # Distance from market entry to SL
                    distance = signal_price - stop_price
                    if distance <= 0:
                        # Degenerate case; fall back to market entry
                        limit_price = signal_price
                    else:
                        limit_price = signal_price - self.limit_offset * distance
                        # Keep strictly above SL
                        if limit_price <= stop_price:
                            limit_price = stop_price * (1.0 + 1e-8)

                    pending_order = {
                        "signal_index": i,
                        "limit_price": float(limit_price),
                        "target_price": target_price,
                        "stop_price": stop_price,
                        # Order validity: until signal_bar + holding_period
                        "order_expiry_index": i + holding_period,
                    }
                else:
                    entry_price = signal_price
                    expiry_index = i + holding_period

                    if risk_mode == "fixed_size":
                        cur_position_size = position_size
                    else:  # fixed_risk
                        cur_position_size = position_size / (
                            abs(stop_price / entry_price - 1.0)
                            + self.slippage
                            + 2*self.taker_fee
                        )


                    current_trade = {
                        "open_date": timestamp,
                        "entry_price": entry_price,
                        "target_price": target_price,
                        "stop_price": stop_price,
                        "expiry_index": expiry_index,
                        "entry_index": i,
                        "entry_equity": equity,
                        "position_size": cur_position_size,
                        "entry_atr": float(atr[i]),
                        "min_low": entry_price,
                        "entry_type": "market",  # <--- NEW
                    }


            # --- 4. Book step returns / equity path ---
            step_return = (equity - prev_equity) / prev_equity if prev_equity else 0.0
            self.returns.append(step_return)
            self.equity_curve.append(equity)
            self.positions.append(1 if current_trade is not None else 0)

        # --- finalize trade log + metrics as before ---
        if trade_records:
            self.trade_log = pd.DataFrame(
                [record.__dict__ for record in trade_records], columns=TRADE_COLUMNS
            )
        else:
            self.trade_log = pd.DataFrame(columns=TRADE_COLUMNS)

        return self.calculate_metrics()


    def _finalize_trade(
        self,
        trade: dict,
        exit_price: float,
        close_timestamp: object,
        exit_reason: str,
        equity: float,
        position_size: float = 1.0,
    ) -> tuple[float, TradeRecord]:
        """Close an active trade and return the updated equity and trade record."""

        entry_equity = trade["entry_equity"]
        entry_price = trade["entry_price"]

        entry_atr = trade.get("entry_atr", np.nan)
        min_low = trade.get("min_low", trade["entry_price"])
        min_point_atr = ((trade["entry_price"] - min_low) / entry_atr) if (entry_atr and entry_atr > 0) else np.nan 

        entry_type = trade.get("entry_type", "market")
        fees = self._commission_path(entry_type=entry_type, exit_reason=exit_reason)
        total_cost = self.slippage + fees

        gross_return = (exit_price / entry_price - 1.0 - total_cost) * position_size

        
        if self.compound:
            equity *= (1 + gross_return)
            position_size = position_size*equity
        else:
            equity += self.initial_equity * gross_return
            position_size = position_size*self.initial_equity

        pnl = equity - entry_equity
        return_pct = (equity / entry_equity - 1) * 100 if entry_equity else 0.0

        record = TradeRecord(
            open_date=trade["open_date"],
            close_date=close_timestamp,
            side="long",
            pnl=pnl,
            return_pct=return_pct,
            total_equity=equity,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason,
            position_size=position_size,
            min_point_atr=min_point_atr,
        )

        return equity, record

    def calculate_metrics(self) -> dict:
        """Calculate and return backtest performance metrics."""
        
        
        equity_array = np.asarray(self.equity_curve, dtype=float)
        returns_array = np.asarray(self.returns, dtype=float)
        
        # ----- Sharpe & volatility -----
        if returns_array.size > 1:
            returns_std = returns_array.std(ddof=1)
        elif returns_array.size == 1:
            returns_std = returns_array.std(ddof=0)
        else:
            returns_std = 0.0
        
        if returns_std > 0 and returns_array.size > 0:
            sharpe = np.sqrt(self.periods_per_year) * returns_array.mean() / returns_std
        else:
            sharpe = 0.0
        
        volatility = returns_std * np.sqrt(self.periods_per_year) if returns_std > 0 else 0.0
        
        # ----- Clean equity for all DD / CAGR-related metrics -----
        E = equity_array[~np.isnan(equity_array)]
        
        if E.size >= 2 and E[0] > 0:
            # Total return (percent)
            total_return = (E[-1] / E[0] - 1.0) * 100.0
        
            # Drawdown series and max drawdown (negative)
            run_max = np.maximum.accumulate(E)
            dd = E / run_max - 1.0            # <= 0
            mdd = float(dd.min())             # most negative drawdown
        
            # CAGR (annualized, assuming equally spaced periods)
            cagr = (E[-1] / E[0]) ** (self.periods_per_year / E.size) - 1.0
        
            # Calmar
            calmar = cagr / abs(mdd) if mdd < 0 else np.nan
        
            # Martin ratio
            martin = martin_ratio(pd.Series(E), cagr)
        else:
            total_return = 0.0
            mdd = 0.0
            cagr = 0.0
            calmar = np.nan
            martin = np.nan
        
        # For backwards compatibility with your existing outputs:
        max_drawdown = mdd
        
        # ----- Winrate and trade count -----
        if self.trade_log.shape[0] > 0:
            wins = (self.trade_log['pnl'] > 0).sum()
            winrate = wins / self.trade_log.shape[0]
        else:
            winrate = 0.0
        
        metrics = {
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,      # negative fraction (e.g. -0.7)
            "total_return": total_return,      # percent
            "volatility": volatility,          # annualized
            "final_equity": float(E[-1]) if E.size else 0.0,
            "winrate": winrate,
            "calmar_ratio": calmar,
            "martin_ratio": martin,
            "num_trades": int(self.trade_log.shape[0]),
        }
        
        return metrics

    
def standard_trade_test(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    ku: float,
    kd: float,
    hold: int,
    *,
    probability_column: str = "p2",
    atr_column: str = "atr_14",
    equity: float = 10000.0,
    position_size: float = 0.01,
    risk_mode: str = "fixed_risk",
    compound: bool = True,
    transaction_cost: bool = True,
    slippage: float = 0,
    min_return: float = 0.0,
    use_limit: bool = False,
    limit_offset: float = 0.0,
    tf: str = '15m',
    
):
    """
    Run a triple-barrier backtest given predictions and prices (no calibration).
    Returns:
        metrics: dict
        tlog:    trade log DataFrame
        equity_info: dict with 'equity' and 'positions' arrays
    """

    periods = periods_per_year(tf)
    
    config = TripleBarrierConfig(
        tp_distance=ku,
        sl_distance=kd,
        holding_period=hold,
        min_return=min_return,
    )

    strategy = TradingStrategy(
        mode="triple_barrier",
        periods_per_year=periods,
        transaction_cost=transaction_cost,
        slippage=slippage,
        triple_barrier_config=config,
        use_limit_entries=use_limit,
        limit_offset=limit_offset,
    )

    metrics = strategy.backtest(
        predictions=predictions,
        prices=prices,
        timestamps=None,
        probability_column=probability_column,
        log_up_column="y_up",
        log_down_column="y_down",
        atr_column=atr_column,
        equity=equity,
        position_size=position_size,
        risk_mode=risk_mode,
        compound=compound,
    )

    tlog = strategy.trade_log.copy()
    equity_info = {
        "equity": np.asarray(strategy.equity_curve, dtype=float),
        "positions": np.asarray(strategy.positions, dtype=int),
    }

    return metrics, tlog, equity_info
def sweep_min_return(
    prices: pd.DataFrame,
    df_pred: pd.DataFrame,
    ku: float,
    kd: float,
    hold: int,
    min_grid,
    artifacts_dir,
    *,
    slippage: float = 0.002,
    transaction_cost: bool = True,
    compound: bool = True,
    use_limit: bool = False,
    limit_offset: float = 0.0,
    min_trades: int = 0,
    volatility_col = 'atr_14',
    probability_col=['p2'],
    tf = '15m',
    risk_mode = 'fixed_risk',
    position_size=0.01
    
):
    """
    Sweep over min_return values and pick best by total_return and Martin.
    If no mr satisfies min_trades, best_*['mr'] and ['metrics'] are None.
    """

    best_return = {"score": -np.inf, "mr": None, "metrics": None}
    best_martin = {"score": -np.inf, "mr": None, "metrics": None}

    for mr in min_grid:
        metrics, tlog, equity_info = standard_trade_test(
            predictions=df_pred,
            prices=prices,
            ku=ku,
            kd=kd,
            hold=hold,
            probability_column=probability_col,
            atr_column= volatility_col,
            equity=10000.0,
            position_size=position_size,
            risk_mode=risk_mode,
            compound=compound,
            transaction_cost=transaction_cost,
            slippage=slippage,
            min_return=mr,
            use_limit=use_limit,
            limit_offset=limit_offset,
            tf=tf
        )

        n_trades = metrics.get("num_trades", len(tlog))
        if n_trades < min_trades:
            continue

        if metrics["total_return"] > best_return["score"]:
            best_return.update(
                {
                    "score": metrics["total_return"],
                    "mr": mr,
                    "metrics": metrics,
                }
            )

        if metrics["martin_ratio"] > best_martin["score"]:
            best_martin.update(
                {
                    "score": metrics["martin_ratio"],
                    "mr": mr,
                    "metrics": metrics,
                }
            )

    res = {
        "best_return": best_return,
        "best_martin": best_martin,
    }
    return resdef optimal_limit_pullback_from_actual_returns(
    trade_log: pd.DataFrame,
    risk_pct: float = 1.0,        # fixed risk per trade, e.g. 1.0 means 1%
    sl_R_base: float = 1.5,       # SL distance in ATR used by the strategy (for x cap)
    x_max: float | None = None,
    grid: int = 401,
    metric: str = "per_setup",    # or "per_taken"
    losers_always_fill: bool = True
) -> tuple[float, pd.DataFrame]:
    """
    Find ATR pullback x that maximizes expected R using ACTUAL trade returns.

    Requires in `trade_log`:
      - 'exit_reason' ∈ {'take_profit','stop_loss'}
      - 'return_pct'  (percent of equity per trade; e.g., -1.0 for -1%)
      - 'min_point_atr' (≥0): intra-trade min drawdown in ATR (excl. entry/exit bars)

    Logic:
      - Convert actual returns to R units: R = return_pct / risk_pct.
      - Let p0 = TP share among {TP,SL} under market entry (from log).
      - With a pullback x:
          winners are taken with probability p0 * q_w(x),
          losers are taken with probability (1 - p0)  (if losers_always_fill).
      - Each filled trade’s payoff shifts by +x R.

    Returns:
      x_best, summary DataFrame (one row per tested x).
    """
    df = trade_log.loc[
        trade_log['exit_reason'].isin(['take_profit', 'stop_loss'])
    ].dropna(subset=['return_pct', 'min_point_atr']).copy()
    if df.empty:
        raise ValueError("No TP/SL trades with 'return_pct' and 'min_point_atr'.")

    # Actual R (risk-normalized return)
    df['R_actual'] = df['return_pct'].astype(float) / float(risk_pct)

    # Split winners/losers
    w_mask = df['exit_reason'].eq('take_profit').to_numpy()
    l_mask = df['exit_reason'].eq('stop_loss').to_numpy()

    R_w = df.loc[w_mask, 'R_actual'].to_numpy(float)
    R_l = df.loc[l_mask, 'R_actual'].to_numpy(float)
    m_w = df.loc[w_mask, 'min_point_atr'].to_numpy(float)
    m_l = df.loc[l_mask, 'min_point_atr'].to_numpy(float)

    nw, nl = R_w.size, R_l.size
    if nw + nl == 0:
        raise ValueError("No TP/SL trades after filtering.")

    # Baseline TP share under market entry
    p0 = nw / (nw + nl)

    # Grid for x (cap below SL distance)
    if x_max is None:
        x_max = sl_R_base - 1e-6
    else:
        x_max = min(x_max, sl_R_base - 1e-6)
    x_vals = np.linspace(0.0, x_max, grid)

    # Winner fill probability q_w(x) = Pr(min >= x | TP)
    qw = np.array([(m_w >= x).mean() if nw else 0.0 for x in x_vals])

    # Mean actual R among winners that would have filled at x
    mean_Rw_filled = np.array([R_w[m_w >= x].mean() if (m_w >= x).any() else 0.0
                               for x in x_vals])

    # Losers: probability mass and mean R
    if losers_always_fill:
        # Probability mass fixed at (1 - p0), independent of x
        take_rate = p0 * qw + (1 - p0)
        mean_Rl_used = np.full_like(x_vals, R_l.mean() if nl else 0.0)
        ql = np.ones_like(x_vals)  # for reporting only
    else:
        ql = np.array([(m_l >= x).mean() if nl else 0.0 for x in x_vals])
        take_rate = p0 * qw + (1 - p0) * ql
        mean_Rl_used = np.array([R_l[m_l >= x].mean() if (m_l >= x).any() else 0.0
                                 for x in x_vals])

    # Expected R per signal (numerator)
    term_win  = p0 * qw * (mean_Rw_filled + x_vals)
    term_loss = (1 - p0) * (mean_Rl_used + x_vals)
    numer = term_win + term_loss

    # Objective
    if metric == "per_taken":
        denom = np.where(take_rate > 0, take_rate, np.nan)
        E_R = numer / denom
    else:  # "per_setup"
        E_R = numer

    summary = pd.DataFrame({
        "x": x_vals,
        "E_R": E_R,
        "qw": qw,
        "ql": ql,
        "take_rate": take_rate,
        "p0": p0,
        "mean_Rw_filled": mean_Rw_filled,
        "mean_Rl_used": mean_Rl_used,
        "metric": metric,
        "risk_pct": risk_pct,
        "sl_R_base": sl_R_base,
        "term_win": term_win,
        "term_loss": term_loss,
    })

    i_best = int(np.nanargmax(E_R))
    return float(x_vals[i_best]), summarydef buy_and_hold_stats(
    prices: pd.DataFrame,
    *,
    equity: float = 10000.0,
    price_col: str = "close",
) -> dict:
    """
    Compute simple Buy&Hold stats on a price series:

    - total_return: % change from first to last price
    - final_equity: equity * (last_price / first_price)
    - max_drawdown: min equity / rolling max - 1 (negative fraction)
    """

    # --- extract and clean price series ---
    if price_col not in prices.columns:
        raise ValueError(f"prices must contain '{price_col}' column")

    px = prices[price_col].astype(float).dropna()
    if px.shape[0] < 2:
        return {
            "total_return": 0.0,
            "final_equity": float(equity),
            "max_drawdown": 0.0,
        }

    # --- equity curve under BnH ---
    # scale prices so that first point corresponds to `equity`
    eq = equity * (px / px.iloc[0])

    # total return in %
    total_return = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0
    final_equity = float(eq.iloc[-1])

    # max drawdown as in calculate_metrics (negative fraction)
    run_max = np.maximum.accumulate(eq.values)
    dd = eq.values / run_max - 1.0
    max_drawdown = float(dd.min())  # e.g. -0.35 for -35%

    return {
        "total_return": total_return,
        "final_equity": final_equity,
        "max_drawdown": max_drawdown,
    }MODELS = {
        "lstm": LSTMClassifier,
        "bilstm": BiLSTMClassifier,
        "gru": GRUClassifier,
        "bigru": BiGRUClassifier,
        "lstm_cat": LSTMCatBoostClassifier,
        "lstm_lgb": LSTMLightGBMClassifier,
        "lstm_xgb": LSTMXGBClassifier,
        "gru_cat": GRUCatBoostClassifier,
        'gru_xgb': GRUXGBClassifier,
        'gru_lgb': GRULightGBMClassifier,
}# FEATURE BLOCKS 

ALL_FEATURES = [
    'volume', 'funding_rate',
    'ema_20', 'ema_50', 'ema_200', 'adx', 'rsi_14',
    'bb_percent', 'bb_width',
    'atr_vol_regime',

    'mfi_14', 'z_volume',
    'vwap', 'vwap_distance',

    'log_return_1h', 'log_return_4h', 'log_return_1d',
    'roll_std_4h', 'roll_std_8h',

    'funding_bias',
    'vwap_session', 'session_vol_mean', 'session_return_mean', 'session_volatility',

    # FD / fractal block
    'fd_24', 'fd_slope', 'fd_ema_12', 'fd_ema_24',
    'fd_trend_strength', 'fd_threshold_causal', 'fd_regime', 'fd_regime_switch',
    'fd_volatility', 'fd_vol_ratio', 'fd_vol_slope', 'fd_slope_atr_norm',
    'fd_entropy', 'fd_vol_adjusted',
    'fd_24_robust_z', 'fd_ema_12_robust_z', 'fd_ema_24_robust_z',
    'fd_trend_strength_robust_z', 'fd_slope_robust_z',
    'fd_slope_atr_norm_robust_z', 'fd_volatility_robust_z',
    'fd_vol_ratio_robust_z', 'fd_vol_slope_robust_z',
    'fd_entropy_robust_z', 'fd_vol_adjusted_robust_z',

    # Patterns / ICT
    'pattern_bullish_engulf', 'pattern_bearish_engulf',
    'pattern_harami', 'pattern_hammer', 'pattern_inverted_hammer',
    'swing_high', 'swing_low', 'last_swing_high', 'last_swing_low',
    'bos_bullish', 'bos_bearish',
    'choch_bullish', 'choch_bearish',
    'mss_bullish', 'mss_bearish',
    'bullish_fvg', 'bearish_fvg', 'fvg_gap',
    'rolling_high', 'rolling_low', 'equilibrium',
    'breakout_bullish', 'breakout_bearish',
    'pattern_count', 'pattern_active',

    # Macro / on-chain
    'macro_event_sentiment', 'macro_event_flag',
    'macro_event_intensity', 'macro_event_intensity_smooth',
    'fear_greed', 'onchain_activity_index',

    # ATR / range & levels
    'atr_pct', 'range_atr', 'body_atr',
    'dist_daily_high', 'dist_daily_low',
    'dist_weekly_high', 'dist_weekly_low',
    'atr_vol_regime_z',

    # Extra TA
    'ppo', 'ppo_signal', 'ppo_hist',
    'bb_z', 'bb_percB',

    # Time / sessions / PDA
    'hour_sin', 'hour_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'session_Asia', 'session_Frankfurt',
    'session_London', 'session_NewYork', 'session_OffHours',
    'pda_Discount', 'pda_Premium',
]

FEATURE_BLOCKS = {
    # core price/vol + basic trend/momentum
    "core_trend_mom": [
        'volume', 'funding_rate',
        'ema_20', 'ema_50', 'ema_200', 'adx', 'rsi_14',
        'bb_percent', 'bb_width',
        'log_return_1h', 'log_return_4h', 'log_return_1d',
        'roll_std_4h', 'roll_std_8h',
        'atr_vol_regime',
        'ppo', 'ppo_signal', 'ppo_hist',
        'bb_z', 'bb_percB',
    ],

    # volume / VWAP / intraday session stats
    "volume_vwap_session": [
        'mfi_14', 'z_volume',
        'vwap', 'vwap_distance',
        'vwap_session', 'session_vol_mean',
        'session_return_mean', 'session_volatility',
    ],

    # fractal / FD regime block
    "fractal_regime": [
        'fd_24', 'fd_slope', 'fd_ema_12', 'fd_ema_24',
        'fd_trend_strength', 'fd_threshold_causal',
        'fd_regime', 'fd_regime_switch',
        'fd_volatility', 'fd_vol_ratio', 'fd_vol_slope',
        'fd_slope_atr_norm', 'fd_entropy', 'fd_vol_adjusted',
        'fd_24_robust_z', 'fd_ema_12_robust_z', 'fd_ema_24_robust_z',
        'fd_trend_strength_robust_z', 'fd_slope_robust_z',
        'fd_slope_atr_norm_robust_z', 'fd_volatility_robust_z',
        'fd_vol_ratio_robust_z', 'fd_vol_slope_robust_z',
        'fd_entropy_robust_z', 'fd_vol_adjusted_robust_z',
    ],

    # ICT-style patterns / structure / FVG / breakouts
    "patterns_ict": [
        'pattern_bullish_engulf', 'pattern_bearish_engulf',
        'pattern_harami', 'pattern_hammer', 'pattern_inverted_hammer',
        'swing_high', 'swing_low', 'last_swing_high', 'last_swing_low',
        'bos_bullish', 'bos_bearish',
        'choch_bullish', 'choch_bearish',
        'mss_bullish', 'mss_bearish',
        'bullish_fvg', 'bearish_fvg', 'fvg_gap',
        'rolling_high', 'rolling_low', 'equilibrium',
        'breakout_bullish', 'breakout_bearish',
        'pattern_count', 'pattern_active',
    ],

    # macro / on-chain
    "macro_onchain": [
        'macro_event_sentiment', 'macro_event_flag',
        'macro_event_intensity', 'macro_event_intensity_smooth',
        'fear_greed', 'onchain_activity_index',
        'funding_bias',
    ],

    # ATR / range & higher-timeframe levels
    "atr_levels": [
        'atr_pct', 'range_atr', 'body_atr',
        'dist_daily_high', 'dist_daily_low',
        'dist_weekly_high', 'dist_weekly_low',
        'atr_vol_regime_z',
    ],

    # time-of-day, weekday, session, PDA
    "time_pda": [
        'hour_sin', 'hour_cos',
        'dayofweek_sin', 'dayofweek_cos',
        'session_Asia', 'session_Frankfurt',
        'session_London', 'session_NewYork', 'session_OffHours',
        'pda_Discount', 'pda_Premium',
    ],
}

essential_features = [
    # Core price/vol/funding
    "volume",
    "funding_rate",
    "mfi_14",
    "z_volume",
    "log_return_15m",
    "log_return_1h",
    "log_return_4h",
    "log_return_1d",
    "roll_std_16",
    "roll_std_32",
    "funding_bias",
    "atr_vol_regime",
    "atr_pct",
    "range_atr",
    "body_atr",

    # Trend / momentum / bands
    "ema_20",
    "ema_50",
    "ema_200",
    "adx",
    "rsi_14",
    "bb_percent",
    "bb_width",
    "bb_percB",
    "ppo",
    "ppo_signal",
    "ppo_hist",

    # Fractal subset
    "fd_96",
    "fd_slope",
    "fd_trend_strength",
    "fd_volatility",
    "fd_vol_ratio",

    # VWAP & intraday stats
    "vwap",
    "vwap_distance",
    "vwap_session",
    "session_vol_mean",
    "session_return_mean",
    "session_volatility",

    # Time-of-day / sessions
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "session_Asia",
    "session_London",
    "session_NewYork",
    "session_OffHours",

    # Macro / on-chain / basis
    "macro_event_flag",
    "macro_event_intensity_smooth",
    "macro_event_sentiment",
    "fear_greed",
    "onchain_activity_index",
    "pda_Discount",
    "pda_Premium",
]
es_metrics = {
    'f1': (f1_metric, 'max'),
    'bss': (bss_metric, 'max'),
    'brier': (brier_metric, 'min'),
}def train_predict(
    train_data, 
    val_data, 
    lr, 
    base_name, 
    input_size, 
    model_type, 
    base_dir='artifacts', 
    dropout = 0.3, 
    num_layers = 2, 
    hidden_size = 128, 
    weights=None, 
    ku=2, 
    kd=1, 
    probability_col=['p2'],
    es_metric = None
):

    if es_metric is not None:
        early_metric_fn = es_metric[0]
        early_metric_mode = es_metric[1]
    else:
        early_metric_fn = None
        early_metric_mode = None
    
    base_dir = f"{base_dir}/{model_type}_{base_name}"
    if model_type in ['gru_cat', 'gru_xgb', 'gru_lgb', 'lstm_cat', 'lstm_xgb', 'lstm_lgb']:
        model_path = base_dir
        loss_plot_path = f"{base_dir}/{model_type}_{base_name}.png"
    else:
        model_path = f"{base_dir}/{model_type}_{base_name}.pt"
        loss_plot_path = f"{base_dir}/{model_type}_{base_name}.png"
    model = MODELS[model_type](
        input_size=input_size, 
        dropout=dropout, 
        num_layers=num_layers, 
        hidden_size=hidden_size, 
        num_classes=num_classes)
    
    output = model.train(train_data, 
                         val_data, 
                         model_path=model_path, 
                         loss_plot_path=loss_plot_path, 
                         batch_size=batch_size, lr=lr, 
                         weights=weights,
                         early_metric_fn=early_metric_fn,
                         early_metric_mode=early_metric_mode,
                        )
    prediction = model.predict(val_data, batch_size=batch_size*2)
    prediction_train = model.predict(train_data, batch_size=batch_size*2)
    # output['val_prediction'] = prediction
    
    # val_metrics = triple_barrier_metrics(
    #                         y_true=prediction["true"],
    #                         y_pred=prediction["pred"],
    #                         p_all=prediction[["p0", "p1", "p2"]],
    #                         ku=ku,
    #                         kd=kd,
    #                     )

    train_metrics = triple_barrier_metrics(
                            y_true=prediction_train["true"],
                            y_pred=prediction_train["pred"],
                            p_all=prediction_train[probability_col],
                            ku=ku,
                            kd=kd,
                        )
    val_metrics = triple_barrier_metrics(
                            y_true=prediction["true"],
                            y_pred=prediction["pred"],
                            p_all=prediction[probability_col],
                            ku=ku,
                            kd=kd,
                        )

    output['val_metrics'] = val_metrics
    output['train_metrics'] = train_metrics
    

    res_json = walk(output)
    with open(_ensure_dir(f"{base_dir}/{model_type}_{base_name}/res-{model_type}_{base_name}.json"), "w", encoding="utf-8") as f:
        json.dump(res_json, f, ensure_ascii=False, indent=2)

    return outputdef train_DL_panel(df_train,
                   df_val,
                   ku, 
                   kd, 
                   hold, 
                   window_size, 
                   lr, 
                   base_name, 
                   base_dir = 'artifacts', 
                   target = ['y'], 
                   model_types=['lstm', 'bilstm', 'gru'], 
                #    min_return = 1.0,
                   hidden_size = 128,
                   num_layers = 2,
                   dropout=0.3,
                   probability_col=['p2'],
                   volatility_col='atr_200',
                   es_metric=None,
                  ):
    
    res = {}
        
    # df_train, df_val, df_test, scaler = data_pipe(df, ku, kd, hold, window_size)

    # val_prices = df_val[['high', 'low', 'close', 'atr_14'] +
    #                         target].iloc[window_size-1:].reset_index(drop=True)
    
    base_name = f'{base_name}_ku{ku}_kd{kd}_hold{hold}_base-window{window_size}_dropout{dropout}_hidden{hidden_size}_layers{num_layers}'
    
    input_size = df_train.drop(
        columns=['open', 'high', 'low', 'close'] + [volatility_col] + target).shape[1]

    print(df_train[target].value_counts(normalize=True))
    train_data = CryptoDataset(
        df_train.drop(columns=['open', 'high', 'low', 'close'] + [volatility_col]), window_size=window_size, target=target)
    val_data = CryptoDataset(df_val.drop(columns=['open', 'high', 'low', 'close'] + [volatility_col]), window_size=window_size, target=target)
    # test_data = CryptoDataset(df_test.drop(columns=[
    #                           'open', 'high', 'low', 'close', 'atr_14']), window_size=window_size, target=target)
    y_train = df_train[target[0]].to_numpy().astype(int)
    cw = make_utility_class_weights(y_train, ku=ku, kd=kd, mode="balanced")
    # cw = [1, 1, 2]
    print(cw)

    for model_type in model_types:
        print('Training ', model_type)        
        res[model_type] = train_predict(train_data, 
                                        val_data, 
                                        lr, 
                                        base_name, 
                                        input_size, 
                                        model_type, 
                                        base_dir=base_dir, 
                                        dropout = dropout, 
                                        num_layers = num_layers, 
                                        hidden_size = hidden_size,
                                        weights=cw,
                                        ku=ku,
                                        kd=kd,
                                        probability_col=probability_col,
                                        es_metric=es_metric
                                        )
        
    # res_json = walk(res)
    # with open(f"{DATA_DIR}/res-{base_name}.json", "w", encoding="utf-8") as f:
    #     json.dump(res_json, f, ensure_ascii=False, indent=2)
    
    
    return resdef load_predict_DL(
    df,
    prices,
    ku, 
    kd, 
    hold, 
    window_size, 
    base_name, 
    base_dir='artifacts', 
    dropout=0.1,
    hidden_size=128,
    num_layers=2,
    target=['y'],
    probability_col=['p2'],
    model_type='gru', 
    min_return=[0],
    slippage=0.00,
    transaction_cost=True,
    position_size=0.01,
    use_limit=False,
    limit_offset=0.0,
    risk_mode='fixed_risk',
    *,
    mr_for_mask: float = 0.0,
    volatility_col='atr_14',
    tf = '15m',
):
    """
    Load DL model, predict, run trading for given min_return values (no calibration),
    and compute model metrics (raw + masked).

    Returns:
      res[mr]:
        'trade_metrics', 'trade_log', 'equity_curve'
      res['predictions']
      res['model_metrics_raw']
      res['model_metrics_masked']
      res['mask']
    """

    res = {}
        
    base_name = (
        f'{base_name}_ku{ku}_kd{kd}_hold{hold}_base-window{window_size}_'
        f'dropout{dropout}_hidden{hidden_size}_layers{num_layers}'
    )
    
    base_dir = f"{base_dir}/{model_type}_{base_name}"
    if model_type in ['gru_cat', 'gru_xgb', 'gru_lgb', 'lstm_cat', 'lstm_xgb', 'lstm_lgb']:
        model_path = base_dir
    else:
        model_path = f"{base_dir}/{model_type}_{base_name}.pt"
    
    input_size = df.drop(
        columns=['open', 'high', 'low', 'close']+ [volatility_col] + target
    ).shape[1]

    data = CryptoDataset(
        df.drop(columns=['open', 'high', 'low', 'close']+ [volatility_col]),
        window_size=window_size,
        target=target
    )
    
    print(df[target].value_counts(normalize=True))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    if model_type in ['gru_cat', 'gru_xgb', 'gru_lgb', 'lstm_cat', 'lstm_xgb', 'lstm_lgb']:
        model = MODELS[model_type].load(model_path)
    else:
        model = MODELS[model_type](
            input_size=input_size,
            dropout=dropout,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes
        )
        state_dict = torch.load(model_path, map_location=device)
        model._load_state_dict(state_dict)
    
    predictions = model.predict(data, batch_size=batch_size*2)

    # --- Build decision mask using reference mr_for_mask ---
    metrics_mask, tlog_mask, equity_info_mask = standard_trade_test(
        predictions=predictions,
        prices=prices,
        ku=ku,
        kd=kd,
        hold=hold,
        probability_column=probability_col,
        atr_column=volatility_col,
        equity=10000.0,
        position_size=position_size,
        risk_mode=risk_mode,
        compound=True,
        transaction_cost=transaction_cost,
        slippage=slippage,
        min_return=mr_for_mask,
        use_limit=use_limit,
        limit_offset=limit_offset,
        tf=tf
    )

    positions = np.asarray(equity_info_mask["positions"], dtype=int)
    if positions.shape[0] != len(predictions):
        raise ValueError("positions length does not match predictions length")
    mask = (positions == 0)

    # --- Model metrics (raw, masked) ---
    model_metrics_raw = triple_barrier_metrics(
        y_true=predictions["true"],
        y_pred=predictions["pred"],
        p_all=predictions[probability_col],
        ku=ku,
        kd=kd,
    )

    if mask.any():
        df_masked = predictions.loc[mask].copy()
        model_metrics_masked = triple_barrier_metrics(
            y_true=df_masked["true"],
            y_pred=df_masked["pred"],
            p_all=df_masked[probability_col],
            ku=ku,
            kd=kd,
        )
        model_metrics_masked["num_decision_bars"] = int(mask.sum())
    else:
        model_metrics_masked = {
            "error": "no decision bars",
            "num_decision_bars": 0,
        }

    # --- Trading for each mr ---
    for mr in min_return:
        tm, tlog, eq_info = standard_trade_test(
            predictions=predictions,
            prices=prices,
            ku=ku,
            kd=kd,
            hold=hold,
            probability_column=probability_col,
            atr_column=volatility_col,
            equity=10000.0,
            position_size=position_size,
            risk_mode="fixed_risk",
            compound=True,
            transaction_cost=transaction_cost,
            slippage=slippage,
            min_return=mr,
            use_limit=use_limit,
            limit_offset=limit_offset,
            tf=tf
        )
        res[mr] = {
            "trade_metrics": tm,
            "trade_log": tlog,
            "equity_curve": eq_info,
        }

    res["predictions"] = predictions
    res["model_metrics_raw"] = model_metrics_raw
    res["model_metrics_masked"] = model_metrics_masked
    res["mask"] = mask

    return res
def HP_optimization(df, 
                    ku, 
                    kd, 
                    hold,
                    lr, 
                    base_name, 
                    base_dir, 
                    target, 
                    model_types, 
                    window_grid, 
                    dropout_grid, 
                    num_layers_grid, 
                    hidden_grid):
    res = {}
    best_values = {model_type : 0 for model_type in model_types}
    res = {model_type: {} for model_type in model_types}

    
    
    for window_size in window_grid:
        df_train, df_val, df_test, scaler = data_pipe(df, ku, kd, hold, window_size, volatility_col=volatility_col)
        for dropout in dropout_grid:
            for num_layers in num_layers_grid:
                for hidden_size in hidden_grid:
                    print(f'Testing window_size: {window_size}, dropout: {dropout}, num_layers: {num_layers}, hidden_size: {hidden_size}')
                    output = train_DL_panel(
                            df_train,
                            df_val,
                            ku, 
                            kd, 
                            hold, 
                            window_size, 
                            lr, 
                            base_name, 
                            base_dir = base_dir, 
                            target = target, 
                            model_types=model_types, 
                            #    min_return = 1.0,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            dropout=dropout,
                  )
                    for model_type in model_types:
                        print(output[model_type]['val_metrics']['tp_f1'])
                        if output[model_type]['val_metrics']['tp_f1'] > best_values[model_type]:
                            best_values[model_type] = output[model_type]['val_metrics']['tp_f1']
                            res[model_type] = {'window_size': window_size,
                                               'dropout': dropout,
                                               'num_layers': num_layers,
                                               'hidden_size': hidden_size,
                                               'val_f1': best_values[model_type]}
                            print(res)
                            with open(f"{base_dir}/ku{ku}_kd{kd}_HP.json", "w", encoding="utf-8") as f:
                                json.dump(res, f, ensure_ascii=False, indent=2)

    with open(f"{base_dir}/ku{ku}_kd{kd}_HP.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(res)
    
    return resdef make_prices(df_split, volatility_col, window_size):
    return (
        df_split[["high", "low", "close", "y", volatility_col]]
        .iloc[window_size - 1 :]
        .reset_index(drop=True)
    )

def fit_logreg_baseline(df_train, window_size, volatility_col, target_col="y"):
    feature_cols = [
        c
        for c in df_train.columns
        if c not in ["open", "high", "low", "close", volatility_col, target_col]
    ]
    df_win = df_train.iloc[window_size - 1 :]
    X_train = df_win[feature_cols].values
    y_train = df_win[target_col].values

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    return clf, feature_cols

def make_pred_df_from_logreg(
    clf,
    df_split,
    feature_cols,
    window_size,
    probability_col,
):
    prob_col_name = probability_col if isinstance(probability_col, str) else probability_col[0]

    df_win = df_split.iloc[window_size - 1 :].reset_index(drop=True)
    X = df_win[feature_cols].values
    y_true = df_win["y"].values
    proba = clf.predict_proba(X)[:, 1]

    pred_df = pd.DataFrame(
        {
            "true": y_true,
            "pred": (proba >= 0.5).astype(int),
            prob_col_name: proba,
        }
    )
    return pred_df

def evaluate_predictions_generic(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    min_returns,
    *,
    tf: str,
    ku: float,
    kd: float,
    hold: int,
    volatility_col: str,
    probability_col,
    position_size: float,
    risk_mode: str,
    use_limit: bool,
    limit_offset: float,
    tc: bool,
    mr_for_mask: float = 0.0,
):
    """
    Generic evaluation (classification + trading) given predictions and prices.
    Returns a dict in the same style as load_predict_DL.
    """
    res = {}
    prob_col_name = probability_col if isinstance(probability_col, str) else probability_col[0]

    # --- mask via triple-barrier backtest at mr_for_mask ---
    metrics_mask, tlog_mask, eq_info_mask = standard_trade_test(
        predictions=predictions,
        prices=prices,
        ku=ku,
        kd=kd,
        hold=hold,
        probability_column=prob_col_name,
        atr_column=volatility_col,
        equity=10000.0,
        position_size=position_size,
        risk_mode=risk_mode,
        compound=True,
        transaction_cost=tc,
        slippage=0.0,
        min_return=mr_for_mask,
        use_limit=use_limit,
        limit_offset=limit_offset,
        tf=tf,
    )

    positions = np.asarray(eq_info_mask["positions"], dtype=int)
    if positions.shape[0] != len(predictions):
        raise ValueError("positions length does not match predictions length")
    mask = positions == 0

    # --- model metrics (raw + masked) ---
    p_all = (
        predictions[[prob_col_name]]
        if isinstance(probability_col, str)
        else predictions[probability_col]
    )
    model_metrics_raw = triple_barrier_metrics(
        y_true=predictions["true"],
        y_pred=predictions["pred"],
        p_all=p_all,
        ku=ku,
        kd=kd,
    )

    if mask.any():
        df_masked = predictions.loc[mask].copy()
        p_all_masked = (
            df_masked[[prob_col_name]]
            if isinstance(probability_col, str)
            else df_masked[probability_col]
        )
        model_metrics_masked = triple_barrier_metrics(
            y_true=df_masked["true"],
            y_pred=df_masked["pred"],
            p_all=p_all_masked,
            ku=ku,
            kd=kd,
        )
        model_metrics_masked["num_decision_bars"] = int(mask.sum())
    else:
        model_metrics_masked = {
            "error": "no decision bars",
            "num_decision_bars": 0,
        }

    # --- trading metrics for each mr ---
    for mr in min_returns:
        metrics, tlog, eq_info = standard_trade_test(
            predictions=predictions,
            prices=prices,
            ku=ku,
            kd=kd,
            hold=hold,
            probability_column=prob_col_name,
            atr_column=volatility_col,
            equity=10000.0,
            position_size=position_size,
            risk_mode=risk_mode,
            compound=True,
            transaction_cost=tc,
            slippage=0.0,
            min_return=mr,
            use_limit=use_limit,
            limit_offset=limit_offset,
            tf=tf,
        )
        res[mr] = {
            "trade_metrics": metrics,
            "trade_log": tlog,
            "equity_curve": eq_info,
        }

    res["predictions"] = predictions
    res["model_metrics_raw"] = model_metrics_raw
    res["model_metrics_masked"] = model_metrics_masked
    res["mask"] = mask

    return res

# ---------------------------------------------------------
# MAIN PIPELINE
# dfs_by_tf = {"1h": df_1h, "4h": df_4h}
# ---------------------------------------------------------
def run_full_comparison(dfs_by_tf):
    """
    dfs_by_tf: dict {"1h": df_1h, "4h": df_4h}
    Returns: all_results[(tf, es_name)] = dict with DL models + ML baseline.
    """
    all_results = {}

    for tf in tfs:
        df = dfs_by_tf[tf].copy()

        # --- tf-specific params ---
        if tf == "1h":
            min_trades = 115
            hold = 332
            window_size = 332
        elif tf == "4h":
            min_trades = 0
            hold = 84
            window_size = 84
        else:
            raise ValueError(f"Unknown tf: {tf}")

        # --- split & prices ---
        df_train, df_val, df_test, scaler = data_pipe(
            df, ku, kd, hold, window_size, volatility_col=volatility_col
        )

        val_prices = make_prices(df_val, volatility_col, window_size)
        test_prices = make_prices(df_test, volatility_col, window_size)

        val_bnh = (val_prices["close"].iloc[-1] / val_prices["close"].iloc[0] - 1) * 100
        test_bnh = (test_prices["close"].iloc[-1] / test_prices["close"].iloc[0] - 1) * 100

        # -------------------------------------------------
        # ML BASELINE (logreg) – trained once per tf
        # -------------------------------------------------
        logreg, feature_cols = fit_logreg_baseline(df_train, window_size, volatility_col)

        prob_col_name = (
            probability_col if isinstance(probability_col, str) else probability_col[0]
        )

        baseline_val_pred = make_pred_df_from_logreg(
            logreg,
            df_val,
            feature_cols,
            window_size,
            probability_col,
        )
        baseline_test_pred = make_pred_df_from_logreg(
            logreg,
            df_test,
            feature_cols,
            window_size,
            probability_col,
        )

        # sweep min_return for baseline on VAL
        baseline_best = sweep_min_return(
            prices=val_prices,
            df_pred=baseline_val_pred,
            ku=ku,
            kd=kd,
            hold=hold,
            min_grid=min_returns,
            artifacts_dir=DATA_DIR,
            slippage=0.0,
            transaction_cost=tc,
            compound=True,
            use_limit=use_limit,
            limit_offset=limit_offset,
            min_trades=min_trades,
            volatility_col=volatility_col,
            probability_col=prob_col_name,
            tf=tf,
            risk_mode=risk_mode,
            position_size=position_size,
        )

        mrs_baseline = [
            baseline_best["best_return"]["mr"],
            baseline_best["best_martin"]["mr"],
            *base_mrs,
        ]
        # unique, non-None
        mrs_baseline = [mr for mr in dict.fromkeys(mrs_baseline) if mr is not None]

        baseline_val = evaluate_predictions_generic(
            baseline_val_pred,
            val_prices,
            mrs_baseline,
            tf=tf,
            ku=ku,
            kd=kd,
            hold=hold,
            volatility_col=volatility_col,
            probability_col=probability_col,
            position_size=position_size,
            risk_mode=risk_mode,
            use_limit=use_limit,
            limit_offset=limit_offset,
            tc=tc,
            mr_for_mask=0.0,
        )
        baseline_test = evaluate_predictions_generic(
            baseline_test_pred,
            test_prices,
            mrs_baseline,
            tf=tf,
            ku=ku,
            kd=kd,
            hold=hold,
            volatility_col=volatility_col,
            probability_col=probability_col,
            position_size=position_size,
            risk_mode=risk_mode,
            use_limit=use_limit,
            limit_offset=limit_offset,
            tc=tc,
            mr_for_mask=0.0,
        )

        # -------------------------------------------------
        # DL MODELS for each ES metric
        # -------------------------------------------------
        for metric_name in es_names:
            base_name = f"rf-vtc_{tf}_new-onchain_{metric_name}"

            # train panel (GRU/LSTM) for this tf + ES metric
            _ = train_DL_panel(
                df_train,
                df_val,
                ku,
                kd,
                hold,
                window_size,
                lr,
                base_name,
                base_dir=DATA_DIR,
                target=target,
                model_types=model_types,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                volatility_col=volatility_col,
                probability_col=probability_col,
                es_metric=es_metrics[metric_name],
            )

            # VAL: predictions + sweep
            val_results = {}
            best_per_model = {}
            best_mrs_per_model = {}

            for model_type in model_types:
                val_res = load_predict_DL(
                    df_val,
                    val_prices,
                    ku,
                    kd,
                    hold,
                    window_size,
                    base_name,
                    base_dir=DATA_DIR,
                    dropout=dropout,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    target=target,
                    model_type=model_type,
                    min_return=base_mrs,
                    slippage=0.0,
                    transaction_cost=tc,
                    position_size=position_size,
                    risk_mode=risk_mode,
                    use_limit=use_limit,
                    limit_offset=limit_offset,
                    mr_for_mask=0.0,
                    volatility_col=volatility_col,
                    probability_col=probability_col,
                    tf=tf,
                )
                val_results[model_type] = val_res

                best = sweep_min_return(
                    prices=val_prices,
                    df_pred=val_res["predictions"],
                    ku=ku,
                    kd=kd,
                    hold=hold,
                    min_grid=min_returns,
                    artifacts_dir=DATA_DIR,
                    slippage=0.0,
                    transaction_cost=tc,
                    compound=True,
                    use_limit=use_limit,
                    limit_offset=limit_offset,
                    min_trades=min_trades,
                    volatility_col=volatility_col,
                    probability_col=prob_col_name,
                    tf=tf,
                    risk_mode=risk_mode,
                    position_size=position_size,
                )
                best_per_model[model_type] = best

                mrs_this = [
                    best["best_return"]["mr"],
                    best["best_martin"]["mr"],
                    *base_mrs,
                ]
                mrs_unique = [mr for mr in dict.fromkeys(mrs_this) if mr is not None]
                best_mrs_per_model[model_type] = mrs_unique

            # TEST: final evaluation
            test_results = {}
            for model_type in model_types:
                test_mrs = best_mrs_per_model[model_type]
                test_res = load_predict_DL(
                    df_test,
                    test_prices,
                    ku,
                    kd,
                    hold,
                    window_size,
                    base_name,
                    base_dir=DATA_DIR,
                    dropout=dropout,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    target=target,
                    model_type=model_type,
                    min_return=test_mrs,
                    slippage=0.0,
                    transaction_cost=tc,
                    position_size=position_size,
                    risk_mode=risk_mode,
                    use_limit=use_limit,
                    limit_offset=limit_offset,
                    mr_for_mask=0.0,
                    volatility_col=volatility_col,
                    probability_col=probability_col,
                    tf=tf,
                )
                test_results[model_type] = test_res

            key = (tf, metric_name)
            all_results[key] = {
                "tf": tf,
                "es_metric": metric_name,
                "val_bnh": val_bnh,
                "test_bnh": test_bnh,
                "val_results": val_results,
                "test_results": test_results,
                "best_per_model": best_per_model,
                "best_mrs_per_model": best_mrs_per_model,
                "baseline_val": baseline_val,
                "baseline_test": baseline_test,
                "baseline_best": baseline_best,
            }

    return all_resultsdef hpo_gru_brier_1h(df_1h: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # 1) single split reused for all HPs (no data leakage across configs)
    df_train, df_val, df_test, scaler = data_pipe(
        df_1h, ku, kd, hold, window_size, volatility_col=volatility_col
    )

    val_prices = (
        df_val[["high", "low", "close", "y", volatility_col]]
        .iloc[window_size - 1 :]
        .reset_index(drop=True)
    )
    test_prices = (
        df_test[["high", "low", "close", "y", volatility_col]]
        .iloc[window_size - 1 :]
        .reset_index(drop=True)
    )

    hpo_rows = []
    best_score = -np.inf
    best_cfg = None

    for hidden_size, dropout, num_layers in itertools.product(
        hidden_grid, dropout_grid, num_layers_grid
    ):
        print("=" * 80)
        print(
            f"HPO trial: hidden={hidden_size}, dropout={dropout}, "
            f"layers={num_layers}"
        )

        # 2) train GRU with Brier ES on this split
        _ = train_DL_panel(
            df_train,
            df_val,
            ku,
            kd,
            hold,
            window_size,
            lr,
            base_name,
            base_dir=DATA_DIR,
            target=target,
            model_types=[model_type],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            volatility_col=volatility_col,
            probability_col=probability_col,
            es_metric=es_metrics[metric_name],
        )

        # 3) VAL: predictions + min_return sweep
        val_res = load_predict_DL(
            df_val,
            val_prices,
            ku,
            kd,
            hold,
            window_size,
            base_name,
            base_dir=DATA_DIR,
            dropout=dropout,
            hidden_size=hidden_size,
            num_layers=num_layers,
            target=target,
            model_type=model_type,
            min_return=base_mrs,         # for base trade metrics; sweep uses full grid
            slippage=0.0,
            transaction_cost=tc,
            position_size=position_size,
            risk_mode=risk_mode,
            use_limit=use_limit,
            limit_offset=limit_offset,
            mr_for_mask=0.0,
            volatility_col=volatility_col,
            probability_col=probability_col,
            tf=tf,
        )

        # sweep over full min_returns grid on VAL
        best = sweep_min_return(
            prices=val_prices,
            df_pred=val_res["predictions"],
            ku=ku,
            kd=kd,
            hold=hold,
            min_grid=min_returns,
            artifacts_dir=DATA_DIR,
            slippage=0.0,
            transaction_cost=tc,
            compound=True,
            use_limit=use_limit,
            limit_offset=limit_offset,
            min_trades=min_trades,
            volatility_col=volatility_col,
            probability_col=probability_col,
            tf=tf,
            risk_mode=risk_mode,
            position_size=position_size,
        )

        best_ret = best["best_return"]
        best_mrt = best["best_martin"]

        # if no mr satisfies min_trades, treat as very bad config
        if best_mrt["mr"] is None or best_mrt["metrics"] is None:
            val_martin = -np.inf
            val_total_ret = -np.inf
            val_mr_star = None
            val_num_trades = 0
        else:
            val_martin = best_mrt["metrics"]["martin_ratio"]
            val_total_ret = best_mrt["metrics"]["total_return"]
            val_mr_star = best_mrt["mr"]
            val_num_trades = best_mrt["metrics"].get("num_trades", 0)

        # classification metrics (unmasked) on VAL
        mm_raw_val = val_res.get("model_metrics_raw", {})
        val_bss = float(mm_raw_val.get("bss", np.nan))
        val_macro_f1 = float(mm_raw_val.get("macro_f1", np.nan))

        # 4) TEST: evaluate at the VAL-best Martin mr (for diagnostics only, not selection)
        test_martin = np.nan
        test_total_ret = np.nan
        test_sharpe = np.nan
        if val_mr_star is not None and np.isfinite(val_martin):
            test_res = load_predict_DL(
                df_test,
                test_prices,
                ku,
                kd,
                hold,
                window_size,
                base_name,
                base_dir=DATA_DIR,
                dropout=dropout,
                hidden_size=hidden_size,
                num_layers=num_layers,
                target=target,
                model_type=model_type,
                min_return=[val_mr_star],
                slippage=0.0,
                transaction_cost=tc,
                position_size=position_size,
                risk_mode=risk_mode,
                use_limit=use_limit,
                limit_offset=limit_offset,
                mr_for_mask=0.0,
                volatility_col=volatility_col,
                probability_col=probability_col,
                tf=tf,
            )
            tm_test = test_res[val_mr_star]["trade_metrics"]
            test_martin = tm_test.get("martin_ratio", np.nan)
            test_total_ret = tm_test.get("total_return", np.nan)
            test_sharpe = tm_test.get("sharpe_ratio", np.nan)

        # 5) record everything
        row = dict(
            tf=tf,
            es_metric=metric_name,
            model_type=model_type,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            window_size=window_size,
            hold=hold,
            ku=ku,
            kd=kd,
            val_mr_star=val_mr_star,
            val_martin=val_martin,
            val_total_return=val_total_ret,
            val_num_trades=val_num_trades,
            val_bss_raw=val_bss,
            val_macro_f1_raw=val_macro_f1,
            test_martin=test_martin,
            test_total_return=test_total_ret,
            test_sharpe=test_sharpe,
        )
        hpo_rows.append(row)

        # 6) HPO selection (VAL only)
        # primary: val_martin
        # tie-break: val_bss_raw, then val_macro_f1_raw, then val_num_trades
        score = val_martin
        if not np.isfinite(score):
            continue  # skip hopeless configs

        def better(a, b):
            """return True if row a is strictly better than row b under our rule"""
            if a["val_martin"] > b["val_martin"] + 1e-8:
                return True
            if abs(a["val_martin"] - b["val_martin"]) <= 1e-8:
                if a["val_bss_raw"] > b["val_bss_raw"] + 1e-8:
                    return True
                if abs(a["val_bss_raw"] - b["val_bss_raw"]) <= 1e-8:
                    if a["val_macro_f1_raw"] > b["val_macro_f1_raw"] + 1e-8:
                        return True
                    if abs(a["val_macro_f1_raw"] - b["val_macro_f1_raw"]) <= 1e-8:
                        if a["val_num_trades"] > b["val_num_trades"]:
                            return True
            return False

        if best_cfg is None:
            best_cfg = row
            best_score = score
        else:
            if better(row, best_cfg):
                best_cfg = row
                best_score = score

        print(
            f"VAL: Martin={val_martin:.3f}, total_ret={val_total_ret:.2f}, "
            f"BSS={val_bss:.4f}, macro_f1={val_macro_f1:.3f}, "
            f"mr*={val_mr_star}, num_trades={val_num_trades}"
        )
        print(
            f"TEST (diag): Martin={test_martin:.3f}, total_ret={test_total_ret:.2f}, "
            f"Sharpe={test_sharpe:.3f}"
        )

    hpo_df = pd.DataFrame(hpo_rows)

    # sort for inspection (VAL-based)
    hpo_df_sorted = hpo_df.sort_values(
        by=["val_martin", "val_bss_raw", "val_macro_f1_raw", "val_num_trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    # save results
    hpo_df_sorted.to_csv(f"gru_brier_1h_ws{window_size}_hold{hold}_hpo_results.csv", index=False)

    print("\nBest config (VAL-based):")
    print(best_cfg)

    return hpo_df_sorted, best_cfgdef add_test_model_metrics_to_hpo(
    hpo_df: pd.DataFrame,
    df_1h: pd.DataFrame,
    save_path: str = "gru_brier_1h_hpo_results_with_model_metrics.csv",
) -> pd.DataFrame:
    """
    For each HPO config, load the trained GRU model and compute
    test classification metrics (raw & masked). Adds columns:

      test_macro_f1_raw,    test_brier_raw,    test_bss_raw,
      test_tp_precision_raw,test_tp_recall_raw,test_tp_f1_raw,
      test_macro_f1_masked, test_brier_masked, test_bss_masked,
      test_tp_precision_masked, test_tp_recall_masked, test_tp_f1_masked

    and saves to `save_path`.
    """
    # 1) Recreate test split once
    df_train, df_val, df_test, scaler = data_pipe(
        df_1h, ku, kd, hold, window_size, volatility_col=volatility_col
    )

    test_prices = (
        df_test[["high", "low", "close", "y", volatility_col]]
        .iloc[window_size - 1 :]
        .reset_index(drop=True)
    )

    # Ensure we work on a copy
    hpo_df = hpo_df.copy()

    # 2) Loop over all HPO rows and compute test model metrics
    for idx, row in hpo_df.iterrows():
        hidden_size = int(row["hidden_size"])
        dropout = float(row["dropout"])
        num_layers = int(row["num_layers"])

        print(
            f"[TEST METRICS] idx={idx}, hidden={hidden_size}, "
            f"dropout={dropout}, layers={num_layers}"
        )

        # We only need model metrics; min_return can be anything (e.g. [0.0])
        test_res = load_predict_DL(
            df_test,
            test_prices,
            ku,
            kd,
            hold,
            window_size,
            base_name,              # same base_name as in HPO
            base_dir=DATA_DIR,
            dropout=dropout,
            hidden_size=hidden_size,
            num_layers=num_layers,
            target=target,
            model_type="gru",       # we did HPO on GRU
            min_return=[0.0],
            slippage=0.0,
            transaction_cost=tc,
            position_size=position_size,
            risk_mode=risk_mode,
            use_limit=use_limit,
            limit_offset=limit_offset,
            mr_for_mask=0.0,
            volatility_col=volatility_col,
            probability_col=probability_col,
            tf="1h",
        )

        mm_raw = test_res.get("model_metrics_raw", {})
        mm_mask = test_res.get("model_metrics_masked", {})

        # Safely extract metrics if present in your triple_barrier_metrics dict
        def get(m, key):
            return float(m.get(key, np.nan))

        # RAW
        hpo_df.loc[idx, "test_macro_f1_raw"]     = get(mm_raw, "macro_f1")
        hpo_df.loc[idx, "test_brier_raw"]        = get(mm_raw, "brier_score")
        hpo_df.loc[idx, "test_bss_raw"]          = get(mm_raw, "bss")
        hpo_df.loc[idx, "test_tp_precision_raw"] = get(mm_raw, "tp_precision")
        hpo_df.loc[idx, "test_tp_recall_raw"]    = get(mm_raw, "tp_recall")
        hpo_df.loc[idx, "test_tp_f1_raw"]        = get(mm_raw, "tp_f1")

        # MASKED (decision bars)
        hpo_df.loc[idx, "test_macro_f1_masked"]     = get(mm_mask, "macro_f1")
        hpo_df.loc[idx, "test_brier_masked"]        = get(mm_mask, "brier_score")
        hpo_df.loc[idx, "test_bss_masked"]          = get(mm_mask, "bss")
        hpo_df.loc[idx, "test_tp_precision_masked"] = get(mm_mask, "tp_precision")
        hpo_df.loc[idx, "test_tp_recall_masked"]    = get(mm_mask, "tp_recall")
        hpo_df.loc[idx, "test_tp_f1_masked"]        = get(mm_mask, "tp_f1")

    # 3) Save and return
    hpo_df.to_csv(save_path, index=False)
    print(f"[OK] Updated HPO results with test model metrics saved to {save_path}")
    return hpo_df# ku/kd sweep

# ---------------------------------
# HELPER
# ---------------------------------
def _get_metric(d, key):
    return float(d.get(key, np.nan)) if isinstance(d, dict) else np.nan

# ---------------------------------
# MAIN SWEEP FUNCTION
# ---------------------------------
def ku_kd_sweep_gru_configs(
    df_1h: pd.DataFrame,
    save_path: str = "gru_brier_1h_ku_kd_sweep.csv",
    base_name_ku_sweep = "btc_sweep"
) -> pd.DataFrame:
    """
    For each (ku,kd) and each selected GRU config:
      - build labels via data_pipe
      - train GRU (Brier ES)
      - sweep min_return on VAL (best_return & best_martin)
      - evaluate both selected mrs (and base_mrs) on TEST
      - collect trade + classification metrics
    """

    rows = []

    for ku_val, kd_val in KU_KD_GRID:
        print("\n" + "#" * 80)
        print(f"=== ku={ku_val}, kd={kd_val} ===")

        # 1) Rebuild split for this ku/kd
        df_train, df_val, df_test, scaler = data_pipe(
            df_1h, ku_val, kd_val, hold, window_size, volatility_col=volatility_col
        )

        val_prices = (
            df_val[["high", "low", "close", "y", volatility_col]]
            .iloc[window_size - 1 :]
            .reset_index(drop=True)
        )
        test_prices = (
            df_test[["high", "low", "close", "y", volatility_col]]
            .iloc[window_size - 1 :]
            .reset_index(drop=True)
        )

        for cfg in GRU_CONFIGS:
            hidden_size = cfg["hidden_size"]
            dropout = cfg["dropout"]
            num_layers = cfg["num_layers"]
            name = cfg["name"]

            print("-" * 80)
            print(f"[TRAIN] {name} | ku={ku_val}, kd={kd_val}")

            # 2) Train GRU with Brier ES on this split
            _ = train_DL_panel(
                df_train,
                df_val,
                ku_val,
                kd_val,
                hold,
                window_size,
                lr,
                base_name_ku_sweep,
                base_dir=DATA_DIR,
                target=target,
                model_types=[model_type],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                volatility_col=volatility_col,
                probability_col=probability_col,
                es_metric=es_metrics[metric_name],
            )

            # 3) VAL: predictions + base-mr run
            val_res = load_predict_DL(
                df_val,
                val_prices,
                ku_val,
                kd_val,
                hold,
                window_size,
                base_name_ku_sweep,
                base_dir=DATA_DIR,
                dropout=dropout,
                hidden_size=hidden_size,
                num_layers=num_layers,
                target=target,
                model_type=model_type,
                min_return=base_mrs,
                slippage=0.0,
                transaction_cost=tc,
                position_size=position_size,
                risk_mode=risk_mode,
                use_limit=use_limit,
                limit_offset=limit_offset,
                mr_for_mask=0.0,
                volatility_col=volatility_col,
                probability_col=probability_col,
                tf=tf,
            )

            df_pred_val = val_res["predictions"]

            # classification metrics (raw) on VAL
            mm_raw_val = val_res.get("model_metrics_raw", {})
            val_macro_f1 = _get_metric(mm_raw_val, "macro_f1")
            val_brier = _get_metric(mm_raw_val, "brier_score")
            val_bss = _get_metric(mm_raw_val, "bss")
            val_tp_prec = _get_metric(mm_raw_val, "tp_precision")
            val_tp_rec = _get_metric(mm_raw_val, "tp_recall")
            val_tp_f1 = _get_metric(mm_raw_val, "tp_f1")

            # 4) sweep min_return on VAL for this ku/kd + model
            best = sweep_min_return(
                prices=val_prices,
                df_pred=df_pred_val,
                ku=ku_val,
                kd=kd_val,
                hold=hold,
                min_grid=min_returns,
                artifacts_dir=DATA_DIR,
                slippage=0.0,
                transaction_cost=tc,
                compound=True,
                use_limit=use_limit,
                limit_offset=limit_offset,
                min_trades=min_trades,
                volatility_col=volatility_col,
                probability_col=probability_col,
                tf=tf,
                risk_mode=risk_mode,
                position_size=position_size,
            )

            best_ret = best["best_return"]
            best_mrt = best["best_martin"]

            val_mr_best_return = best_ret["mr"]
            val_mr_best_martin = best_mrt["mr"]

            # build set of mrs to evaluate on TEST
            mrs_to_eval = []
            for mr in [val_mr_best_return, val_mr_best_martin] + base_mrs:
                if mr is not None and mr not in mrs_to_eval:
                    mrs_to_eval.append(mr)

            print(f"[VAL] {name} ku={ku_val},kd={kd_val}: mrs_to_eval={mrs_to_eval}")

            # 5) TEST: evaluate each mr
            if mrs_to_eval:
                test_res = load_predict_DL(
                    df_test,
                    test_prices,
                    ku_val,
                    kd_val,
                    hold,
                    window_size,
                    base_name_ku_sweep,
                    base_dir=DATA_DIR,
                    dropout=dropout,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    target=target,
                    model_type=model_type,
                    min_return=mrs_to_eval,
                    slippage=0.0,
                    transaction_cost=tc,
                    position_size=position_size,
                    risk_mode=risk_mode,
                    use_limit=use_limit,
                    limit_offset=limit_offset,
                    mr_for_mask=0.0,
                    volatility_col=volatility_col,
                    probability_col=probability_col,
                    tf=tf,
                )

                mm_raw_test = test_res.get("model_metrics_raw", {})
                test_macro_f1 = _get_metric(mm_raw_test, "macro_f1")
                test_brier = _get_metric(mm_raw_test, "brier_score")
                test_bss = _get_metric(mm_raw_test, "bss")
                test_tp_prec = _get_metric(mm_raw_test, "tp_precision")
                test_tp_rec = _get_metric(mm_raw_test, "tp_recall")
                test_tp_f1 = _get_metric(mm_raw_test, "tp_f1")
            else:
                test_res = None
                test_macro_f1 = test_brier = test_bss = np.nan
                test_tp_prec = test_tp_rec = test_tp_f1 = np.nan

            # 6) collect rows for each mr in mrs_to_eval
            for mr in mrs_to_eval:
                tm_val = None
                tm_test = None

                # get VAL trade metrics if available
                if mr in val_res:
                    tm_val = val_res[mr]["trade_metrics"]

                # get TEST trade metrics
                if test_res is not None and mr in test_res:
                    tm_test = test_res[mr]["trade_metrics"]

                row = dict(
                    tf=tf,
                    es_metric=metric_name,
                    model_type=model_type,
                    model_name=name,
                    ku=ku_val,
                    kd=kd_val,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    num_layers=num_layers,
                    window_size=window_size,
                    hold=hold,
                    mr=mr,
                    # selection tag
                    selection=(
                        "best_return" if mr == val_mr_best_return
                        else "best_martin" if mr == val_mr_best_martin
                        else "base"
                    ),
                    # VAL trade metrics
                    val_total_return=_get_metric(tm_val or {}, "total_return"),
                    val_martin=_get_metric(tm_val or {}, "martin_ratio"),
                    val_sharpe=_get_metric(tm_val or {}, "sharpe_ratio"),
                    val_num_trades=_get_metric(tm_val or {}, "num_trades"),
                    # TEST trade metrics
                    test_total_return=_get_metric(tm_test or {}, "total_return"),
                    test_martin=_get_metric(tm_test or {}, "martin_ratio"),
                    test_sharpe=_get_metric(tm_test or {}, "sharpe_ratio"),
                    test_num_trades=_get_metric(tm_test or {}, "num_trades"),
                    # VAL classification (raw)
                    val_macro_f1_raw=val_macro_f1,
                    val_brier_raw=val_brier,
                    val_bss_raw=val_bss,
                    val_tp_precision_raw=val_tp_prec,
                    val_tp_recall_raw=val_tp_rec,
                    val_tp_f1_raw=val_tp_f1,
                    # TEST classification (raw)
                    test_macro_f1_raw=test_macro_f1,
                    test_brier_raw=test_brier,
                    test_bss_raw=test_bss,
                    test_tp_precision_raw=test_tp_prec,
                    test_tp_recall_raw=test_tp_rec,
                    test_tp_f1_raw=test_tp_f1,
                )
                rows.append(row)

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(save_path, index=False)
    print(f"[OK] ku/kd sweep saved to {save_path}")
    return sweep_df


# Example call:
# sweep_results = ku_kd_sweep_gru_configs(df_1h)
def run_block_ablation_experiment(
    df: pd.DataFrame,
    *,
    ku: float,
    kd: float,
    hold: int,
    window_size: int,
    base_name: str,
    DATA_DIR: str,
    model_types: list[str],
    volatility_col: str,
    probability_col: str,
    lr: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    tc: float,
    use_limit: bool,
    limit_offset: float,
    mr_fixed: float,
    tf: str = '1h',
    es_metric = None,
) -> pd.DataFrame:
    """
    For each block in FEATURE_BLOCKS:
      - drop that block from features (keep all others),
      - train models from scratch on reduced features,
      - evaluate on val & test at fixed mr_fixed,
      - collect metrics into a single DataFrame.

    Only reduced-feature models are trained; full-feature model is assumed
    to already exist (you can merge with its metrics later).
    """

    non_feature_cols = [
        "timestamp", "open", "high", "low", "close",
        volatility_col, "y"
    ]
    non_feature_cols = [c for c in non_feature_cols if c in df.columns]

    all_ablation_rows = []

    # shared BnH will be re-computed per ablation (same prices)
    for block_name, block_feats in FEATURE_BLOCKS.items():
        print("\n" + "#" * 80)
        print(f"ABLATION: dropping block '{block_name}'")

        # features to KEEP = all features minus this block
        keep_features = [f for f in ALL_FEATURES if f not in set(block_feats)]
        keep_features = [f for f in keep_features if f in df.columns]

        print(f"  Keeping {len(keep_features)} features, "
              f"dropping {len(block_feats)} from block '{block_name}'")

        df_sub = df[non_feature_cols + keep_features].copy()

        # --- split & scale
        df_train, df_val, df_test, scaler = data_pipe(
            df_sub,
            ku,
            kd,
            hold,
            window_size,
            volatility_col=volatility_col,
        )

        # --- prices (as in your DL pipeline)
        val_prices = df_val[["high", "low", "close", "y", volatility_col]].iloc[window_size-1:].reset_index(drop=True)
        test_prices = df_test[["high", "low", "close", "y", volatility_col]].iloc[window_size-1:].reset_index(drop=True)

        val_bnh  = (val_prices["close"].iloc[-1]  / val_prices["close"].iloc[0]  - 1) * 100
        test_bnh = (test_prices["close"].iloc[-1] / test_prices["close"].iloc[0] - 1) * 100

        # --- train on reduced features
        exp_base = f"{base_name}_minus_{block_name}"

        _ = train_DL_panel(
            df_train,
            df_val,
            ku=ku,
            kd=kd,
            hold=hold,
            window_size=window_size,
            lr=lr,
            base_name=exp_base,
            base_dir=DATA_DIR,
            target=["y"],
            model_types=model_types,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            probability_col=probability_col,
            volatility_col=volatility_col,
            es_metric=es_metric,
            
        )

        # --- eval on val & test at fixed mr ---
        base_mrs = [mr_fixed]
        val_results = {}
        test_results = {}

        for model_type in model_types:
            # VAL
            val_res = load_predict_DL(
                df_val,
                val_prices,
                ku,
                kd,
                hold,
                window_size,
                base_name=exp_base,
                base_dir=DATA_DIR,
                dropout=dropout,
                hidden_size=hidden_size,
                num_layers=num_layers,
                target=["y"],
                model_type=model_type,
                min_return=base_mrs,
                slippage=0.0,
                transaction_cost=tc,
                position_size=0.01,
                risk_mode="fixed_risk",
                use_limit=use_limit,
                limit_offset=limit_offset,
                mr_for_mask=0.0,
                volatility_col=volatility_col,
                probability_col=probability_col,
                tf=tf
            )
            val_results[model_type] = val_res

            # TEST
            test_res = load_predict_DL(
                df_test,
                test_prices,
                ku,
                kd,
                hold,
                window_size,
                base_name=exp_base,
                base_dir=DATA_DIR,
                dropout=dropout,
                hidden_size=hidden_size,
                num_layers=num_layers,
                target=["y"],
                model_type=model_type,
                min_return=[mr_fixed],
                slippage=0.0,
                transaction_cost=tc,
                position_size=0.01,
                risk_mode="fixed_risk",
                use_limit=use_limit,
                limit_offset=limit_offset,
                mr_for_mask=0.0,
                volatility_col=volatility_col,
                probability_col=probability_col,
                tf=tf
            )
            test_results[model_type] = test_res

        # --- gather metrics for this ablation ---
        for model_type in model_types:
            vr = val_results[model_type]
            tr = test_results[model_type]

            vm_raw = vr["model_metrics_raw"]
            tm_raw = tr["model_metrics_raw"]

            vtrade = vr[mr_fixed]["trade_metrics"]
            ttrade = tr[mr_fixed]["trade_metrics"]

            row = {
                "dropped_block": block_name,
                "model": model_type,
                "n_features_used": len(keep_features),

                # ---- val (unmasked) ----
                "val_tp_precision": vm_raw.get("tp_precision", np.nan),
                "val_tp_recall": vm_raw.get("tp_recall", np.nan),
                "val_tp_f1": vm_raw.get("tp_f1", np.nan),
                "val_macro_f1": vm_raw.get("macro_f1", np.nan),
                "val_brier": vm_raw.get("brier", np.nan),
                "val_bss": vm_raw.get("bss", np.nan),

                "val_total_return": vtrade.get("total_return", np.nan),
                "val_martin": vtrade.get("martin_ratio", np.nan),
                "val_calmar": vtrade.get("calmar_ratio", np.nan),
                "val_sharpe": vtrade.get("sharpe_ratio", np.nan),
                "val_num_trades": vtrade.get("num_trades", np.nan),

                # ---- test (unmasked) ----
                "test_tp_precision": tm_raw.get("tp_precision", np.nan),
                "test_tp_recall": tm_raw.get("tp_recall", np.nan),
                "test_tp_f1": tm_raw.get("tp_f1", np.nan),
                "test_macro_f1": tm_raw.get("macro_f1", np.nan),
                "test_brier": tm_raw.get("brier", np.nan),
                "test_bss": tm_raw.get("bss", np.nan),

                "test_total_return": ttrade.get("total_return", np.nan),
                "test_martin": ttrade.get("martin_ratio", np.nan),
                "test_calmar": ttrade.get("calmar_ratio", np.nan),
                "test_sharpe": ttrade.get("sharpe_ratio", np.nan),
                "test_num_trades": ttrade.get("num_trades", np.nan),

                "val_bnh": val_bnh,
                "test_bnh": test_bnh,
                "mr_fixed": mr_fixed,
            }
            all_ablation_rows.append(row)

    ablation_df = pd.DataFrame(all_ablation_rows)
    ablation_df = ablation_df.sort_values(
        ["model", "dropped_block"]
    ).reset_index(drop=True)

    return ablation_dfif in_kaggle():
    DATA_DIR = "/kaggle/working"
elif in_colab():
    DATA_DIR = "/content"   # typical Colab working dir
else:
    DATA_DIR = "artifacts"# PARAMS

target = ['y']
# X_cols = CORE_PRICE_VOL
volatility_col = 'atr_200'
probability_col = 'p1'

window_grid=[48, 96]

lr = 1e-4

num_classes = 2
dropout = 0.1
dropout_grid = [0.1, 0.5]

num_layers_grid=[2]
num_layers = 1

hidden_grid=[128, 384, 512]
hidden_size = 384

ku = 6
kd = 2

min_returns = np.linspace(0, 0.5, int(200+1))

use_limit = True
limit_offset = 0.0
position_size = 0.05
risk_mode = 'fixed_risk'
min_return = 0.2
tc = True

tf = '1h'
if tf == '1h':
    min_trades = 115
elif tf == '15m': 
    min_trades = 230
else:
    min_trades = 0

if tf == '1h':
    hold = 336
    window_size = 336
elif tf == '4h':
    hold = 84
    window_size = 84

batch_size = 256
metric = 'brier'
# base_name = f'rf-vtc_{tf}_new-onchain_{metric}'
base_name = f'btc-new_{tf}_{metric}'

model_types = ['lstm', 'bilstm', 'gru', 'bigru','gru_cat']# Update the path below to point to the desired preprocessed parquet file.
df = fetch_data(tf)

df_1h = fetch_data('1h')

print(list(df.columns))
print(len(list(df.columns)))
print(len(df))
# df#training
df_train, df_val, df_test, scaler = data_pipe(df, ku, kd, hold, window_size, volatility_col=volatility_col)

output = train_DL_panel(
                            df_train,
                            df_val,
                            ku, 
                            kd, 
                            hold, 
                            window_size, 
                            lr, 
                            base_name, 
                            base_dir = DATA_DIR, 
                            target = target, 
                            model_types=model_types, 
                            #    min_return = 1.0,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            dropout=dropout,
                            volatility_col=volatility_col,
                            probability_col=probability_col,
                            es_metric=es_metrics[metric]
                  )
#TRAIN METRICS
for model in model_types:
    print(model) 
    print("Train metrics: ", output[model]['train_metrics'])
    print("Val metrics: ", output[model]['val_metrics'])#RANDOM METRICS

# y_train: labels from train set (e.g. df_train[target])
# test_preds: model predictions dict from load_predict_DL(...) on test
# we reuse the same true labels as for the model metrics
y_train = df_train[target]              # Series of 0/1/2
y_true_test = test_prices['y']
y_true_val = val_prices['y']

test_pred_df = make_random_prediction_df_from_priors(
    y_train=y_train,
    y_true_test=y_true_test,
    seed=42,
)

val_pred_df = make_random_prediction_df_from_priors(
    y_train=y_train,
    y_true_test=y_true_val,
    seed=42,
)

# best = sweep_min_return(
#         prices=val_prices,
#         df_pred=df_pred_val,
#         ku=ku,
#         kd=kd,
#         hold=hold,
#         min_grid=min_returns,
#         artifacts_dir=DATA_DIR,
#         slippage=0.0,
#         transaction_cost=tc,
#         compound=True,
#         use_limit=use_limit,
#         limit_offset=limit_offset,
#         min_trades=min_trades,
#         volatility_col=volatility_col,
#         probability_col=probability_col
#     )
# best_per_model = best

# for key in ["best_return", "best_martin"]:
#         mr_star = best[key]["mr"]
#         metrics_star = best[key]["metrics"]
#         print(f"  {key}:")
#         if mr_star is None:
#             print("    no mr satisfies min_trades on this split")
#         else:
#             print(f"    mr = {mr_star}")
#             print(f"    metrics = {metrics_star}")

# # Build mr list for test: best_return, best_martin, plus base_mrs
# mrs_this = [
#     best["best_return"]["mr"],
#     best["best_martin"]["mr"],
# ] + base_mrs
# mrs_this = [mr for mr in mrs_this if mr is not None]

# # unique, keep order
# seen = set()
# mrs_unique = []
# for mr in mrs_this:
#     if mr not in seen:
#         seen.add(mr)
#         mrs_unique.append(mr)

# best_mrs_per_model = mrs_unique

rand_val_metrics = triple_barrier_metrics(
    y_true=val_pred_df["true"],
    y_pred=val_pred_df["pred"],
    p_all=val_pred_df[probability_col],
    ku=ku,
    kd=kd,
)
rand_test_metrics = triple_barrier_metrics(
    y_true=test_pred_df["true"],
    y_pred=test_pred_df["pred"],
    p_all=test_pred_df[probability_col],
    ku=ku,
    kd=kd,
)

rand_trade_metrics = {}

# for mr in best_mrs:
#     rand_trade_metrics[mr], _, _ = standard_trade_test(
#         predictions=test_pred_df,
#         prices=test_prices,
#         ku=ku,
#         kd=kd,
#         hold=hold,
#         probability_column="p2",
#         atr_column="atr_14",
#         equity=10000.0,
#         position_size=position_size,
#         risk_mode="fixed_risk",
#         compound=True,
#         transaction_cost=tc,
#         slippage=0,
#         min_return=mr,
#         use_limit=use_limit,
#         limit_offset=limit_offset,
#     tf=tf
#     )


print("Val rand metrics: ", rand_val_metrics)
print("Test rand metrics: ", rand_test_metrics)
# print(rand_trade_metrics)
#PERFORMANCE TEST 

# --- Split ---
df_train, df_val, df_test, scaler = data_pipe(
    df, ku, kd, hold, window_size, volatility_col=volatility_col
)
df_test = df_test.iloc[int(len(df_test)/2):]

val_prices = df_val[["high", "low", "close", "y", volatility_col]].iloc[window_size-1:].reset_index(drop=True)
test_prices = df_test[["high", "low", "close", "y", volatility_col]].iloc[window_size-1:].reset_index(drop=True)
test_prices = test_prices.iloc[int(len(test_prices)/2):]

# --- BnH ---
val_bnh  = (val_prices["close"].iloc[-1]  / val_prices["close"].iloc[0]  - 1) * 100
test_bnh = (test_prices["close"].iloc[-1] / test_prices["close"].iloc[0] - 1) * 100

# Base mrs just for reference trading run
base_mrs = [0.0]

# ---------------- VAL: predictions + base-mr trading ----------------
val_results = {}

for model_type in model_types:
    val_results[model_type] = load_predict_DL(
        df_val,
        val_prices,
        ku,
        kd,
        hold,
        window_size,
        base_name,
        base_dir=DATA_DIR,
        dropout=dropout,
        hidden_size=hidden_size,
        num_layers=num_layers,
        target=["y"],
        model_type=model_type,
        min_return=base_mrs,
        slippage=0.0,
        transaction_cost=tc,
        position_size=position_size,
        risk_mode=risk_mode,
        use_limit=use_limit,
        limit_offset=limit_offset,
        mr_for_mask=0.0,
        volatility_col=volatility_col,
        probability_col=probability_col,
        tf=tf,
    )

# ---------------- VAL: sweep min_return per model ----------------
best_per_model = {}
mrs_for_test = {}

for model_type in model_types:
    df_pred_val = val_results[model_type]["predictions"]

    best = sweep_min_return(
        prices=val_prices,
        df_pred=df_pred_val,
        ku=ku,
        kd=kd,
        hold=hold,
        min_grid=min_returns,
        artifacts_dir=DATA_DIR,
        slippage=0.0,
        transaction_cost=tc,
        compound=True,
        use_limit=use_limit,
        limit_offset=limit_offset,
        min_trades=min_trades,
        volatility_col=volatility_col,
        probability_col=probability_col,
        position_size=position_size,
        risk_mode=risk_mode,
        tf=tf,
    )
    best_per_model[model_type] = best

    # Build mr list for test: best_return, best_martin, plus base_mrs
    mrs_this = [
        best["best_return"]["mr"],
        best["best_martin"]["mr"],
    ] + base_mrs
    mrs_this = [mr for mr in mrs_this if mr is not None]

    # unique, keep order
    seen = set()
    mrs_unique = []
    for mr in mrs_this:
        if mr not in seen:
            seen.add(mr)
            mrs_unique.append(mr)

    mrs_for_test[model_type] = mrs_unique

# ---------------- TEST: final evaluation ----------------
test_results = {}
eq = {}

for model_type in model_types:
    test_mrs = mrs_for_test[model_type]

    test_res = load_predict_DL(
        df_test,
        test_prices,
        ku,
        kd,
        hold,
        window_size,
        base_name,
        base_dir=DATA_DIR,
        dropout=dropout,
        hidden_size=hidden_size,
        num_layers=num_layers,
        target=["y"],
        model_type=model_type,
        min_return=test_mrs,
        slippage=0.0,
        transaction_cost=tc,
        position_size=position_size,
        risk_mode=risk_mode,
        use_limit=use_limit,
        limit_offset=limit_offset,
        mr_for_mask=0.0,
        volatility_col=volatility_col,
        probability_col=probability_col,
        tf=tf,
    )
    test_results[model_type] = test_res
    eq[model_type] = test_res[0]['equity_curve']['equity']

# ---------------- CLASSIFICATION DF (UNMASKED ONLY) ----------------
cls_rows = []

def _get(d, key):
    return d.get(key, np.nan)

for model_type in model_types:
    # VAL
    cls_val = val_results[model_type]["model_metrics_raw"]
    cls_rows.append(
        {
            "split": "val",
            "model": model_type,
            "tp_precision": _get(cls_val, "tp_precision"),
            "tp_recall": _get(cls_val, "tp_recall"),
            "tp_f1": _get(cls_val, "tp_f1"),
            "macro_precision": _get(cls_val, "macro_precision"),
            "macro_recall": _get(cls_val, "macro_recall"),
            "macro_f1": _get(cls_val, "macro_f1"),
            "brier": _get(cls_val, "brier"),
            "bss": _get(cls_val, "bss"),
        }
    )

    # TEST
    cls_test = test_results[model_type]["model_metrics_raw"]
    cls_rows.append(
        {
            "split": "test",
            "model": model_type,
            "tp_precision": _get(cls_test, "tp_precision"),
            "tp_recall": _get(cls_test, "tp_recall"),
            "tp_f1": _get(cls_test, "tp_f1"),
            "macro_precision": _get(cls_test, "macro_precision"),
            "macro_recall": _get(cls_test, "macro_recall"),
            "macro_f1": _get(cls_test, "macro_f1"),
            "brier": _get(cls_test, "brier"),
            "bss": _get(cls_test, "bss"),
        }
    )

cls_summary = pd.DataFrame(cls_rows).sort_values(["split", "model"]).reset_index(drop=True)

# ---------------- TRADING DF (WITH CRITERION) ----------------
trade_rows = []

# VAL: only base mrs, criterion="base"
for model_type in model_types:
    for mr in base_mrs:
        tm = val_results[model_type][mr]["trade_metrics"]
        trade_rows.append(
            {
                "split": "val",
                "model": model_type,
                "mr": mr,
                "criterion": "base",
                "sharpe_ratio": _get(tm, "sharpe_ratio"),
                "calmar_ratio": _get(tm, "calmar_ratio"),
                "martin_ratio": _get(tm, "martin_ratio"),
                "total_return": _get(tm, "total_return"),
                "max_drawdown": _get(tm, "max_drawdown"),
                "volatility": _get(tm, "volatility"),
                "winrate": _get(tm, "winrate"),
                "num_trades": _get(tm, "num_trades"),
                "final_equity": _get(tm, "final_equity"),
            }
        )

# TEST: mrs from best_return / best_martin / base_mrs
for model_type in model_types:
    best_ret_mr = best_per_model[model_type]["best_return"]["mr"]
    best_mrt_mr = best_per_model[model_type]["best_martin"]["mr"]

    # map mr -> list of tags
    crit_map = {}
    if best_ret_mr is not None:
        crit_map.setdefault(best_ret_mr, []).append("best_return")
    if best_mrt_mr is not None:
        crit_map.setdefault(best_mrt_mr, []).append("best_martin")
    for mr_base in base_mrs:
        if mr_base not in crit_map:
            crit_map.setdefault(mr_base, []).append("base")

    for mr in mrs_for_test[model_type]:
        tags = crit_map.get(mr, ["base"])
        # precedence: best_return > best_martin > base
        if "best_return" in tags:
            criterion = "best_return"
        elif "best_martin" in tags:
            criterion = "best_martin"
        else:
            criterion = "base"

        tm = test_results[model_type][mr]["trade_metrics"]
        trade_rows.append(
            {
                "split": "test",
                "model": model_type,
                "mr": mr,
                "criterion": criterion,
                "sharpe_ratio": _get(tm, "sharpe_ratio"),
                "calmar_ratio": _get(tm, "calmar_ratio"),
                "martin_ratio": _get(tm, "martin_ratio"),
                "total_return": _get(tm, "total_return"),
                "max_drawdown": _get(tm, "max_drawdown"),
                "volatility": _get(tm, "volatility"),
                "winrate": _get(tm, "winrate"),
                "num_trades": _get(tm, "num_trades"),
                "final_equity": _get(tm, "final_equity"),
            }
        )

# BnH rows in trading DF
trade_rows.append(
    {
        "split": "val",
        "model": "bnh",
        "mr": np.nan,
        "criterion": "bnh",
        "sharpe_ratio": np.nan,
        "calmar_ratio": np.nan,
        "martin_ratio": np.nan,
        "total_return": float(val_bnh),
        "max_drawdown": np.nan,
        "volatility": np.nan,
        "winrate": np.nan,
        "num_trades": np.nan,
        "final_equity": np.nan,
    }
)
trade_rows.append(
    {
        "split": "test",
        "model": "bnh",
        "mr": np.nan,
        "criterion": "bnh",
        "sharpe_ratio": np.nan,
        "calmar_ratio": np.nan,
        "martin_ratio": np.nan,
        "total_return": float(test_bnh),
        "max_drawdown": np.nan,
        "volatility": np.nan,
        "winrate": np.nan,
        "num_trades": np.nan,
        "final_equity": np.nan,
    }
)

trade_summary = pd.DataFrame(trade_rows).sort_values(
    ["split", "model", "criterion", "mr"]
).reset_index(drop=True)

# optional: save
RESULTS_DIR = os.path.join(DATA_DIR, "res")
os.makedirs(RESULTS_DIR, exist_ok=True)

cls_summary.to_csv("res/cls_summary.csv", index=False)
trade_summary.to_csv("res/trade_summary.csv", index=False)


cls_summarytrade_summary[trade_summary['mr']==0]df_train, df_val, df_test, scaler = data_pipe(
    df, ku, kd, hold, window_size, volatility_col=volatility_col
)
test_prices = df_test[["high", "low", "close", "y", volatility_col]].iloc[window_size-1:].reset_index(drop=True)

test_bnh_stats = buy_and_hold_stats(test_prices, equity=10000.0, price_col="close")
test_bnh_stats# ABLATION TEST
ALL_FROM_BLOCKS = sorted({f for fs in FEATURE_BLOCKS.values() for f in fs})
missing = sorted(set(ALL_FEATURES) - set(ALL_FROM_BLOCKS))
if missing:
    print("[WARN] Some features are not assigned to blocks:", missing)

model_types = ['gru_cat']

ablation_results = run_block_ablation_experiment(
    df=df,                   # full df with all features + OHLC + y + atr_200 etc.
    ku=ku,
    kd=kd,
    hold=hold,
    window_size=window_size,
    base_name=base_name,     # same base_name you used for full-feature training
    DATA_DIR=DATA_DIR,
    model_types=model_types, # e.g. ["gru", "lstm"]
    volatility_col=volatility_col,
    probability_col=probability_col,
    lr=lr,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    tc=tc,
    use_limit=use_limit,
    limit_offset=limit_offset,
    mr_fixed=min_return,
    tf=tf,
    es_metric=es_metrics[metric],
)

ablation_results.to_csv(f"{DATA_DIR}/ablation.csv")
ablation_results# ------HPO PARAMS----------
tf = "1h"
metric_name = "brier"          # ES = Brier
model_type = "gru_cat"             # HPO only for GRU

window_size = 336
hold = 336
min_trades = 115
base_mrs = [0.0]

base_name = f"hpo_{model_type}_{window_size}_{hold}_{tf}_{metric_name}"

# HPO search space (adjust if needed)
hidden_grid = [128, 256, 384, 512]
dropout_grid = [0.1]
num_layers_grid = [1, 2, 3]# ------HPO----------
hpo_results_df, best_config = hpo_gru_brier_1h(df_1h)hpo_results_with_metrics = add_test_model_metrics_to_hpo(hpo_results_df, df_1h)
hpo_results_with_metricsRESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
fname = 'HPO_res_gru_brier_1h'
full_path = os.path.join(RESULTS_DIR, fname)
hpo_results_df.to_csv(full_path, index=False)
# ML ASSESSMENT

def evaluate_test_only_models(
    model_dfs: dict,
    prices: pd.DataFrame,
    *,
    ku: float,
    kd: float,
    hold: int,
    tf: str,
    atr_col: str = "atr_200",
    y_col: str = "y",
    proba_col: str = "p1",
    pred_col: str = "pred",
    base_mrs=(0.0, 0.06, 0.1),
    position_size: float = 0.01,
    risk_mode: str = "fixed_risk",
    use_limit: bool = True,
    limit_offset: float = 0.0,
    tc: bool = True,
    slippage: float = 0.0,
):
    """
    Parameters
    ----------
    model_dfs : dict[str, pd.DataFrame]
        Mapping model_name -> dataframe with at least columns:
        [y_col, pred_col, proba_col].

        Length must match `prices`.

    prices : pd.DataFrame
        Shared price frame (test slice), e.g. like test_prices in your DL pipeline.
        Must contain at least: ["close", atr_col].
        (If you use OHLC triple-barrier, it can also contain ["high","low"] etc.)

    Returns
    -------
    cls_summary : pd.DataFrame
        Classification metrics (test only, unmasked).

    trade_summary : pd.DataFrame
        Trading metrics (test only) with 'criterion' in {'base', 'bnh'}.
    """

    def _get(d, key):
        return d.get(key, np.nan)

    # --- ensure prices is clean and indexed 0..N-1 ---
    prices = prices.reset_index(drop=True)

    # --- Buy & Hold, once for the shared test window ---
    if "close" not in prices.columns:
        raise ValueError("prices must contain 'close' column")

    bnh_ret = (prices["close"].iloc[-1] / prices["close"].iloc[0] - 1) * 100

    cls_rows = []
    trade_rows = []

    # BnH baseline row (shared for all models)
    trade_rows.append(
        {
            "split": "test",
            "model": "bnh",
            "mr": np.nan,
            "criterion": "bnh",
            "sharpe_ratio": np.nan,
            "calmar_ratio": np.nan,
            "martin_ratio": np.nan,
            "total_return": float(bnh_ret),
            "max_drawdown": np.nan,
            "volatility": np.nan,
            "winrate": np.nan,
            "num_trades": np.nan,
            "final_equity": np.nan,
        }
    )

    # --- per-model evaluation ---
    for model_name, df in model_dfs.items():
        df = df.reset_index(drop=True)

        # length alignment check
        if len(df) != len(prices):
            raise ValueError(
                f"Length mismatch for model '{model_name}': "
                f"len(preds)={len(df)}, len(prices)={len(prices)}"
            )

        # ----- prediction frame for metrics & backtest -----
        if y_col not in df.columns:
            raise ValueError(f"'{y_col}' not found in model df for '{model_name}'")
        if pred_col not in df.columns:
            raise ValueError(f"'{pred_col}' not found in model df for '{model_name}'")
        if proba_col not in df.columns:
            raise ValueError(f"'{proba_col}' not found in model df for '{model_name}'")

        preds = pd.DataFrame(
            {
                "true": df[y_col].astype(int).values,
                "pred": df[pred_col].astype(int).values,
                proba_col: df[proba_col].astype(float).values,
            }
        )

        # ----- classification metrics (unmasked) -----
        cls = triple_barrier_metrics(
            y_true=preds["true"],
            y_pred=preds["pred"],
            p_all=preds[[proba_col]],
            ku=ku,
            kd=kd,
        )

        cls_rows.append(
            {
                "split": "test",
                "model": model_name,
                "tp_precision": _get(cls, "tp_precision"),
                "tp_recall": _get(cls, "tp_recall"),
                "tp_f1": _get(cls, "tp_f1"),
                "macro_precision": _get(cls, "macro_precision"),
                "macro_recall": _get(cls, "macro_recall"),
                "macro_f1": _get(cls, "macro_f1"),
                "brier": _get(cls, "brier"),
                "bss": _get(cls, "bss"),
            }
        )

        # ----- trading metrics for base min_return values -----
        for mr in base_mrs:
            tm, tlog, eq = standard_trade_test(
                predictions=preds,
                prices=prices,
                ku=ku,
                kd=kd,
                hold=hold,
                probability_column=proba_col,
                atr_column=atr_col,
                equity=10000.0,
                position_size=position_size,
                risk_mode=risk_mode,
                compound=True,
                transaction_cost=tc,
                slippage=slippage,
                min_return=mr,
                use_limit=use_limit,
                limit_offset=limit_offset,
                tf=tf,
            )

            trade_rows.append(
                {
                    "split": "test",
                    "model": model_name,
                    "mr": mr,
                    "criterion": "base",
                    "sharpe_ratio": _get(tm, "sharpe_ratio"),
                    "calmar_ratio": _get(tm, "calmar_ratio"),
                    "martin_ratio": _get(tm, "martin_ratio"),
                    "total_return": _get(tm, "total_return"),
                    "max_drawdown": _get(tm, "max_drawdown"),
                    "volatility": _get(tm, "volatility"),
                    "winrate": _get(tm, "winrate"),
                    "num_trades": _get(tm, "num_trades"),
                    "final_equity": _get(tm, "final_equity"),
                }
            )

    cls_summary = (
        pd.DataFrame(cls_rows)
        .sort_values(["split", "model"])
        .reset_index(drop=True)
    )

    trade_summary = (
        pd.DataFrame(trade_rows)
        .sort_values(["split", "model", "criterion", "mr"])
        .reset_index(drop=True)
    )

    return cls_summary, trade_summaryRESULTS_DIR = Path('/kaggle/input/ml-preds')

# === 2. Model name patterns to look for in filenames ===
MODEL_PATTERNS = {
    "cat": "cat",
    "lgbm": "lgbm",
    "rf": "rf",
    "stacking_raw": "stacking_raw",
    "stacking_cal": "stacking_cal",   # in case filename has "stack" instead of "stacking"
    "xgb": "xgb",
}

# === 3. Scan folder and load into dfs ===
model_dfs = {}  # e.g. {"cat": df_cat, "lgbm": df_lgbm, ...}

for path in sorted(RESULTS_DIR.glob("**/*")):
    if not path.is_file():
        continue

    suffix = path.suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        continue

    fname = path.name.lower()

    # detect model by substring in filename
    model_name = None
    for pattern, canonical in MODEL_PATTERNS.items():
        if pattern in fname:
            model_name = canonical
            break

    if model_name is None:
        # file doesn't match any known model pattern
        continue

    # load file
    if suffix == ".csv":
        df = pd.read_csv(path)
    else:  # ".parquet"
        df = pd.read_parquet(path)

    # handle duplicates (if more than one file per model)
    if model_name in model_dfs:
        i = 2
        alt_name = f"{model_name}_{i}"
        while alt_name in model_dfs:
            i += 1
            alt_name = f"{model_name}_{i}"
        print(f"Warning: multiple files for model '{model_name}', "
              f"storing additional as '{alt_name}' from {path.name}")
        model_name = alt_name

    model_dfs[model_name] = df

# === 4. (Optional) expose as df_cat, df_lgbm, df_rf, df_stacking, df_xgb ===
for name, df in model_dfs.items():
    globals()[f"df_{name}"] = df

# Quick check:
list(model_dfs.keys())test_prices = df_test[['high','low','close','y',volatility_col]].reset_index(drop=True)

ml_cls_summary, ml_trade_summary = evaluate_test_only_models(
    model_dfs=model_dfs,          # {"cat": df_cat, "lgbm": df_lgbm, ...}
    prices=test_prices,
    ku=6,
    kd=2,
    hold=336,
    tf="1h",
    atr_col="atr_200",
    y_col="y_true",
    proba_col="p_model",
    pred_col="y_pred",
    base_mrs=(0.0, 0.06, 0.1),
)

ml_cls_summary.to_csv(f"{DATA_DIR}/ml_cls_summary.csv")
ml_trade_summary.to_csv(f"{DATA_DIR}/ml_trade_summary.csv")ml_cls_summaryml_trade_summary