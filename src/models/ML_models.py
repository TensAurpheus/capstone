from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..data.data_utils import split_scale, DataScaler


def prepare_data(
    parquet_path: str,
    *,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "target",
    test_size: float = 0.2,
    val_size: float = 0.1,
    scale: bool = True,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    scaler_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load parquet file, split into train/val/test, and apply scaling.
    Automatically assigns unique scaler path per symbol/timeframe.
    """

    frame = pd.read_parquet(parquet_path)

    if feature_cols is None:
        feature_cols = [col for col in frame.columns if col != target_col]

    # --- Determine scaler path dynamically ---
    if scaler_path is None:
        if symbol:
            scaler_name = f"standard_scaler_{symbol.replace('/', '_')}_{timeframe}"
            if timeframe:
                scaler_name += f"_{timeframe}"
            scaler_path = f"data/model/scalers/{scaler_name}.pkl"
        else:
            scaler_path = "data/model/scalers/standard_scaler_generic.pkl"

    print(f"[INFO] Using scaler path: {scaler_path}")

    # === Split + Scale ===
    train_df, val_df, test_df, scaler = split_scale(
        frame,
        target_cols=target_col,
        test_size=test_size,
        val_size=val_size,
        scale=scale,
        scaler_path=scaler_path,
    )

    timestamps = frame["timestamp"] if "timestamp" in frame.columns else None

    def _to_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {
            "X": df[feature_cols].to_numpy(dtype=np.float32),
            "y": df[target_col if target_col in df.columns else "target"].to_numpy(dtype=np.float32),
        }

    print("[OK] Data prepared successfully.")
    return {
        "train": _to_arrays(train_df),
        "val": _to_arrays(val_df),
        "test": _to_arrays(test_df),
        "feature_cols": feature_cols,
        "timestamps": timestamps,
        "scaler": scaler,
    }


# === MODEL BUILDERS ===

def _build_model(name: str, params: Dict[str, Any]) -> Any:
    name = name.lower()
    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CatBoost is required for the 'catboost' model. Install it with 'pip install catboost'."
            ) from exc

        defaults = {
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 500,
            "l2_leaf_reg": 3.0,
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": False,
        }
        defaults.update(params)
        return CatBoostRegressor(**defaults)

    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LightGBM is required for the 'lightgbm' model. Install it with 'pip install lightgbm'."
            ) from exc

        defaults = {"n_estimators": 500, "learning_rate": 0.05, "random_state": 42}
        defaults.update(params)
        return LGBMRegressor(**defaults)

    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "XGBoost is required for the 'xgboost' model. Install it with 'pip install xgboost'."
            ) from exc

        defaults = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "objective": "reg:squarederror",
        }
        defaults.update(params)
        return XGBRegressor(**defaults)

    if name == "random_forest":
        defaults = {"n_estimators": 300, "max_depth": None, "random_state": 42}
        defaults.update(params)
        return RandomForestRegressor(**defaults)

    raise ValueError(f"Unsupported model name: {name}")


# === METRICS & HELPERS ===

def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _start_progress(total: int, desc: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(total=total, desc=desc, leave=False)
    return None


def _apply_progress_defaults(name: str, params: Dict[str, Any], enabled: bool) -> None:
    if not enabled:
        return

    defaults: Dict[str, Dict[str, Any]] = {
        "catboost": {"verbose": 100},
        "lightgbm": {"verbosity": 1},
        "xgboost": {"verbosity": 1},
        "random_forest": {"verbose": 1},
    }

    for key, value in defaults.get(name.lower(), {}).items():
        params.setdefault(key, value)


def _extract_feature_importances(model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
    """Return a mapping of feature name to importance if the model provides it."""
    importance_values: Optional[np.ndarray] = None

    if hasattr(model, "feature_importances_"):
        importance_values = np.asarray(getattr(model, "feature_importances_", None))
    elif hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_", None))
        if coef.ndim == 1:
            importance_values = np.abs(coef)
        elif coef.ndim == 2 and coef.shape[0] == 1:
            importance_values = np.abs(coef[0])
    elif hasattr(model, "get_feature_importance"):
        try:
            importance_values = np.asarray(model.get_feature_importance())
        except TypeError:  # pragma: no cover - API differences
            importance_values = None

    if importance_values is None or importance_values.size != len(feature_names):
        return None

    return dict(zip(feature_names, importance_values.tolist()))


# === MAIN TRAINING ENTRY ===

def train_and_predict(
    model_name: str,
    data: Dict[str, Any],
    *,
    predict: bool = True,
    model_params: Optional[Dict[str, Any]] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Train the requested model and optionally return predictions and metrics."""

    model_params = dict(model_params or {})
    _apply_progress_defaults(model_name, model_params, progress)

    model = _build_model(model_name, model_params)

    train_data = data["train"]

    progress_bar = _start_progress(1, f"{model_name} fit", progress)

    if progress_bar is not None:
        progress_bar.update(0)
    model.fit(train_data["X"], train_data["y"])
    if progress_bar is not None:
        progress_bar.update(1)

    result: Dict[str, Any] = {"model": model}

    feature_importances = _extract_feature_importances(model, data["feature_cols"])
    if feature_importances is not None:
        result["feature_importances"] = feature_importances

    if predict:
        val_preds = model.predict(data["val"]["X"])
        test_preds = model.predict(data["test"]["X"])

        result["predictions"] = {"val": val_preds, "test": test_preds}
        result["metrics"] = {
            "val": _evaluate_regression(data["val"]["y"], val_preds),
            "test": _evaluate_regression(data["test"]["y"], test_preds),
        }

    if progress_bar is not None:
        progress_bar.close()

    return result