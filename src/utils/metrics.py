import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def f1_metric(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_pred = (probs[:, 1] >= 0.5).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def brier_metric(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Brier score for binary classification."""
    y_true = y_true.astype(float)
    p_hat = probs[:, 1].astype(float)  # prob of class 1
    return float(np.mean((p_hat - y_true) ** 2))


def bss_metric(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Brier Skill Score for binary TP vs not-TP.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(probs, dtype=float)
    if p.ndim == 2:
        if p.shape[1] != 2:
            raise ValueError(
                f"bss_metric: expected probs shape (N,2) for binary, got {p.shape}"
            )
        p = p[:, 1]
    elif p.ndim == 1:
        pass
    else:
        raise ValueError(f"bss_metric: probs must be 1D or 2D, got ndim={p.ndim}")

    p = np.clip(p.ravel(), 0.0, 1.0)
    if p.shape[0] != y.shape[0]:
        raise ValueError(f"bss_metric: shape mismatch y={y.shape}, p={p.shape}")

    bs_model = np.mean((p - y) ** 2)
    pi = y.mean()
    bs_clim = np.mean((pi - y) ** 2)

    if bs_clim <= 0:
        return 0.0

    return 1.0 - bs_model / bs_clim


def triple_barrier_metrics(
    *,
    y_true,
    y_pred,
    p_all,
    ku: float,
    kd: float,
) -> dict:
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    if isinstance(p_all, pd.DataFrame):
        if p_all.shape[1] != 1:
            # If multi-column, assume it's [p0, p1, p2] and we want p2 (TP)
            if "p2" in p_all.columns:
                p_tp = p_all["p2"].to_numpy(dtype=float).ravel()
            else:
                raise ValueError(
                    "For binary metrics, p_all must have exactly one column = P(TP) or a 'p2' column."
                )
        else:
            p_tp = p_all.iloc[:, 0].to_numpy(dtype=float).ravel()
    else:
        p_tp = np.asarray(p_all, dtype=float).ravel()

    if not (len(y_true) == len(y_pred) == len(p_tp)):
        raise ValueError("y_true, y_pred and p_tp must have the same length.")

    tp_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    tp_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    tp_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    brier = np.mean((p_tp - y_true) ** 2)
    bss = bss_metric(y_true, p_tp)

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


def build_model_metrics_df(all_results: dict) -> pd.DataFrame:
    """Flatten all_results into a DF with classification metrics."""
    rows = []
    for (tf, es_metric), info in all_results.items():
        # DL models: val + test
        for split_name, split_key in [("val", "val_results"), ("test", "test_results")]:
            split_res = info.get(split_key, {})
            for model_type, res_model in split_res.items():
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
        # Baseline (logreg)
        for split_name, base_key in [
            ("val", "baseline_val"),
            ("test", "baseline_test"),
        ]:
            base_res = info.get(base_key, {})
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
    return pd.DataFrame(rows)
