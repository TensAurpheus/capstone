
import numpy as np
def bss_metric(y_true: np.ndarray, probs: np.ndarray) -> float:
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
    return 1.0 - bs_model / bs_clim

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
