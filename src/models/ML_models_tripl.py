from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import optuna
import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

from src.utils.metrics import *
from src.utils.data_utils import *
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from collections import defaultdict

def find_best_threshold_dl(
    model,
    X_val,
    y_val,
    KU,
    KD,
    alpha=0.2,               # weight for tp_precision у combined_score
    thresholds=np.linspace(0.01, 0.99, 199),
):
    """
    Threshold selection:
      1) maximize macro_f1
      2) tie-breaker: maximize tp_precision
      3) fallback: maximize (macro_f1 + alpha * tp_precision)

    Returns:
       best_thr, best_metrics, best_combined_score
    """

    proba = model.predict_proba(X_val)[:, 1]

    best_thr = 0.5
    best_macro_f1 = -999
    best_tp_precision = -999
    best_combined = -999
    best_metrics = None

    for thr in thresholds:
        y_pred = (proba >= thr).astype(int)

        m = triple_barrier_metrics(
            y_true=y_val,
            y_pred=y_pred,
            p_all=proba,
            ku=KU,
            kd=KD,
        )

        macro_f1 = m["macro_f1"]
        tp_precision = m["tp_precision"]
        combined = macro_f1 + alpha * tp_precision

        better = False

        # -----------------
        # Threshold logic
        # -----------------
        if macro_f1 > best_macro_f1:
            better = True
        elif macro_f1 == best_macro_f1:
            if tp_precision > best_tp_precision:
                better = True
            elif tp_precision == best_tp_precision:
                if combined > best_combined:
                    better = True

        if better:
            best_thr = thr
            best_macro_f1 = macro_f1
            best_tp_precision = tp_precision
            best_combined = combined
            best_metrics = m

    return best_thr, best_metrics, best_combined



def tune_lightgbm_for_ku_kd(
    KU: float,
    KD: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    HOLD: int = 336,
    artifacts_dir: str = "artifacts",
    keep_ratios=[1.0],
):

    print(f"\n============== LightGBM (binary): KU={KU}, KD={KD}, HOLD={HOLD} ==============\n")

    y_train = np.asarray(y_train, dtype=int).ravel()
    y_val   = np.asarray(y_val,   dtype=int).ravel()

    # ---------------------------------------------------------
    # OPTIMIZED COMPACT GRID
    # ---------------------------------------------------------
    leaves_list      = [511, 1023]
    depth_list       = [20]
    n_list           = [3000]
    lr_list          = [0.005]
    subsample_list   = [0.9]
    colsample_list   = [0.6, 0.8]
    reg_alpha_list   = [0.0]
    reg_lambda_list  = [10.0, 20.0]
    min_child_list   = [1, 2]

    total = (
        len(keep_ratios)
        * len(leaves_list)
        * len(depth_list)
        * len(n_list)
        * len(lr_list)
        * len(subsample_list)
        * len(colsample_list)
        * len(reg_alpha_list)
        * len(reg_lambda_list)
        * len(min_child_list)
    )
    print(f"[INFO] Total LGBM models: {total}")

    best_macro_f1 = -999
    best_tp_precision = -999
    best_BSS = -999
    best_model = None
    best_params = None
    all_results = []

    model_n = 0

    for kr in keep_ratios:

        idx_0 = np.where(y_train == 0)[0]
        idx_1 = np.where(y_train == 1)[0]

        keep_n0 = int(len(idx_0) * kr)
        idx_0_down = np.random.choice(idx_0, keep_n0, replace=False)
        new_idx = np.concatenate([idx_0_down, idx_1])

        X_train_kr = X_train.iloc[new_idx]
        y_train_kr = y_train[new_idx]

        w_vec = make_utility_class_weights(y_train_kr, ku=KU, kd=KD, mode="balanced")
        classes = np.sort(np.unique(y_train_kr))
        class_weights = {int(c): float(w) for c, w in zip(classes, w_vec)}

        for leaves in leaves_list:
            for depth in depth_list:
                for n_estimators in n_list:
                    for lr in lr_list:
                        for subsample in subsample_list:
                            for colsample in colsample_list:
                                for alpha in reg_alpha_list:
                                    for lam in reg_lambda_list:
                                        for min_child in min_child_list:

                                            model_n += 1
                                            print(f"\n[LGBM MODEL {model_n}/{total}]")

                                            model = LGBMClassifier(
                                                objective="binary",
                                                learning_rate=lr,
                                                n_estimators=n_estimators,
                                                num_leaves=leaves,
                                                max_depth=depth,
                                                subsample=subsample,
                                                colsample_bytree=colsample,
                                                class_weight=class_weights,
                                                reg_alpha=alpha,
                                                reg_lambda=lam,
                                                min_child_samples=min_child,
                                                random_state=42,
                                                n_jobs=-1,
                                                verbose=-1,
                                            )

                                            model.fit(
                                                X_train_kr,
                                                y_train_kr,
                                                eval_set=[(X_val, y_val)],
                                                callbacks=[early_stopping(stopping_rounds=40, verbose=False)],
                                            )

                                            proba = model.predict_proba(X_val)[:, 1]
                                            y_pred = (proba >= 0.5).astype(int)

                                            m = triple_barrier_metrics(
                                                y_true=y_val,
                                                y_pred=y_pred,
                                                p_all=proba,
                                                ku=KU,
                                                kd=KD,
                                            )

                                            macro_f1 = m["macro_f1"]
                                            tp_precision = m["tp_precision"]
                                            BSS = m["bss"]

                                            entry = {
                                                "keep_ratio": kr,
                                                "n_estimators": n_estimators,
                                                "num_leaves": leaves,
                                                "max_depth": depth,
                                                "learning_rate": lr,
                                                "subsample": subsample,
                                                "colsample_bytree": colsample,
                                                "reg_alpha": alpha,
                                                "reg_lambda": lam,
                                                "min_child_samples": min_child,
                                                "macro_f1": float(macro_f1),
                                                "tp_precision": float(tp_precision),
                                                "BSS": float(BSS),
                                            }
                                            all_results.append(entry)

                                            better = False
                                            if macro_f1 > best_macro_f1:
                                                better = True
                                            elif macro_f1 == best_macro_f1:
                                                if tp_precision > best_tp_precision:
                                                    better = True
                                                elif tp_precision == best_tp_precision and BSS > best_BSS:
                                                    better = True

                                            if better:
                                                best_macro_f1 = macro_f1
                                                best_tp_precision = tp_precision
                                                best_BSS = BSS
                                                best_model = model
                                                best_params = entry.copy()

    # ---------------------------------------------------------
    # Threshold tuning (combined metric)
    # ---------------------------------------------------------
    best_thr, thr_metrics, best_combined = find_best_threshold_dl(
        best_model, X_val, y_val, KU, KD, alpha=0.2
    )

    best_params["threshold"] = best_thr
    best_params["combined_score"] = best_combined

    print(f"\n RF Best threshold = {best_thr:.3f}")
    print(f"  macro_f1={thr_metrics['macro_f1']:.5f}, tp_prec={thr_metrics['tp_precision']:.5f}")

    # Save artifacts
    SAVE_DIR = Path(artifacts_dir) / "lgbm"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"ku{KU}_kd{KD}_hold{HOLD}"

    joblib.dump(best_model, SAVE_DIR / f"best_lgbm_{tag}.pkl")

    with open(SAVE_DIR / f"best_lgbm_params_{tag}.json", "w") as f:
        json.dump(best_params, f, indent=2)

    pd.DataFrame(all_results).to_excel(SAVE_DIR / f"all_lgbm_results_{tag}.xlsx", index=False)
    with open(SAVE_DIR / f"all_lgbm_results_{tag}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n Best LGBM saved")
    print(" PARAMS:", best_params)

    return (KU, KD, thr_metrics["macro_f1"], thr_metrics["tp_precision"], best_params)


def tune_random_forest_for_ku_kd(
    KU: float,
    KD: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    HOLD: int = 336,
    artifacts_dir: str = "artifacts",
):
    print(f"\n============== RandomForest (binary): KU={KU}, KD={KD}, HOLD={HOLD} ==============\n")

    y_train = np.asarray(y_train, dtype=int).ravel()
    y_val   = np.asarray(y_val,   dtype=int).ravel()

    # ---------------------------------------------------------
    # class weights
    # ---------------------------------------------------------

    w_vec = make_utility_class_weights(y_train, ku=KU, kd=KD, mode="balanced")
    classes = np.sort(np.unique(y_train))
    class_weights = {int(c): float(w) for c, w in zip(classes, w_vec)}

    # ---------------------------------------------------------
    # GRID SEARCH
    # ---------------------------------------------------------

    n_list     = [450]
    depth_list = [ 15, 18]
    leaf_list  = [4, 6]

    best_macro_f1 = -999
    best_tp_precision = -999
    best_BSS = -999
    best_model = None
    best_params = None
    all_results = []

    model_n = 0
    total = len(n_list) * len(depth_list) * len(leaf_list)

    for n_est in n_list:
        for depth in depth_list:
            for leaf in leaf_list:

                model_n += 1
                print(f"[{model_n}/{total}] RF n_est={n_est}, depth={depth}, leaf={leaf}")

                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    max_features="sqrt",
                    class_weight=class_weights,
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)

                proba_val = model.predict_proba(X_val)[:, 1]
                y_pred_val = (proba_val >= 0.5).astype(int)

                metrics = triple_barrier_metrics(
                    y_true=y_val,
                    y_pred=y_pred_val,
                    p_all=proba_val,
                    ku=KU,
                    kd=KD,
                )

                macro_f1 = metrics["macro_f1"]
                tp_precision = metrics["tp_precision"]
                BSS = metrics["bss"]

                entry = {
                    "n_estimators": n_est,
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "macro_f1": float(macro_f1),
                    "tp_precision": float(tp_precision),
                    "BSS": float(BSS),
                }
                all_results.append(entry)

                better = False
                if macro_f1 > best_macro_f1:
                    better = True
                elif macro_f1 == best_macro_f1:
                    if tp_precision > best_tp_precision:
                        better = True
                    elif tp_precision == best_tp_precision and BSS > best_BSS:
                        better = True

                if better:
                    best_macro_f1 = macro_f1
                    best_tp_precision = tp_precision
                    best_BSS = BSS
                    best_model = model
                    best_params = entry.copy()

    # ---------------------------------------------------------
    # Threshold tuning for best model
    # ---------------------------------------------------------
    best_thr, thr_metrics, best_combined = find_best_threshold_dl(
        best_model, X_val, y_val, KU, KD, alpha=0.2
    )

    best_params["threshold"] = best_thr
    best_params["combined_score"] = best_combined

    print(f"\n RF Best threshold = {best_thr:.3f}")
    print(f"  macro_f1={thr_metrics['macro_f1']:.5f}, tp_prec={thr_metrics['tp_precision']:.5f}")

    # ---------------------------------------------------------
    # Save artifacts
    # ---------------------------------------------------------

    SAVE_DIR = Path(artifacts_dir) / "rf"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"ku{KU}_kd{KD}_hold{HOLD}"

    joblib.dump(best_model, SAVE_DIR / f"best_rf_{tag}.pkl")

    with open(SAVE_DIR / f"best_rf_params_{tag}.json", "w") as f:
        json.dump(best_params, f, indent=2)

    pd.DataFrame(all_results).to_excel(SAVE_DIR / f"all_rf_results_{tag}.xlsx", index=False)
    with open(SAVE_DIR / f"all_rf_results_{tag}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n Best RF saved")
    print(" PARAMS:", best_params)

    return (KU, KD, thr_metrics["macro_f1"], thr_metrics["tp_precision"], best_params)



def tune_xgb_for_ku_kd(
    KU: float,
    KD: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    HOLD: int = 336,
    artifacts_dir: str = "artifacts",
):

    print(f"\n============== XGBoost (binary): KU={KU}, KD={KD}, HOLD={HOLD} ==============\n")

    y_train = np.asarray(y_train, dtype=int).ravel()
    y_val   = np.asarray(y_val,   dtype=int).ravel()

    # -----------------------------------------
    # Class weights 
    # -----------------------------------------
    w_vec = make_utility_class_weights(y_train, ku=KU, kd=KD, mode="balanced")
    classes = np.sort(np.unique(y_train))
    class_weights = {int(c): float(w) for c, w in zip(classes, w_vec)}

    print("[INFO] Class weights:", class_weights)

    sample_weights = np.array([class_weights[int(c)] for c in y_train], dtype=float)

    # ---------------------------------------------------------
    # GRID SEARCH
    # ---------------------------------------------------------
    depths          = [3, 4]
    learning_rates  = [0.03, 0.05]
    estimators      = [2000, 3000]

    total = len(depths) * len(learning_rates) * len(estimators)
    print(f"[INFO] Total XGB models: {total}")

    best_BSS = -999.0
    best_recall = -999.0
    best_params = None
    best_model = None
    all_results = []

    model_n = 0

    for depth in depths:
        for lr in learning_rates:
            for estim in estimators:

                model_n += 1
                print(f"[{model_n}/{total}] depth={depth}, lr={lr}, n_estimators={estim}")

                model = XGBClassifier(
                    objective="binary:logistic",
                    max_depth=depth,
                    learning_rate=lr,
                    n_estimators=estim,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    tree_method="hist",
                    eval_metric="logloss",
                    early_stopping_rounds=60,
                    verbosity=0,
                )

                model.fit(
                    X_train,
                    y_train,
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                proba_val = model.predict_proba(X_val)[:, 1]
                y_pred = (proba_val >= 0.5).astype(int)

                m = triple_barrier_metrics(
                    y_true=y_val,
                    y_pred=y_pred,
                    p_all=proba_val,
                    ku=KU,
                    kd=KD,
                )

                BSS = m["bss"]
                tp_recall = m["tp_recall"]
                tp_precision = m["tp_precision"]
                tp_f1 = m["tp_f1"]

                print(
                    f"  → BSS={BSS:.5f}, "
                    f"TP_recall={tp_recall:.4f}, "
                    f"TP_precision={tp_precision:.4f}, "
                    f"TP_f1={tp_f1:.4f}"
                )

                entry = {
                    "max_depth": depth,
                    "learning_rate": lr,
                    "n_estimators": estim,
                    "BSS": float(BSS),
                    "tp_recall": float(tp_recall),
                    "tp_precision": float(tp_precision),
                    "tp_f1": float(tp_f1),
                    "best_iteration": int(model.best_iteration),
                }
                all_results.append(entry)

                is_better = False
                if BSS > best_BSS:
                    is_better = True
                elif BSS == best_BSS and tp_recall > best_recall:
                    is_better = True

                if is_better:
                    best_BSS = BSS
                    best_recall = tp_recall
                    best_model = model
                    best_params = entry

    # ---------------------------------------------------------
    # THRESHOLD TUNING
    # ---------------------------------------------------------
    best_thr, thr_metrics, best_comb = find_best_threshold_dl(
        best_model, X_val, y_val, KU, KD, alpha=0.2
    )

    best_params["threshold"] = best_thr
    best_params["combined_score"] = best_comb
    best_params["macro_f1"] = thr_metrics["macro_f1"]
    best_params["tp_precision"] = thr_metrics["tp_precision"]
    best_params["BSS"] = thr_metrics["bss"]

    print(f"\n XGB Best threshold = {best_thr:.3f}")
    print(f"  macro_f1={thr_metrics['macro_f1']:.5f}, tp_prec={thr_metrics['tp_precision']:.5f}")

    # ---------------------------------------------------------
    # SAVE ARTIFACTS
    # ---------------------------------------------------------
    SAVE_DIR = Path(artifacts_dir) / "xgb"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    tag = f"ku{KU}_kd{KD}_hold{HOLD}"

    joblib.dump(best_model, SAVE_DIR / f"best_xgb_{tag}.pkl")

    with open(SAVE_DIR / f"best_xgb_params_{tag}.json", "w") as f:
        json.dump(best_params, f, indent=2)

    with open(SAVE_DIR / f"all_xgb_results_{tag}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n XGBoost model & params saved to:", SAVE_DIR)

    return (KU, KD, best_BSS, best_recall, best_params)

def tune_catboost_for_ku_kd(
    KU: float,
    KD: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    HOLD: int = 336,
    n_trials: int = 30,
    no_improve_stop: int = 10,
    artifacts_dir: str = "artifacts",
):
    """
    Optuna tuning for CatBoost (binary classification).

    Optimization / selection priority:
        - During Optuna: maximize BSS (study.best_value)
        - For final best_trial selection:
              1) maximize BSS
              2) tie-breaker: maximize TP Recall
        - Threshold tuning: 
              1) macro_f1
              2) tp_precision
              3) combined = macro_f1 + alpha * tp_precision
    """

    print(f"\n============== CatBoost Optuna (binary): KU={KU}, KD={KD}, HOLD={HOLD} ==============\n")

    y_train = np.asarray(y_train, dtype=int).ravel()
    y_val   = np.asarray(y_val,   dtype=int).ravel()

    # ---------------------------------------------------------
    # Class weights (utility-aware)
    # ---------------------------------------------------------
    w_vec = make_utility_class_weights(y_train, ku=KU, kd=KD, mode="balanced")
    classes = np.sort(np.unique(y_train))
    class_weights = {int(c): float(w) for c, w in zip(classes, w_vec)}

    print("[INFO] Class weights:", class_weights)

    # CatBoost expects list of weights in class index order
    cb_class_weights = [class_weights[i] for i in sorted(class_weights.keys())]

    # ---------------------------------------------------------
    # Early stop for Optuna if no improvement in best BSS
    # ---------------------------------------------------------
    best_so_far = -999.0
    no_improve_counter = 0

    def early_stopping_callback(study, trial):
        nonlocal best_so_far, no_improve_counter

        if study.best_value > best_so_far:
            best_so_far = study.best_value
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if no_improve_counter >= no_improve_stop:
            print(f"\n⏹ Optuna early stopped after {no_improve_counter} non-improving trials\n")
            study.stop()

    # ---------------------------------------------------------
    # Optuna objective (maximize BSS)
    # ---------------------------------------------------------
    def objective(trial):

        params = {
            "loss_function": "Logloss",
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "iterations": trial.suggest_int("iterations", 300, 1200),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "random_seed": 42,
            "class_weights": cb_class_weights,
            "verbose": False,
        }

        model = CatBoostClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=80,
            verbose=False
        )

        proba_val = model.predict_proba(X_val)[:, 1]
        y_pred = (proba_val >= 0.5).astype(int)

        m = triple_barrier_metrics(
            y_true=y_val,
            y_pred=y_pred,
            p_all=proba_val,
            ku=KU,
            kd=KD,
        )

        BSS          = m["bss"]
        tp_recall    = m["tp_recall"]
        tp_precision = m["tp_precision"]
        tp_f1        = m["tp_f1"]
        macro_f1     = m["macro_f1"]

        print(
            f"Trial={trial.number}: "
            f"BSS={BSS:.5f}, "
            f"Recall={tp_recall:.4f}, "
            f"Precision={tp_precision:.4f}, "
            f"F1={tp_f1:.4f}, "
            f"macro_f1={macro_f1:.4f}"
        )

        # Save trial attributes
        trial.set_user_attr("BSS", float(BSS))
        trial.set_user_attr("tp_recall", float(tp_recall))
        trial.set_user_attr("tp_precision", float(tp_precision))
        trial.set_user_attr("tp_f1", float(tp_f1))
        trial.set_user_attr("macro_f1", float(macro_f1))
        trial.set_user_attr("model", model)

        return BSS

    # ---------------------------------------------------------
    # RUN OPTUNA
    # ---------------------------------------------------------
    print("Running Optuna...")
    study = optuna.create_study(direction="maximize")

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[early_stopping_callback],
        show_progress_bar=True
    )

    # ---------------------------------------------------------
    # Manual selection among trials:
    #   1) BSS
    #   2) tie-breaker: TP Recall
    # ---------------------------------------------------------
    best_trial = None
    best_BSS = -999.0
    best_recall = -999.0

    all_results = []

    for t in study.trials:
        # can be None
        if "BSS" not in t.user_attrs:
            continue

        BSS          = t.user_attrs["BSS"]
        tp_recall    = t.user_attrs["tp_recall"]
        tp_precision = t.user_attrs["tp_precision"]
        tp_f1        = t.user_attrs["tp_f1"]
        macro_f1     = t.user_attrs["macro_f1"]

        entry = {
            "trial_number": t.number,
            "BSS": float(BSS),
            "tp_recall": float(tp_recall),
            "tp_precision": float(tp_precision),
            "tp_f1": float(tp_f1),
            "macro_f1": float(macro_f1),
        }
        # add trial parameters
        entry.update(t.params)
        all_results.append(entry)

        is_better = False
        if BSS > best_BSS:
            is_better = True
        elif BSS == best_BSS and tp_recall > best_recall:
            is_better = True

        if is_better:
            best_BSS = BSS
            best_recall = tp_recall
            best_trial = t

    assert best_trial is not None, "No successful trials in CatBoost Optuna."

    best_params = best_trial.params
    best_model = best_trial.user_attrs["model"]

    print("\n Best CatBoost Optuna Trial (BSS/Recall):")
    print(json.dumps(best_params, indent=2))
    print(f"BSS = {best_trial.user_attrs['BSS']:.5f}")
    print(f"TP Recall = {best_trial.user_attrs['tp_recall']:.4f}")
    print(f"TP Precision = {best_trial.user_attrs['tp_precision']:.4f}")
    print(f"TP F1 = {best_trial.user_attrs['tp_f1']:.4f}")
    print(f"macro_f1 = {best_trial.user_attrs['macro_f1']:.4f}")

    final_metrics = {
        "BSS":         best_trial.user_attrs["BSS"],
        "tp_recall":   best_trial.user_attrs["tp_recall"],
        "tp_precision":best_trial.user_attrs["tp_precision"],
        "tp_f1":       best_trial.user_attrs["tp_f1"],
        "macro_f1":    best_trial.user_attrs["macro_f1"],
    }

    # ---------------------------------------------------------
    # THRESHOLD TUNING 
    # ---------------------------------------------------------
    best_thr, thr_metrics, best_combined = find_best_threshold_dl(
        best_model,
        X_val,
        y_val,
        KU,
        KD,
        alpha=0.2,
    )

    print(f"\n CatBoost Best threshold = {best_thr:.3f}")
    print(f"  macro_f1={thr_metrics['macro_f1']:.5f}, tp_prec={thr_metrics['tp_precision']:.5f}")

    final_metrics_thr = {
        "BSS":         thr_metrics["bss"],
        "tp_recall":   thr_metrics["tp_recall"],
        "tp_precision":thr_metrics["tp_precision"],
        "tp_f1":       thr_metrics["tp_f1"],
        "macro_f1":    thr_metrics["macro_f1"],
        "threshold":   best_thr,
        "combined_score": best_combined,
    }

    # add threshold and combined_score
    best_params_extended = best_params.copy()
    best_params_extended["threshold"] = best_thr
    best_params_extended["combined_score"] = best_combined

    # ---------------------------------------------------------
    # SAVE best model & artifacts
    # ---------------------------------------------------------
    save_dir = Path(artifacts_dir) / "cat"
    save_dir.mkdir(parents=True, exist_ok=True)

    tag = f"ku{KU}_kd{KD}_hold{HOLD}"

    best_model.save_model(str(save_dir / f"best_catboost_{tag}.cbm"))

    with open(save_dir / f"best_catboost_params_{tag}.json", "w") as f:
        json.dump(
            {
                "ku": KU,
                "kd": KD,
                "hold": HOLD,
                "best_metrics_raw": final_metrics,          # without threshold-tuning
                "best_metrics_dl_threshold": final_metrics_thr,  # after threshold
                "best_params": best_params_extended,
            },
            f,
            indent=2,
        )

    with open(save_dir / f"catboost_all_results_{tag}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n CatBoost Optuna saved → {save_dir}")

    return KU, KD, final_metrics_thr, best_params_extended


def data_pipe(df, ku, kd, hold, window_size, volatility_col='atr_200'):

    # ---------------------------------------------------------
    # 1) Triple-barrier labeling: 3-class y in {0,1,2} 
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 2) Split + scale with 3-class labels (no binarization yet) 
    # ---------------------------------------------------------

    df_train, df_val, df_test, scaler = split_scale(
        df_labeled,
        target_cols=target,
        scale=True,
        volatility=volatility_col,
    )

    # ---------------------------------------------------------
    # 3) One neat table of 3-class label distribution per split 
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 4) Binarize: TP (=2) vs not-TP (0 or 1), keep original 3-class as y_3c
    # ---------------------------------------------------------

    for d in (df_train, df_val, df_test):
        # d['y_3c'] = d['y'].copy()
        d['y'] = (d['y'] == 2).astype(int)

    return df_train, df_val, df_test, scaler


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

EXCLUDE_FEATURES = [
    "open", "high", "low", "close", "atr_200",
]

def drop_exclude_features(df):
    return df.drop(columns=EXCLUDE_FEATURES, errors="ignore")


def prepare_split_for_ku_kd(
    df: pd.DataFrame,
    KU: float,
    KD: float,
    HOLD: int = 336,
    save_dir: str = "artifacts/splits",
    force: bool = False,
    volatility_col: str = "atr_200",
    window_size: int = 336     # ← ДОДАЛИ
):
    """
    Read or make train/val/test split for KU/KD/HOLD.
    Returns train/val/test splits with binarized y.

    Binary target:
        1 = TP (previous label 2)
        0 = NOT TP (expiry or SL)
    """

    save_dir = Path(save_dir) / f"ku{KU}_kd{KD}_hold{HOLD}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # parquet files
    train_p = save_dir / "train.parquet"
    val_p   = save_dir / "val.parquet"
    test_p  = save_dir / "test.parquet"
    scaler_p = save_dir / "scaler.pkl"

    # ----------------------------------------------------------------------
    # 1) if train/val/test splits exist, read them
    # ----------------------------------------------------------------------
    if train_p.exists() and val_p.exists() and test_p.exists() and not force:
        print(f"\n Use existing splits KU={KU}, KD={KD}, HOLD={HOLD}")
        print(" Read splits:")

        print("   🟢", train_p)
        print("   🟢", val_p)
        print("   🟢", test_p)

        train_df = pd.read_parquet(train_p)
        val_df   = pd.read_parquet(val_p)
        test_df  = pd.read_parquet(test_p)

        try:
            scaler = joblib.load(scaler_p)
        except:
            scaler = None

    else:
        # ------------------------------------------------------------------
        # 2) Create train/val/test splits
        # ------------------------------------------------------------------
        print(f"\n Create splits KU={KU}, KD={KD}, HOLD={HOLD}")

        df_labeled = triple_barrier_label(
            df,
            close="close",
            high="high",
            low="low",
            volatility=volatility_col,
            ku=KU,
            kd=KD,
            hold=HOLD,
            debug=False,
        )

        # ------------------------------------------------------------------
        # 3) Split+Scale 
        # ------------------------------------------------------------------
        train_df, val_df, test_df, scaler = split_scale(
            df_labeled,
            target_cols=["y"],
            scale=True,
            volatility=volatility_col,
            test_size=0.09,
            val_size=0.09,
        )

        # ------------------------------------------------------------------
        # 4) Save splits
        # ------------------------------------------------------------------
        train_df.to_parquet(train_p)
        val_df.to_parquet(val_p)
        test_df.to_parquet(test_p)
        joblib.dump(scaler, scaler_p)

        print(f" Saved splits to {save_dir}")

    # ----------------------------------------------------------------------
    # 5) Binarize y (1=TP, 0=NOT TP)
    # ----------------------------------------------------------------------
    for d in (train_df, val_df, test_df):
        d["y"] = (d["y"] == 2).astype(int)

    # ----------------------------------------------------------------------
    # 6) split into X, y (SYNC WITH DL)
    # ----------------------------------------------------------------------

    # remove first window_size rows
    train_df = train_df.iloc[window_size:].reset_index(drop=True)
    #val_df   = val_df.iloc[window_size:].reset_index(drop=True)
    #test_df  = test_df.iloc[window_size:].reset_index(drop=True)

    X_train, y_train = train_df.drop(columns=["y"]), train_df["y"]
    X_val,   y_val   = val_df.drop(columns=["y"]),   val_df["y"]
    X_test,  y_test  = test_df.drop(columns=["y"]),  test_df["y"]


    # ----------------------------------------------------------------------
    # 7) remove price-based features
    # ----------------------------------------------------------------------
    X_train = drop_exclude_features(X_train)
    X_val   = drop_exclude_features(X_val)
    X_test  = drop_exclude_features(X_test)

    # ----------------------------------------------------------------------
    # 8) print stats
    # ----------------------------------------------------------------------
    n_total = len(train_df) + len(val_df) + len(test_df)
    print(f"\n Split stats:")
    print(f"   Train: {len(train_df):>6} | Val: {len(val_df):>6} | Test: {len(test_df):>6} | Total: {n_total}")

    print("\n Binary label distribution (TP rate):")
    print(" Train:", train_df["y"].value_counts(normalize=True).round(4).to_dict())
    print(" Val:  ", val_df["y"].value_counts(normalize=True).round(4).to_dict())
    print(" Test: ", test_df["y"].value_counts(normalize=True).round(4).to_dict())

    return X_train, y_train, X_val, y_val, X_test, y_test

def run_full_binary_pipeline(
    df,
    KU_list,
    KD_list,
    HOLD=336,
    save_results="artifacts/results_binary"
):
    Path(save_results).mkdir(parents=True, exist_ok=True)

    results = [] 

    for KU in KU_list:
        for KD in KD_list:

            print("\n" + "="*80)
            print(f" START: KU={KU}, KD={KD}, HOLD={HOLD}")
            print("="*80)

            # ---------------------------------------------------------
            # 1) DATA SPLIT
            # ---------------------------------------------------------
            X_train, y_train, X_val, y_val, X_test, y_test = prepare_split_for_ku_kd(
                df,
                KU=KU,
                KD=KD,
                HOLD=HOLD,
                force=False
            )

            # CatBoost safety
            for d in [X_train, X_val, X_test, y_train, y_val, y_test]:
                d.reset_index(drop=True, inplace=True)

            # ---------------------------------------------------------
            # 2) RUN ALL MODELS
            # ---------------------------------------------------------
            all_models = {}

            all_models["RF"] = tune_random_forest_for_ku_kd(
                KU, KD, X_train, y_train, X_val, y_val, HOLD
            )

            all_models["LGBM"] = tune_lightgbm_for_ku_kd(
                KU, KD, X_train, y_train, X_val, y_val, HOLD
            )

            all_models["XGB"] = tune_xgb_for_ku_kd(
                KU, KD, X_train, y_train, X_val, y_val, HOLD
            )

            all_models["CAT"] = tune_catboost_for_ku_kd(
                KU, KD, X_train, y_train, X_val, y_val, HOLD
            )

            # ---------------------------------------------------------
            # 3) VAL + TEST evaluation
            # ---------------------------------------------------------
            test_records = []

            for model_name, model_output in all_models.items():

                best_params = model_output[-1]
                thr = best_params.get("threshold", 0.5)   # default threshold

                # Load model
                if model_name == "RF":
                    model = joblib.load(f"artifacts/rf/best_rf_ku{KU}_kd{KD}_hold{HOLD}.pkl")
                elif model_name == "LGBM":
                    model = joblib.load(f"artifacts/lgbm/best_lgbm_ku{KU}_kd{KD}_hold{HOLD}.pkl")
                elif model_name == "XGB":
                    model = joblib.load(f"artifacts/xgb/best_xgb_ku{KU}_kd{KD}_hold{HOLD}.pkl")
                elif model_name == "CAT":
                    model = CatBoostClassifier()
                    model.load_model(f"artifacts/cat/best_catboost_ku{KU}_kd{KD}_hold{HOLD}.cbm")

                # VAL
                proba_val = model.predict_proba(X_val)[:, 1]
                y_pred_val = (proba_val >= thr).astype(int)

                m_val = triple_barrier_metrics(
                    y_true=y_val,
                    y_pred=y_pred_val,
                    p_all=proba_val,
                    ku=KU,
                    kd=KD
                )

                # TEST
                proba_test = model.predict_proba(X_test)[:, 1]
                y_pred_test = (proba_test >= thr).astype(int)

                m_test = triple_barrier_metrics(
                    y_true=y_test,
                    y_pred=y_pred_test,
                    p_all=proba_test,
                    ku=KU,
                    kd=KD
                )

                macro_gap = abs(m_test["macro_f1"] - m_val["macro_f1"])
                precision_gap = abs(m_test["tp_precision"] - m_val["tp_precision"])
                bss_gap = abs(m_test["bss"] - m_val["bss"])

                test_records.append({
                    "KU": KU,
                    "KD": KD,
                    "model": model_name,
                    "threshold_used": thr,

                    # VAL
                    "macro_f1_val": m_val["macro_f1"],
                    "tp_precision_val": m_val["tp_precision"],
                    "tp_recall_val": m_val["tp_recall"],
                    "tp_f1_val": m_val["tp_f1"],
                    "BSS_val": m_val["bss"],

                    # TEST
                    "macro_f1_test": m_test["macro_f1"],
                    "tp_precision_test": m_test["tp_precision"],
                    "tp_recall_test": m_test["tp_recall"],
                    "tp_f1_test": m_test["tp_f1"],
                    "BSS_test": m_test["bss"],

                    # Stability
                    "macro_f1_gap": macro_gap,
                    "precision_gap": precision_gap,
                    "bss_gap": bss_gap,
                })


            # ---------------------------------------------------------
            # 4) SAVE FULL TABLE
            # ---------------------------------------------------------
            df_test_full = pd.DataFrame(test_records)
            df_test_full.to_excel(
                f"{save_results}/all_models_KU{KU}_KD{KD}.xlsx",
                index=False
            )

            print("\n📊 FULL VAL + TEST RESULTS FOR THIS KU/KD:")
            print(df_test_full)

            # ---------------------------------------------------------
            # 5) SELECT BEST MODEL WITH NEW CRITERIA
            # ---------------------------------------------------------
            df_sorted = df_test_full.sort_values(
                [
                    "macro_f1_test",     # 1) main metric
                    "tp_precision_test", # 2) quality
                    "BSS_test",          # 3) calibration
                    "macro_f1_gap"       # 4) stability
                ],
                ascending=[False, False, False, True]
            )

            best_row = df_sorted.iloc[0]

            # ---------------------------------------------------------
            # 5) SELECT BEST MODEL WITH NEW CRITERIA
            # ---------------------------------------------------------
            df_sorted = df_test_full.sort_values(
                [
                    "macro_f1_test",     # 1) main metric
                    "tp_precision_test", # 2) quality
                    "BSS_test",          # 3) calibration
                    "macro_f1_gap"       # 4) stability
                ],
                ascending=[False, False, False, True]
            )

            best_row = df_sorted.iloc[0]

            best_entry = {
                "KU": KU,
                "KD": KD,
                "best_model": best_row["model"],

                # --- METRICS (TEST) ---
                "best_macro_f1_test": best_row["macro_f1_test"],
                "best_precision_test": best_row["tp_precision_test"],
                "best_recall_test": best_row["tp_recall_test"],
                "best_BSS_test": best_row["BSS_test"],

                # --- STABILITY ---
                "stability_macro_gap": best_row["macro_f1_gap"],
                "stability_precision_gap": abs(best_row["tp_precision_test"] - best_row["tp_precision_val"]),
                "stability_bss_gap": abs(best_row["BSS_test"] - best_row["BSS_val"]),

                # --- THRESHOLD (якщо є) ---
                "threshold_used": best_row["threshold_used"]
            }

            results.append(best_entry)

            print("\n🔥 BEST MODEL FOR THIS KU/KD:")
            print(best_entry)

    # ---------------------------------------------------------
    # 6) Save summary
    # ---------------------------------------------------------
    df_results = pd.DataFrame(results)
    df_results.to_excel(f"{save_results}/summary_binary.xlsx", index=False)

    print("\n COMPLETE! Results saved:")
    print(f"   {save_results}/summary_binary.xlsx")

    return df_results


def dl_threshold_for_probs(proba, y_true, KU, KD, thresholds=None):
    """
    treshold selection:
      1) max macro_f1
      2) при рівності → max tp_precision
      3) при рівності → max BSS
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 61)

    best_thr = 0.5
    best_macro = -999
    best_prec = -999
    best_bss = -999
    best_m = None

    for thr in thresholds:
        y_pred = (proba >= thr).astype(int)

        m = triple_barrier_metrics(
            y_true=y_true,
            y_pred=y_pred,
            p_all=proba,
            ku=KU,
            kd=KD,
        )

        macro = m["macro_f1"]
        prec  = m["tp_precision"]
        bss   = m["bss"]

        better = False
        if macro > best_macro:
            better = True
        elif macro == best_macro:
            if prec > best_prec:
                better = True
            elif prec == best_prec and bss > best_bss:
                better = True

        if better:
            best_macro = macro
            best_prec  = prec
            best_bss   = bss
            best_thr   = thr
            best_m     = m

    print(f"\n Best threshold = {best_thr:.3f}")
    print(f"  macro_f1={best_macro:.5f}, tp_precision={best_prec:.5f}, BSS={best_bss:.5f}")

    return best_thr, best_m