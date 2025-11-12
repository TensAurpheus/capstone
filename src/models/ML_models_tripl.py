# -*- coding: utf-8 -*-
"""
Узгоджений ModelTrainer для triple-barrier ML-моделей.
Метрики і profit повністю співпадають з DL та ML_models_tripl.py.
Сумісний з threshold tuning, optuna і shap.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import json
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier, Booster
from sklearn.model_selection import ParameterGrid
from sklearn.utils.class_weight import compute_class_weight
import datetime
import time

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    HAS_LGBM = True
except:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except:
    HAS_CAT = False

try:
    import optuna
    HAS_OPTUNA = True
except:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except:
    HAS_SHAP = False

RANDOM_STATE = 42

def triple_barrier_metrics(cm: np.ndarray):
    """
    cm: 3x3 confusion matrix (rows=true: 0=expiry,1=SL,2=TP; cols=pred)
    Returns precision/recall/F1 for TP (2) and SL (1), plus macro over {TP,SL}.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape != (3, 3):
        raise ValueError(
            "cm must be 3x3 with classes [0,1,2] = [expiry, SL, TP].")

    def prf(k: int):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    # Class 2 = TP-first, Class 1 = SL-first
    p_tp, r_tp, f1_tp = prf(2)
    p_sl, r_sl, f1_sl = prf(1)

    macro_precision = (p_tp + p_sl) / 2
    macro_recall = (r_tp + r_sl) / 2
    macro_f1 = (f1_tp + f1_sl) / 2

    return {
        "tp_precision": p_tp, "tp_recall": r_tp, "tp_f1": f1_tp,
        "sl_precision": p_sl, "sl_recall": r_sl, "sl_f1": f1_sl,
        "macro_precision_tp_sl": macro_precision,
        "macro_recall_tp_sl": macro_recall,
        "macro_f1_tp_sl": macro_f1,
    }

class ModelTrainer:
    def __init__(self, artifacts_dir: str | Path, k_profit: float = 2.0):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.k_profit = k_profit
        self.models: Dict[str, object] = {}
        self.records: List[Dict] = []

    # ---------- TRAIN BASELINES ----------
    def train_baselines(self, X_train, y_train, class_weights: Dict[int, float]):
        rf = RandomForestClassifier(
            n_estimators=400, random_state=RANDOM_STATE,
            class_weight=class_weights, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models["RandomForest"] = rf
        joblib.dump(rf, self.artifacts_dir / "RandomForest.pkl")

        if HAS_LGBM:
            lgbm = LGBMClassifier(
                objective="multiclass", num_class=3,
                n_estimators=700, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                class_weight=class_weights,
                random_state=RANDOM_STATE
            )
            lgbm.fit(X_train, y_train)
            self.models["LightGBM"] = lgbm
            lgbm.booster_.save_model(str(self.artifacts_dir / "LightGBM.txt"))

        if HAS_XGB:
            xgb = XGBClassifier(
                objective="multi:softprob", num_class=3,
                n_estimators=600, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                random_state=RANDOM_STATE, tree_method="hist"
            )
            xgb.fit(X_train, y_train)
            self.models["XGBoost"] = xgb
            xgb.save_model(str(self.artifacts_dir / "XGBoost.json"))

        if HAS_CAT:
            cb = CatBoostClassifier(
                loss_function="MultiClass",
                iterations=700, depth=6, learning_rate=0.05,
                random_seed=RANDOM_STATE,
                class_weights=list(class_weights.values()),
                verbose=False
            )
            cb.fit(X_train, y_train)
            self.models["CatBoost"] = cb
            cb.save_model(str(self.artifacts_dir / "CatBoost.cbm"))


    def load_baselines(self):
        for name, filename in [
            ("RandomForest", "RandomForest.pkl"),
            ("LightGBM", "LightGBM.txt"),
            ("XGBoost", "XGBoost.json"),
            ("CatBoost", "CatBoost.cbm")
        ]:
            path = self.artifacts_dir / filename
            if not path.exists():
                continue

            print(f"🔄 Loading {name} from disk...")

            if name == "RandomForest":
                self.models[name] = joblib.load(path)

            elif name == "LightGBM":
                # Restore core booster
                booster = Booster(model_file=str(path))
                model = LGBMClassifier()
                model._Booster = booster
                model.fitted_ = True
                model._n_features_in_ = booster.num_feature()
                self.models[name] = model

            elif name == "XGBoost":
                model = XGBClassifier()
                model.load_model(str(path))
                self.models[name] = model

            elif name == "CatBoost":
                model = CatBoostClassifier()
                model.load_model(str(path))
                self.models[name] = model      

    # ---------------------- Optuna LGBM ---------------------- #

    def train_optuna_lgbm(self, X_train, y_train, class_weights, n_trials=80):
        import optuna
        from sklearn.model_selection import StratifiedKFold
        from lightgbm import LGBMClassifier
        from sklearn.metrics import confusion_matrix
        import numpy as np
        import joblib

        def evaluate_profit(model, X, y, thr_tp, thr_sl):
            """Dual-threshold + profit score."""
            probas = model.predict_proba(X)
            pred_tp = (probas[:, 2] >= thr_tp).astype(int)  # сигнал "TP?"
            pred_sl = (probas[:, 1] >= thr_sl).astype(int)  # сигнал "SL?"
            
            # Фінальний клас:
            preds = np.where(pred_tp == 1, 2, np.where(pred_sl == 1, 1, 0))

            cm = confusion_matrix(y, preds, labels=[0,1,2])   
            profit = self.k_profit * cm[2,2] - (cm[1,2] + cm[0,2])
            return profit

        def objective(trial):
            # Оптимізуємо гіперпараметри моделі
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "class_weight": class_weights,
                "learning_rate": trial.suggest_float("lr", 0.03, 0.08),
                "n_estimators": trial.suggest_int("n_estimators", 400, 900),
                "max_depth": trial.suggest_int("max_depth", 4, 6),
                "num_leaves": trial.suggest_int("num_leaves", 20, 120),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "random_state": 42,
            }

            # Оптимізуємо також **пороги**
            thr_tp = trial.suggest_float("thr_tp", 0.15, 0.55)
            thr_sl = trial.suggest_float("thr_sl", 0.15, 0.55)

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            profits = []

            for tr_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

                model = LGBMClassifier(**params)
                model.fit(X_tr, y_tr)

                # Оцінюємо **прибуток**, а не accuracy/dF1
                profit = evaluate_profit(model, X_val, y_val, thr_tp, thr_sl)
                profits.append(profit)

            return np.mean(profits)

        print("🔎 Запуск Optuna (може зайняти 30-60 хв)...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        print("🏆 Найкращі параметри:", study.best_params)

        # Фіксуємо найкращу модель
        best = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            class_weight=class_weights,
            random_state=42,
            learning_rate=study.best_params["lr"],
            n_estimators=study.best_params["n_estimators"],
            max_depth=study.best_params["max_depth"],
            num_leaves=study.best_params["num_leaves"],
            subsample=study.best_params["subsample"],
            colsample_bytree=study.best_params["colsample_bytree"],
        )

        best.fit(X_train, y_train)

        # Зберігаємо модель та пороги
        self.models["LightGBM_Optuna"] = best
        joblib.dump(best, self.artifacts_dir / "LightGBM_Optuna.pkl")

        # Зберігаємо пороги у файл
        best_thresholds = {
            "thr_tp": study.best_params["thr_tp"],
            "thr_sl": study.best_params["thr_sl"],
        }
        joblib.dump(best_thresholds, self.artifacts_dir / "LightGBM_Optuna_thresholds.pkl")

        print("✅ Optuna завершено. Модель + пороги збережено.")
        return best


    # ---------- EVALUATE ----------
    def evaluate(self, model_name: str, X, y, split="val", variant="plain"):
        model = self.models[model_name]
        preds = model.predict(X)

        # побудова confusion matrix
        cm = confusion_matrix(y, preds, labels=[0, 1, 2])

        # triple-barrier F1/logits метрики
        m = triple_barrier_metrics(cm)

        # profit (TP дає прибуток, FP штрафуються)
        tp_true = cm[2, 2]
        fp_sl = cm[1, 2]   # передбачили TP, але це був SL
        fp_exp = cm[0, 2]  # передбачили TP, але була expiry/none
        profit = self.k_profit * tp_true - (fp_sl + fp_exp)

        rec = {
            "model": model_name,
            "variant": variant,
            "split": split,
            **m,
            "profit": float(profit),
        }
        self.records.append(rec)

        # збереження
        path = self.artifacts_dir / f"metrics_{model_name}_{variant}_{split}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

        return rec

    # ---------- THRESHOLD TUNING ----------
    def apply_dual_threshold_on_test(self, model, X_val, y_val, X_test, y_test, model_name="model"):
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Model does not support predict_proba()")

        proba_val = model.predict_proba(X_val)

        best_profit = -1e18
        best_thr_tp, best_thr_sl = 0.5, 0.5

        for thr_tp in np.linspace(0.3, 0.8, 26):
            for thr_sl in np.linspace(0.3, 0.8, 26):

                # Формуємо предикт
                preds_val = np.where(
                    proba_val[:,2] >= thr_tp, 2,
                    np.where(proba_val[:,1] >= thr_sl, 1, 0)
                )

                # ✅ Вважаємо cm → profit
                cm = confusion_matrix(y_val, preds_val, labels=[0,1,2])
                profit = self.k_profit * cm[2,2] - (cm[1,2] + cm[0,2])

                if profit > best_profit:
                    best_profit = profit
                    best_thr_tp, best_thr_sl = thr_tp, thr_sl

        # ---- застосовуємо оптимальні пороги на тесті ---- #
        proba_test = model.predict_proba(X_test)
        preds_test = np.where(
            proba_test[:,2] >= best_thr_tp, 2,
            np.where(proba_test[:,1] >= best_thr_sl, 1, 0)
        )

        cm_test = confusion_matrix(y_test, preds_test, labels=[0,1,2])
        m = triple_barrier_metrics(cm_test)
        profit_test = self.k_profit * cm_test[2,2] - (cm_test[1,2] + cm_test[0,2])

        rec = {
            "model": model_name, "variant": "dual-threshold", "split": "test",
            "thr_tp": best_thr_tp, "thr_sl": best_thr_sl,
            **m, "profit": float(profit_test)
        }
        self.records.append(rec)

        with open(self.artifacts_dir / f"metrics_{model_name}_dual-threshold_test.json", "w") as f:
            json.dump(rec, f, indent=2)

        return rec

    # ---------- LEADERBOARD ----------
    def build_leaderboard(self):
        if not self.records:
            return pd.DataFrame()

        df = pd.DataFrame(self.records)

        df = df.sort_values(
            ["split", "variant", "macro_f1_tp_sl", "profit"],
            ascending=[True, True, False, False]
        ).reset_index(drop=True)

        df.to_csv(self.artifacts_dir / "leaderboard_latest.csv", index=False)
        return df

    # ---------- UTILS ----------
    def _make_class_weights(self, y):
        classes = np.array([0,1,2])
        w = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        # dict: {0: w0, 1: w1, 2: w2}
        return {int(c): float(wi) for c, wi in zip(classes, w)}

    def _build_model(self, name: str, params: Dict, class_weights: Dict[int, float] | None):
        # Фабрика моделей з урахуванням клас-ваг
        if name == "RandomForest":
            base = {
                "n_estimators": 600,
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
                "class_weight": "balanced"
            }
            base.update(params)
            return RandomForestClassifier(**base)

        elif name == "LightGBM":
            assert HAS_LGBM, "LightGBM не встановлено"
            base = {
                "objective": "multiclass",
                "num_class": 3,
                "random_state": RANDOM_STATE,
            }
            if class_weights is not None:
                base["class_weight"] = class_weights
            base.update(params)
            return LGBMClassifier(**base)

        elif name == "XGBoost":
            assert HAS_XGB, "XGBoost не встановлено"
            base = {
                "objective": "multi:softprob",
                "num_class": 3,
                "random_state": RANDOM_STATE,
                "tree_method": "hist",
            }
            base.update(params)
            return XGBClassifier(**base)

        elif name == "CatBoost":
            assert HAS_CAT, "CatBoost не встановлено"
            base = {
                "loss_function": "MultiClass",
                "random_seed": RANDOM_STATE,
                "verbose": False
            }
            # CatBoost приймає список ваг у порядку класів
            if class_weights is not None:
                base["class_weights"] = [class_weights.get(i, 1.0) for i in [0,1,2]]
            base.update(params)
            return CatBoostClassifier(**base)

        else:
            raise ValueError(f"Невідома модель: {name}")

    def tune_on_val(
        self,
        model_name: str,
        X_train, y_train, X_val, y_val,
        param_grid: Dict[str, list],
        save_tag: str
    ):
        """
        Перебирає грід гіперпараметрів, тренує на train, обчислює tp_f1 на val,
        зберігає JSON з найкращими параметрами та метаданими.
        save_tag: суфікс у назві файлу (наприклад: 'ku2_kd1_hold48')
        """
        # class_weights тільки там, де підтримується
        class_weights = self._make_class_weights(y_train)

        best = {"tp_f1": -1.0, "params": None}
        best_cm = None

        for params in ParameterGrid(param_grid):
            model = self._build_model(model_name, params, class_weights)
            model.fit(X_train, y_train)

            preds_val = model.predict(X_val)
            cm = confusion_matrix(y_val, preds_val, labels=[0,1,2])
            m = triple_barrier_metrics(cm)
            score = m["tp_f1"]

            if score > best["tp_f1"]:
                best = {"tp_f1": float(score), "params": params}
                best_cm = cm

        # зберігаємо найкращі параметри
        payload = {
            "model": model_name,
            "save_tag": save_tag,
            "best_tp_f1": best["tp_f1"],
            "best_params": best["params"],
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "n_features": int(X_train.shape[1]),
        }
        out_path = self.artifacts_dir / f"best_params_{model_name}_{save_tag}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[OK] Збережено: {out_path}")

        return payload