import os
from pathlib import Path
import json
import math
import random
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import List, Tuple, Union, Optional, Callable, Dict, Any, Sequence
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# ------------------------- Utils -------------------------


def set_deterministic(seed: int = 42):
    import os, random, numpy as np, torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _select_device() -> Tuple[torch.device, bool]:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        return torch.device("cuda"), count > 1
    return torch.device("cpu"), False


def _ensure_dir(path: Union[str, Path]) -> Path:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _to_loader(
    data, batch_size: int, shuffle: bool, num_workers: int = 0
) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    return DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


# ------------------------- RNN Components -------------------------


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
        self.out = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, _ = self.rnn(inputs)
        last_timestep = outputs[:, -1, :]
        out = self.dropout(last_timestep)
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
        random_state: int = 42,
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
            return val_loss, None
        if not all_logits:
            return val_loss, 0.0
        logits = torch.cat(all_logits, dim=0)
        y_true = torch.cat(all_labels, dim=0).numpy()
        probs = torch.softmax(logits, dim=1).numpy()
        val_metric = metric_fn(y_true, probs)
        return val_loss, val_metric

    def train(
        self,
        train_data,
        val_data,
        *,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 5,
        model_path: Union[str, os.PathLike] = "artifacts/sequence_model.pt",
        loss_plot_path: Union[str, os.PathLike] = "artifacts/loss_curve.png",
        num_workers: int = 0,
        weights: Optional[Sequence[float]] = None,
        early_metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        early_metric_mode: str = "max",
    ) -> dict:
        set_deterministic(self.random_state)
        if weights is None:
            weights = np.ones(self.num_classes, dtype=np.float32)
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        train_losses: List[float] = []
        val_losses: List[float] = []
        val_metrics: List[Optional[float]] = []
        best_val_loss = float("inf")
        best_monitor: Optional[float] = None
        best_state: Optional[dict] = None
        bad_epochs = 0
        tol = 1e-6

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            sample_count = 0
            for features, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
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
            val_loss, val_metric = self._evaluate(
                val_loader, criterion, metric_fn=early_metric_fn
            )
            val_losses.append(val_loss)
            val_metrics.append(val_metric)
            if early_metric_fn is None:
                monitor_value = val_loss
                mode = "min"
            else:
                monitor_value = val_metric
                mode = early_metric_mode
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
        state_dict = torch.load(_ensure_dir(model_path), map_location=self.device)
        self._load_state_dict(state_dict)
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
        target = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )
        target.load_state_dict(state_dict)

    def _plot_losses(
        self,
        train_losses: Sequence[float],
        val_losses: Sequence[float],
        path: Union[str, os.PathLike],
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

    def predict(self, data, *, batch_size: int = 128):
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        preds, probas, y_true = [], [], []
        self.model.eval()
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                y_true.extend(y.view(-1).cpu().tolist())
                logits = self.model(X)
                probs = torch.softmax(logits, dim=1)
                probas.extend(probs.cpu().tolist())
                preds.extend(probs.argmax(dim=1).cpu().tolist())
        arr = np.asarray(probas)
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
        )


# ------------------------- Hybrid GRU-ML -------------------------


class _GRUEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
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
        _, h_n = self.gru(x)
        if self.bi == 2:
            z = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            z = h_n[-1]
        return z


class _GRUPretrainHead(nn.Module):
    def __init__(self, encoder: _GRUEncoder, out_classes: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        hdim = encoder.gru.hidden_size * (2 if encoder.gru.bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hdim, out_classes)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        return self.out(self.drop(z))


class _HybridGRUML:
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
        enc = _GRUEncoder(input_size, hidden_size, num_layers, dropout, bidirectional)
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
            use_pooling=True,
            pool_last_k=16,
            add_gru_proba=True,
        )

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
        model_path: Union[str, os.PathLike] = "artifacts/hybrid_gru_ml",
        loss_plot_path: Optional[Union[str, os.PathLike]] = None,
        use_gpu_in_ml: bool = True,
        weights=None,
        use_pooling: bool = True,
        pool_last_k: int = 16,
        add_gru_proba: bool = True,
        early_metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        early_metric_mode: str = "min",
    ) -> dict:
        set_deterministic(self.random_state)
        from pathlib import Path

        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        if loss_plot_path is None:
            loss_plot_path = model_dir / "loss_curve.png"
        self._meta["use_pooling"] = bool(use_pooling)
        self._meta["pool_last_k"] = int(pool_last_k)
        self._meta["add_gru_proba"] = bool(add_gru_proba)
        Ltr = _to_loader(train_data, batch_size, shuffle=True, num_workers=num_workers)
        Lva = _to_loader(val_data, batch_size, shuffle=False, num_workers=num_workers)
        w = (
            torch.tensor(weights, dtype=torch.float32, device=self.device)
            if weights is not None
            else None
        )
        crit = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_val_loss, best_metric, best_state, bad = float("inf"), None, None, 0
        tr_losses, va_losses = [], []
        for ep in range(1, epochs + 1):
            self.model.train()
            tr_loss, n = 0.0, 0
            for Xb, yb in tqdm(Ltr, desc=f"GRU pretrain {ep}/{epochs}", leave=False):
                Xb, yb = Xb.to(self.device), yb.view(-1).long().to(self.device)
                opt.zero_grad()
                logits = self.model(Xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                bs = yb.size(0)
                tr_loss += loss.item() * bs
                n += bs
            tr_losses.append(tr_loss / max(n, 1))
            self.model.eval()
            va_loss, n, all_probs, all_true = 0.0, 0, [], []
            with torch.no_grad():
                for Xb, yb in Lva:
                    Xb, yb = Xb.to(self.device), yb.view(-1).long().to(self.device)
                    logits = self.model(Xb)
                    loss = crit(logits, yb)
                    bs = yb.size(0)
                    va_loss += loss.item() * bs
                    n += bs
                    if early_metric_fn is not None:
                        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                        all_true.append(yb.cpu().numpy())
            va_loss /= max(n, 1)
            va_losses.append(va_loss)
            if early_metric_fn is not None:
                current_metric = (
                    float(
                        early_metric_fn(np.concatenate(all_true), np.vstack(all_probs))
                    )
                    if all_probs
                    else (float("inf") if early_metric_mode == "min" else float("-inf"))
                )
                improved = (
                    (current_metric < best_metric - 1e-6)
                    if (best_metric is not None and early_metric_mode == "min")
                    else (
                        current_metric > best_metric + 1e-6
                        if best_metric is not None
                        else True
                    )
                )
                if improved:
                    best_metric = current_metric
                    best_state = self._state_dict()
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            else:
                if va_loss < best_val_loss - 1e-6:
                    best_val_loss = va_loss
                    best_state = self._state_dict()
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            self._load_state_dict(best_state)
        self._plot_losses(tr_losses, va_losses, loss_plot_path)
        Ztr, ytr = self._extract_features(Ltr)
        Zva, yva = self._extract_features(Lva)
        self._ml = self._fit_ml(Ztr, ytr, Zva, yva, use_gpu_in_ml)
        self.save(model_dir)
        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": len(tr_losses),
            "loss_plot_path": str(loss_plot_path),
            "model_dir": str(model_dir),
            "best_early_metric": best_metric,
        }

    def predict(
        self, data, *, batch_size: int = 1024, num_workers: int = 0
    ) -> pd.DataFrame:
        Z, y = self._extract_features(
            _to_loader(data, batch_size, shuffle=False, num_workers=num_workers)
        )
        proba = self._predict_proba_ml(Z)
        df = pd.DataFrame(proba, columns=[f"p{i}" for i in range(proba.shape[1])])
        df.insert(0, "pred", proba.argmax(axis=1))
        df["true"] = y.astype(int)
        return df

    def save(self, model_dir: Union[str, os.PathLike]) -> None:
        from pathlib import Path

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._state_dict(), model_dir / "encoder.pt")
        joblib.dump(self._ml, model_dir / "ml_model.pkl")
        with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_dir: Union[str, os.PathLike]) -> "_HybridGRUML":
        from pathlib import Path

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
        obj._load_state_dict(
            torch.load(model_dir / "encoder.pt", map_location=obj.device)
        )
        obj._ml = joblib.load(model_dir / "ml_model.pkl")
        obj._meta = meta
        return obj

    def _state_dict(self) -> dict:
        return (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )

    def _load_state_dict(self, sd: dict) -> None:
        (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        ).load_state_dict(sd, strict=True)

    def _plot_losses(
        self, tr: Sequence[float], va: Sequence[float], path: Union[str, os.PathLike]
    ) -> None:
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
        self.model.eval()
        use_pool, k, add_prob = (
            self._meta.get("use_pooling", True),
            self._meta.get("pool_last_k", 16),
            self._meta.get("add_gru_proba", True),
        )
        enc = (
            self.model.module.encoder
            if isinstance(self.model, nn.DataParallel)
            else self.model.encoder
        )
        head = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )
        Z_list, Y_list = [], []
        with torch.no_grad():
            gru = enc.gru
            bi, H = (2 if gru.bidirectional else 1), gru.hidden_size
            for Xb, yb in loader:
                Xb = Xb.to(self.device)
                out, h_n = gru(Xb)
                z_last = torch.cat([h_n[-2], h_n[-1]], dim=1) if bi == 2 else h_n[-1]
                feats = [z_last]
                if use_pool and out.size(1) >= 1:
                    kk = min(k, out.size(1))
                    tail = out[:, -kk:, :]
                    feats += [tail.mean(dim=1), tail.max(dim=1)[0]]
                if add_prob:
                    feats += [torch.softmax(head(Xb), dim=-1)]
                Z_list.append(torch.cat(feats, dim=1).cpu().numpy())
                Y_list.append(yb.view(-1).cpu().numpy())
        return np.vstack(Z_list), np.concatenate(Y_list)

    def _fit_ml(self, Ztr, ytr, Zva, yva, use_gpu: bool):
        raise NotImplementedError

    def _predict_proba_ml(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GRUCatBoostClassifier(_HybridGRUML):
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
            Ztr,
            ytr,
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
            tree_method=(
                "gpu_hist" if (use_gpu and torch.cuda.is_available()) else "hist"
            ),
            random_state=self.random_state,
            n_jobs=0,
        )
        model.fit(
            Ztr, ytr, eval_set=[(Zva, yva)], early_stopping_rounds=200, verbose=False
        )
        return model

    def _predict_proba_ml(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(self._ml.predict_proba(Z))


# ------------------------- Hybrid LSTM-ML -------------------------


class _LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()
        self.bi = 2 if bidirectional else 1
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

    def forward(self, x: Tensor) -> Tensor:
        _, (h_n, c_n) = self.LSTM(x)
        return torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bi == 2 else h_n[-1]


class _LSTMPretrainHead(nn.Module):
    def __init__(self, encoder: _LSTMEncoder, out_classes: int, dropout: float = 0.0):
        super().__init__()
        self.encoder, self.drop = encoder, nn.Dropout(dropout)
        self.out = nn.Linear(
            encoder.LSTM.hidden_size * (2 if encoder.LSTM.bidirectional else 1),
            out_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.drop(self.encoder(x)))


class _HybridLSTMML:
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
        self.num_classes, self.random_state = num_classes, random_state
        device, use_parallel = _select_device()
        head = _LSTMPretrainHead(
            _LSTMEncoder(input_size, hidden_size, num_layers, dropout, bidirectional),
            out_classes=num_classes,
            dropout=dropout,
        )
        if use_parallel:
            head = nn.DataParallel(head)
        self.model, self.device, self.use_parallel = (
            head.to(device),
            device,
            use_parallel,
        )
        self._ml = None
        self._meta = dict(
            ml_kind=self.ML_KIND,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
            random_state=random_state,
            use_pooling=True,
            pool_last_k=16,
            add_LSTM_proba=True,
        )

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
        model_path: Union[str, os.PathLike] = "artifacts/hybrid_LSTM_ml",
        loss_plot_path: Optional[Union[str, os.PathLike]] = None,
        use_gpu_in_ml: bool = True,
        weights=None,
        use_pooling: bool = True,
        pool_last_k: int = 16,
        add_LSTM_proba: bool = True,
        early_metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        early_metric_mode: str = "min",
    ) -> dict:
        set_deterministic(self.random_state)
        from pathlib import Path

        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        if loss_plot_path is None:
            loss_plot_path = model_dir / "loss_curve.png"
        self._meta.update(
            {
                "use_pooling": bool(use_pooling),
                "pool_last_k": int(pool_last_k),
                "add_LSTM_proba": bool(add_LSTM_proba),
            }
        )
        Ltr = _to_loader(train_data, batch_size, shuffle=True, num_workers=num_workers)
        Lva = _to_loader(val_data, batch_size, shuffle=False, num_workers=num_workers)
        crit = nn.CrossEntropyLoss(
            weight=(
                torch.tensor(weights, dtype=torch.float32, device=self.device)
                if weights is not None
                else None
            )
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_val_loss, best_metric, best_state, bad = float("inf"), None, None, 0
        tr_losses, va_losses = [], []
        for ep in range(1, epochs + 1):
            self.model.train()
            tr_loss, n = 0.0, 0
            for Xb, yb in tqdm(Ltr, desc=f"LSTM pretrain {ep}/{epochs}", leave=False):
                Xb, yb = Xb.to(self.device), yb.view(-1).long().to(self.device)
                opt.zero_grad()
                logits = self.model(Xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                bs = yb.size(0)
                tr_loss += loss.item() * bs
                n += bs
            tr_losses.append(tr_loss / max(n, 1))
            self.model.eval()
            va_loss, n, all_probs, all_true = 0.0, 0, [], []
            with torch.no_grad():
                for Xb, yb in Lva:
                    Xb, yb = Xb.to(self.device), yb.view(-1).long().to(self.device)
                    logits = self.model(Xb)
                    loss = crit(logits, yb)
                    bs = yb.size(0)
                    va_loss += loss.item() * bs
                    n += bs
                    if early_metric_fn is not None:
                        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                        all_true.append(yb.cpu().numpy())
            va_loss /= max(n, 1)
            va_losses.append(va_loss)
            if early_metric_fn is not None:
                current_metric = (
                    float(
                        early_metric_fn(np.concatenate(all_true), np.vstack(all_probs))
                    )
                    if all_probs
                    else (float("inf") if early_metric_mode == "min" else float("-inf"))
                )
                improved = (
                    (current_metric < best_metric - 1e-6)
                    if (best_metric is not None and early_metric_mode == "min")
                    else (
                        current_metric > best_metric + 1e-6
                        if best_metric is not None
                        else True
                    )
                )
                if improved:
                    best_metric = current_metric
                    best_state = self._state_dict()
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            else:
                if va_loss < best_val_loss - 1e-6:
                    best_val_loss = va_loss
                    best_state = self._state_dict()
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            self._load_state_dict(best_state)
        self._plot_losses(tr_losses, va_losses, loss_plot_path)
        Ztr, ytr = self._extract_features(Ltr)
        Zva, yva = self._extract_features(Lva)
        self._ml = self._fit_ml(Ztr, ytr, Zva, yva, use_gpu_in_ml)
        self.save(model_dir)
        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": len(tr_losses),
            "loss_plot_path": str(loss_plot_path),
            "model_dir": str(model_dir),
            "best_early_metric": best_metric,
        }

    def predict(
        self, data, *, batch_size: int = 1024, num_workers: int = 0
    ) -> pd.DataFrame:
        Z, y = self._extract_features(
            _to_loader(data, batch_size, shuffle=False, num_workers=num_workers)
        )
        proba = self._predict_proba_ml(Z)
        df = pd.DataFrame(proba, columns=[f"p{i}" for i in range(proba.shape[1])])
        df.insert(0, "pred", proba.argmax(axis=1))
        df["true"] = y.astype(int)
        return df

    def save(self, model_dir: Union[str, os.PathLike]) -> None:
        from pathlib import Path

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._state_dict(), model_dir / "encoder.pt")
        joblib.dump(self._ml, model_dir / "ml_model.pkl")
        with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_dir: Union[str, os.PathLike]) -> "_HybridLSTMML":
        from pathlib import Path

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
        obj._load_state_dict(
            torch.load(model_dir / "encoder.pt", map_location=obj.device)
        )
        obj._ml = joblib.load(model_dir / "ml_model.pkl")
        obj._meta = meta
        return obj

    def _state_dict(self) -> dict:
        return (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )

    def _load_state_dict(self, sd: dict) -> None:
        (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        ).load_state_dict(sd, strict=True)

    def _plot_losses(
        self, tr: Sequence[float], va: Sequence[float], path: Union[str, os.PathLike]
    ) -> None:
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
        self.model.eval()
        use_pool, k, add_prob = (
            self._meta.get("use_pooling", True),
            self._meta.get("pool_last_k", 16),
            self._meta.get("add_LSTM_proba", True),
        )
        head = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )
        enc = head.encoder
        Z_list, Y_list = [], []
        with torch.no_grad():
            LSTM = enc.LSTM
            bi = 2 if LSTM.bidirectional else 1
            for Xb, yb in loader:
                Xb = Xb.to(self.device)
                out, (h_n, c_n) = LSTM(Xb)
                z_last = torch.cat([h_n[-2], h_n[-1]], dim=1) if bi == 2 else h_n[-1]
                feats = [z_last]
                if use_pool and out.size(1) >= 1:
                    kk = min(k, out.size(1))
                    tail = out[:, -kk:, :]
                    feats += [tail.mean(dim=1), tail.max(dim=1)[0]]
                if add_prob:
                    feats += [torch.softmax(head(Xb), dim=-1)]
                Z_list.append(torch.cat(feats, dim=1).cpu().numpy())
                Y_list.append(yb.view(-1).cpu().numpy())
        return np.vstack(Z_list), np.concatenate(Y_list)

    def _fit_ml(self, Ztr, ytr, Zva, yva, use_gpu: bool):
        raise NotImplementedError

    def _predict_proba_ml(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


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
            tree_method=(
                "gpu_hist" if (use_gpu and torch.cuda.is_available()) else "hist"
            ),
            random_state=self.random_state,
            n_jobs=0,
        )
        model.fit(
            Ztr, ytr, eval_set=[(Zva, yva)], early_stopping_rounds=200, verbose=False
        )
        return model

    def _predict_proba_ml(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(self._ml.predict_proba(Z))
