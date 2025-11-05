from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


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
        self.fc = nn.Linear(
            hidden_size * (2 if bidirectional else 1), num_classes*2)
        self.out = nn.Linear(num_classes*2, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, _ = self.rnn(inputs)
        last_timestep = outputs[:, -1, :]
        out = self.dropout(last_timestep)
        out = self.fc(out)
        out = torch.relu(out)
        out = self.out(out)
        # out = torch.softmax(out, dim=1)

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
    ) -> None:
        device, use_parallel = _select_device()
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

    # ------------------------------------------------------------------ training
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
    ) -> dict:

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

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_val = float("inf")
        best_state: Optional[dict] = None
        bad_epochs = 0
        train_losses: List[float] = []
        val_losses: List[float] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            sample_count = 0
            for features, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                features = features.to(self.device)
                labels = labels.view(-1).long()
                labels = labels.to(self.device)
                optimizer.zero_grad()
                preds = self.model(features)
                # print(preds.shape, labels.shape)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                batch = labels.size(0)
                running_loss += loss.item() * batch
                sample_count += batch

            train_loss = running_loss / max(sample_count, 1)
            val_loss = self._evaluate(val_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                bad_epochs = 0
                best_state = self._state_dict()
                torch.save(best_state, _ensure_dir(model_path))
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best_state is not None:
            self._load_state_dict(best_state)

        self._plot_losses(train_losses, val_losses, loss_plot_path)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val,
            "epochs_trained": len(train_losses),
            "model_path": str(model_path),
            "loss_plot_path": str(loss_plot_path),
        }

    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.view(-1).long()
                labels = labels.to(self.device)
                loss = criterion(self.model(features), labels)
                batch = labels.size(0)
                total += loss.item() * batch
                count += batch
        return total / max(count, 1)

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
        df.insert(0, "prediction", preds)
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


if __name__ == "__main__":
    model = LSTMClassifier(input_size=10)
