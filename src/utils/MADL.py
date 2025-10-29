import torch
import torch.nn as nn


def madl_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # (-1) * sign(R_i * R̂_i) * abs(R_i), averaged
    return torch.mean((-1.0) * torch.sign(y_true * y_pred) * torch.abs(y_true))


class SoftMADL(nn.Module):
    """Differentiable surrogate for training."""

    def __init__(self, k: float = 5.0):
        super().__init__()
        self.k = k

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        score = torch.tanh(self.k * (y_true * y_pred))   # ~ sign(...)
        return torch.mean((-1.0) * score * torch.abs(y_true))
