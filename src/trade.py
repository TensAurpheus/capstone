
"""Trading utilities for running strategy backtests on pre-generated predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


TRADE_COLUMNS = [
    "open_date",
    "close_date",
    "side",
    "pnl",
    "return_pct",
    "total_equity",
]


@dataclass
class TradeRecord:
    """Container for a single completed trade."""

    open_date: object
    close_date: object
    side: str
    pnl: float
    return_pct: float
    total_equity: float


class TradingStrategy:
    """Simple long/short strategy driven by external prediction sequences."""

    def __init__(
        self,
        threshold: float = 0.5,
        neutral_band: float = 0.1,
        position_size: float = 1.0,
        initial_equity: float = 1.0,
        periods_per_year: int = 365 * 24,
    ) -> None:
        self.threshold = threshold
        self.neutral_band = neutral_band
        self.position_size = position_size
        self.initial_equity = initial_equity
        self.periods_per_year = periods_per_year

        self.signals: np.ndarray = np.array([])
        self.predictions: np.ndarray = np.array([])
        self.positions: List[int] = []
        self.returns: List[float] = []
        self.equity_curve: List[float] = [initial_equity]
        self.trade_log: pd.DataFrame = pd.DataFrame(columns=TRADE_COLUMNS)

    def reset(self) -> None:
        """Reset tracked performance series."""

        self.positions = []
        self.returns = []
        self.equity_curve = [self.initial_equity]
        self.trade_log = pd.DataFrame(columns=TRADE_COLUMNS)

    def generate_signals(self, predictions: Sequence[float]) -> np.ndarray:
        """Convert a prediction sequence into discrete trading signals.

        Args:
            predictions: Iterable of floats representing the model output or
                directional scores. Predictions greater than ``threshold`` plus
                ``neutral_band`` trigger a long position, values below
                ``threshold`` minus the band trigger a short position, and
                anything in-between results in no position.

        Returns:
            Numpy array of -1, 0, or 1 signals.
        """

        preds = np.asarray(predictions, dtype=float)

        if np.all(np.isin(preds, (-1, 0, 1))):
            return preds.astype(int)

        signals = np.zeros_like(preds, dtype=int)
        signals[preds > self.threshold + self.neutral_band] = 1
        signals[preds < self.threshold - self.neutral_band] = -1
        return signals

    def backtest(
        self,
        predictions: Sequence[float],
        returns: Sequence[float],
        timestamps: Sequence[object] | None = None,
        transaction_cost: float = 0.001,
    ) -> dict:
        """Run a backtest using pre-generated prediction outputs.

        Args:
            predictions: Sequence of model predictions or scores.
            returns: Sequence of realized returns for each period (as decimals).
            timestamps: Optional sequence of labels associated with each period.
            transaction_cost: Proportional transaction cost applied when
                entering or exiting a position.

        Returns:
            Dictionary of performance metrics along with the trade log table.
        """

        predictions_arr = np.asarray(predictions, dtype=float)
        returns_arr = np.asarray(returns, dtype=float)
        if predictions_arr.shape[0] != returns_arr.shape[0]:
            raise ValueError("Predictions and returns must have the same length.")

        if timestamps is None:
            timestamps_arr = np.arange(len(predictions_arr))
        else:
            timestamps_arr = np.asarray(timestamps)
            if timestamps_arr.shape[0] != predictions_arr.shape[0]:
                raise ValueError("Timestamps must align with predictions/returns.")

        self.reset()
        self.predictions = predictions_arr
        self.signals = self.generate_signals(predictions_arr)

        equity = float(self.initial_equity)
        current_position = 0
        current_trade: dict | None = None
        trade_records: List[TradeRecord] = []

        for i, (raw_signal, period_return, timestamp) in enumerate(
            zip(self.signals, returns_arr, timestamps_arr)
        ):
            signal = int(raw_signal)
            prev_equity = equity

            if signal != current_position:
                if current_position != 0 and current_trade is not None:
                    equity *= (1 - transaction_cost)
                    trade_records.append(
                        self._close_trade(current_trade, timestamp, equity)
                    )
                    current_trade = None
                current_position = 0

                if signal != 0:
                    equity *= (1 - transaction_cost)
                    current_trade = {
                        "open_date": timestamp,
                        "position": signal,
                        "entry_equity": equity,
                    }
                    current_position = signal

            if current_position != 0:
                equity *= (1 + current_position * period_return * self.position_size)

            next_signal = self.signals[i + 1] if i < len(self.signals) - 1 else 0
            if current_position != 0 and next_signal != current_position:
                equity *= (1 - transaction_cost)
                if current_trade is not None:
                    trade_records.append(
                        self._close_trade(current_trade, timestamp, equity)
                    )
                    current_trade = None
                current_position = 0

            step_return = (equity - prev_equity) / prev_equity if prev_equity else 0.0
            self.returns.append(step_return)
            self.equity_curve.append(equity)
            self.positions.append(signal)

        if trade_records:
            self.trade_log = pd.DataFrame(
                [record.__dict__ for record in trade_records], columns=TRADE_COLUMNS
            )
        else:
            self.trade_log = pd.DataFrame(columns=TRADE_COLUMNS)

        return self.calculate_metrics()

    def _close_trade(
        self, trade: dict, close_timestamp: object, equity_after_close: float
    ) -> TradeRecord:
        """Create a :class:`TradeRecord` when a position is closed."""

        entry_equity = trade["entry_equity"]
        pnl = equity_after_close - entry_equity
        return_pct = (equity_after_close / entry_equity - 1) * 100 if entry_equity else 0.0
        side = "long" if trade["position"] > 0 else "short"

        return TradeRecord(
            open_date=trade["open_date"],
            close_date=close_timestamp,
            side=side,
            pnl=pnl,
            return_pct=return_pct,
            total_equity=equity_after_close,
        )

    def calculate_metrics(self) -> dict:
        """Calculate and return backtest performance metrics."""

        equity_array = np.asarray(self.equity_curve, dtype=float)
        returns_array = np.asarray(self.returns, dtype=float)

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

        cummax = np.maximum.accumulate(equity_array) if equity_array.size else np.array([])
        drawdown = (equity_array - cummax) / cummax if cummax.size else np.array([0.0])
        max_drawdown = drawdown.min() if drawdown.size else 0.0
        total_return = (
            (equity_array[-1] / equity_array[0] - 1) * 100 if equity_array.size else 0.0
        )

        metrics = {
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
            "volatility": volatility,
            "final_equity": equity_array[-1] if equity_array.size else self.initial_equity,
            "positions": self.trade_log.copy(),
        }

        return metrics
