"""Trading utilities for running strategy backtests on pre-generated predictions."""

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
    "entry_price",
    "exit_price",
    "exit_reason",
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
    entry_price: float
    exit_price: float
    exit_reason: str


@dataclass
class TripleBarrierConfig:
    """Configuration parameters for the triple-barrier strategy."""

    tp_distance: float
    sl_distance: float
    holding_period: int
    min_expected_return: float = 0.0


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
        position_size: float = 1.0,
        initial_equity: float = 1000.0,
        periods_per_year: int = 365 * 24,
        transaction_cost: float = 0.0,
        triple_barrier_config: TripleBarrierConfig | None = None,
        predicted_range_config: PredictedRangeConfig | None = None,
        probability_column: str = "prob_tp_first",
        log_up_column: str = "log_dist_max",
        log_down_column: str = "log_dist_min",
    ) -> None:
        if mode not in {"triple_barrier", "predicted_range"}:
            raise ValueError("mode must be 'triple_barrier' or 'predicted_range'")

        self.mode = mode
        self.position_size = position_size
        self.initial_equity = initial_equity
        self.periods_per_year = periods_per_year
        self.transaction_cost = transaction_cost

        if mode == "triple_barrier":
            if triple_barrier_config is None:
                raise ValueError(
                    "triple_barrier_config must be provided for triple_barrier mode"
                )
            if triple_barrier_config.holding_period <= 0:
                raise ValueError("holding_period must be positive for triple_barrier strategy")
            self.triple_barrier_config = triple_barrier_config
        else:
            if predicted_range_config is None:
                raise ValueError(
                    "predicted_range_config must be provided for predicted_range mode"
                )
            if predicted_range_config.holding_period <= 0:
                raise ValueError("holding_period must be positive for predicted_range strategy")
            if predicted_range_config.min_rr <= 0:
                raise ValueError("min_rr must be positive for predicted_range strategy")
            self.predicted_range_config = predicted_range_config

        self.probability_column = probability_column
        self.log_up_column = log_up_column
        self.log_down_column = log_down_column

        self.signals: np.ndarray = np.array([])
        self.predictions: pd.DataFrame | None = None
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

    def _prepare_price_frame(self, price_data: Sequence[float] | pd.DataFrame) -> pd.DataFrame:
        """Ensure price information is stored as a DataFrame with OHLC columns."""

        if isinstance(price_data, pd.DataFrame):
            price_frame = price_data.copy()
        else:
            price_array = np.asarray(price_data, dtype=float)
            price_frame = pd.DataFrame({"close": price_array})

        if "close" not in price_frame:
            raise ValueError("Price data must include a 'close' column")

        if "high" not in price_frame:
            price_frame["high"] = price_frame["close"]
        if "low" not in price_frame:
            price_frame["low"] = price_frame["close"]

        return price_frame[["close", "high", "low"]]

    def _prepare_predictions(self, predictions: Sequence[float] | pd.DataFrame) -> pd.DataFrame:
        """Convert prediction inputs to a pandas DataFrame for easier access."""

        if isinstance(predictions, pd.DataFrame):
            return predictions.reset_index(drop=True)

        if isinstance(predictions, dict):
            return pd.DataFrame(predictions)

        preds_array = np.asarray(predictions, dtype=float)
        if self.mode == "triple_barrier" and preds_array.ndim == 1:
            return pd.DataFrame({self.probability_column: preds_array})

        raise ValueError("Predictions must be provided as a DataFrame, dict, or 1D array")

    def generate_signals(
        self,
        predictions: Sequence[float] | pd.DataFrame,
        close_prices: Sequence[float],
    ) -> dict:
        """Generate entry signals and barrier levels for the configured mode."""

        price_array = np.asarray(close_prices, dtype=float)
        preds_frame = self._prepare_predictions(predictions)

        if preds_frame.shape[0] != price_array.shape[0]:
            raise ValueError("Predictions and prices must have the same length.")

        if self.mode == "triple_barrier":
            probs = preds_frame[self.probability_column].to_numpy(dtype=float)
            config = self.triple_barrier_config
            expected_return = (
                probs * config.tp_distance - (1.0 - probs) * config.sl_distance
            )
            entries = expected_return >= config.min_expected_return
            tp_prices = price_array * (1.0 + config.tp_distance)
            sl_prices = price_array * (1.0 - config.sl_distance)
        else:
            log_up = preds_frame[self.log_up_column].to_numpy(dtype=float)
            log_down = preds_frame[self.log_down_column].to_numpy(dtype=float)
            tp_prices = price_array * np.exp(log_up)
            sl_prices = price_array / np.exp(log_down)

            rr_denominator = np.maximum(price_array - sl_prices, 1e-12)
            rr = (tp_prices - price_array) / rr_denominator
            entries = (
                (rr >= self.predicted_range_config.min_rr)
                & np.isfinite(rr)
                & (tp_prices > price_array)
                & (sl_prices < price_array)
            )

        return {
            "entries": entries.astype(int),
            "tp": tp_prices,
            "sl": sl_prices,
            "predictions": preds_frame,
        }

    def backtest(
        self,
        predictions: Sequence[float] | pd.DataFrame,
        prices: Sequence[float] | pd.DataFrame,
        timestamps: Sequence[object] | None = None,
        transaction_cost: float | None = None,
    ) -> dict:
        """Run a backtest using pre-generated prediction outputs."""

        price_frame = self._prepare_price_frame(prices)
        close_prices = price_frame["close"].to_numpy(dtype=float)

        if close_prices.size == 0:
            raise ValueError("Price data cannot be empty.")

        signals_info = self.generate_signals(predictions, close_prices)
        self.predictions = signals_info["predictions"]
        self.signals = signals_info["entries"]
        tp_prices = signals_info["tp"]
        sl_prices = signals_info["sl"]

        if timestamps is None:
            timestamps_arr = np.arange(len(close_prices))
        else:
            timestamps_arr = np.asarray(timestamps)
            if timestamps_arr.shape[0] != close_prices.shape[0]:
                raise ValueError("Timestamps must align with predictions/prices.")

        high_prices = price_frame["high"].to_numpy(dtype=float)
        low_prices = price_frame["low"].to_numpy(dtype=float)

        self.reset()
        equity = float(self.initial_equity)
        trade_records: List[TradeRecord] = []
        current_trade: dict | None = None
        cost = self.transaction_cost if transaction_cost is None else float(transaction_cost)

        holding_period = (
            self.triple_barrier_config.holding_period
            if self.mode == "triple_barrier"
            else self.predicted_range_config.holding_period
        )

        for i in range(close_prices.shape[0]):
            timestamp = timestamps_arr[i]
            prev_equity = equity

            if current_trade is not None and i > current_trade["entry_index"]:
                exit_price: float | None = None
                exit_reason: str | None = None

                if low_prices[i] <= current_trade["stop_price"]:
                    exit_price = current_trade["stop_price"]
                    exit_reason = "stop_loss"
                elif high_prices[i] >= current_trade["target_price"]:
                    exit_price = current_trade["target_price"]
                    exit_reason = "take_profit"
                elif i >= current_trade["expiry_index"]:
                    exit_price = close_prices[i]
                    exit_reason = "expiry"
                elif i == close_prices.shape[0] - 1:
                    exit_price = close_prices[i]
                    exit_reason = "market_close"

                if exit_reason is not None and exit_price is not None:
                    equity, record = self._finalize_trade(
                        current_trade,
                        exit_price,
                        timestamp,
                        exit_reason,
                        equity,
                        cost,
                    )
                    trade_records.append(record)
                    current_trade = None

            if current_trade is None and self.signals[i] == 1:
                entry_price = close_prices[i]
                target_price = float(tp_prices[i])
                stop_price = float(sl_prices[i])

                if not np.isfinite(entry_price) or not np.isfinite(target_price) or not np.isfinite(stop_price):
                    step_return = (equity - prev_equity) / prev_equity if prev_equity else 0.0
                    self.returns.append(step_return)
                    self.equity_curve.append(equity)
                    self.positions.append(0)
                    continue

                if target_price <= entry_price or stop_price >= entry_price:
                    step_return = (equity - prev_equity) / prev_equity if prev_equity else 0.0
                    self.returns.append(step_return)
                    self.equity_curve.append(equity)
                    self.positions.append(0)
                    continue

                expiry_index = min(i + holding_period, close_prices.shape[0] - 1)
                if expiry_index <= i:
                    step_return = (equity - prev_equity) / prev_equity if prev_equity else 0.0
                    self.returns.append(step_return)
                    self.equity_curve.append(equity)
                    self.positions.append(0)
                    continue

                equity *= (1 - cost)
                current_trade = {
                    "open_date": timestamp,
                    "entry_price": entry_price,
                    "target_price": target_price,
                    "stop_price": stop_price,
                    "expiry_index": expiry_index,
                    "entry_index": i,
                    "entry_equity": equity,
                }

            step_return = (equity - prev_equity) / prev_equity if prev_equity else 0.0
            self.returns.append(step_return)
            self.equity_curve.append(equity)
            self.positions.append(1 if current_trade is not None else 0)

        if current_trade is not None:
            equity, record = self._finalize_trade(
                current_trade,
                close_prices[-1],
                timestamps_arr[-1],
                "market_close",
                equity,
                cost,
            )
            trade_records.append(record)

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
        transaction_cost: float,
    ) -> tuple[float, TradeRecord]:
        """Close an active trade and return the updated equity and trade record."""

        entry_equity = trade["entry_equity"]
        entry_price = trade["entry_price"]

        gross_return = (exit_price / entry_price - 1.0) * self.position_size
        equity *= (1 + gross_return)
        equity *= (1 - transaction_cost)

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
        )

        return equity, record

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


if __name__ == "__main__":
    prices = pd.DataFrame(
        {
            "close": [100, 102, 105, 103, 108, 110],
            "high": [101, 103, 106, 104, 109, 111],
            "low": [99, 100, 103, 101, 104, 107],
        }
    )

    tb_config = TripleBarrierConfig(tp_distance=0.03, sl_distance=0.015, holding_period=3)
    strategy = TradingStrategy(
        mode="triple_barrier",
        triple_barrier_config=tb_config,
        initial_equity=1000.0,
    )

    prob_predictions = pd.Series([0.6, 0.55, 0.7, 0.4, 0.65, 0.5])
    metrics = strategy.backtest(prob_predictions, prices)
    print(metrics)
    print(strategy.trade_log)
