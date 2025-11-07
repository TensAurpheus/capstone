from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

TRADE_COLUMNS = [
    "open_date",
    "close_date",
    "side",
    "position_size",
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
    position_size: float = 1.0


@dataclass
class TripleBarrierConfig:
    """Configuration parameters for the triple-barrier strategy."""

    tp_distance: float
    sl_distance: float
    holding_period: int
    min_return: float = 0.0


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
        periods_per_year: int = 365 * 24 * 4,
        transaction_cost: float = 0.0,
        slippage: float = 0.0,
        triple_barrier_config: TripleBarrierConfig | None = None,
        predicted_range_config: PredictedRangeConfig | None = None,

    ) -> None:
        if mode not in {"triple_barrier", "predicted_range"}:
            raise ValueError(
                "mode must be 'triple_barrier' or 'predicted_range'")

        self.mode = mode
        self.periods_per_year = periods_per_year
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        if mode == "triple_barrier":
            if triple_barrier_config is None:
                raise ValueError(
                    "triple_barrier_config must be provided for triple_barrier mode"
                )
            if triple_barrier_config.holding_period <= 0:
                raise ValueError(
                    "holding_period must be positive for triple_barrier strategy")
            self.triple_barrier_config = triple_barrier_config
        else:
            if predicted_range_config is None:
                raise ValueError(
                    "predicted_range_config must be provided for predicted_range mode"
                )
            if predicted_range_config.holding_period <= 0:
                raise ValueError(
                    "holding_period must be positive for predicted_range strategy")
            if predicted_range_config.min_rr <= 0:
                raise ValueError(
                    "min_rr must be positive for predicted_range strategy")
            self.predicted_range_config = predicted_range_config

        self.signals: np.ndarray = np.array([])
        self.positions: List[int] = []
        self.returns: List[float] = []
        self.equity_curve: List[float] = []
        self.trade_log: pd.DataFrame = pd.DataFrame(columns=TRADE_COLUMNS)

    def reset(self) -> None:
        """Reset tracked performance series."""

        self.positions = []
        self.returns = []
        self.equity_curve = []
        self.trade_log = pd.DataFrame(columns=TRADE_COLUMNS)

    def generate_signals(
        self,
        predictions: Sequence[float] | pd.DataFrame,
        prices: pd.DataFrame,
        probability_column: str = "p2",
        log_up_column: str = "y_up",
        log_down_column: str = "y_down",
    ) -> dict:
        """Generate entry signals and barrier levels for the configured mode."""

        if predictions.shape[0] != prices.shape[0]:
            raise ValueError(
                "Predictions and prices must have the same length.")

        if self.mode == "triple_barrier":
            probs = predictions[probability_column].to_numpy(dtype=float)
            config = self.triple_barrier_config
            expected_return = probs * \
                (config.tp_distance/config.sl_distance -
                 self.slippage - self.transaction_cost) - 1
            entries = (expected_return >= config.min_return).astype(int)

        elif self.mode == "predicted_range":
            close_prices = prices['close'].to_numpy(dtype=float)
            log_up = predictions[log_up_column].to_numpy(dtype=float)
            log_down = predictions[log_down_column].to_numpy(dtype=float)
            tp_prices = close_prices * np.exp(log_up)
            sl_prices = close_prices * np.exp(-log_down)
            ranges = tp_prices - sl_prices
            sl_prices = sl_prices - ranges * 0.25

            rr = (tp_prices - close_prices) / (close_prices - sl_prices + 1e-8)
            entries = (
                rr >= self.predicted_range_config.min_rr + self.slippage + self.transaction_cost
            ).astype(int)

        return entries

    def backtest(
        self,
        predictions: Sequence[float] | pd.DataFrame,
        prices: Sequence[float] | pd.DataFrame,
        timestamps: Sequence[object] | None = None,
        probability_column: str = "TP_prob",
        log_up_column: str = "y_up",
        log_down_column: str = "y_down",
        atr_column: str = "atr_14",
        equity: float = 1000.0,
        position_size: float = 1.0,
        risk_mode: str = "fixed_size",
        compound: bool = False,
    ) -> dict:
        """Run a backtest using pre-generated prediction outputs."""

        self.initial_equity = equity
        self.compound = compound

        if risk_mode not in {"fixed_size", "fixed_risk"}:
            raise ValueError("risk_mode must be 'fixed_size' or 'fixed_risk'")
        close_prices = prices["close"].to_numpy(dtype=float)

        if close_prices.size == 0:
            raise ValueError("Price data cannot be empty.")

        signals = self.generate_signals(predictions, prices, probability_column=probability_column,
                                        log_up_column=log_up_column, log_down_column=log_down_column)

        self.signals = signals
        if self.mode == "predicted_range":
            log_up = predictions[log_up_column].to_numpy(dtype=float)
            log_down = predictions[log_down_column].to_numpy(dtype=float)
            tp_prices = close_prices * np.exp(log_up)
            sl_prices = close_prices * np.exp(-log_down)
            ranges = tp_prices - sl_prices
            sl_prices = sl_prices - ranges * 0.25
        elif self.mode == "triple_barrier":
            atr = prices[atr_column].to_numpy(dtype=float)
            tp_prices = close_prices + self.triple_barrier_config.tp_distance * atr
            sl_prices = close_prices - self.triple_barrier_config.sl_distance * atr

        if timestamps is None:
            timestamps_arr = np.arange(len(close_prices))
        else:
            timestamps_arr = np.asarray(timestamps)
            if timestamps_arr.shape[0] != close_prices.shape[0]:
                raise ValueError(
                    "Timestamps must align with predictions/prices.")

        high_prices = prices["high"].to_numpy(dtype=float)
        low_prices = prices["low"].to_numpy(dtype=float)

        self.reset()
        trade_records: List[TradeRecord] = []
        current_trade: dict | None = None

        holding_period = (
            self.triple_barrier_config.holding_period
            if self.mode == "triple_barrier"
            else self.predicted_range_config.holding_period
        )

        exit_price: float | None = None
        exit_reason: str | None = None

        for i in range(close_prices.shape[0]):
            timestamp = timestamps_arr[i]
            prev_equity = equity
            # print('step ', i+1)

            if current_trade is not None and i > current_trade["entry_index"]:

                if low_prices[i] <= current_trade["stop_price"]:
                    exit_price = current_trade["stop_price"]
                    exit_reason = "stop_loss"
                    # print(
                    # f'sl low_price: {low_prices[i]}, SL: {current_trade["stop_price"]}')
                elif high_prices[i] >= current_trade["target_price"]:
                    exit_price = current_trade["target_price"]
                    exit_reason = "take_profit"
                    # print(
                    #     f'tp high_price: {high_prices[i]}, TP: {current_trade["target_price"]}')
                elif i >= current_trade["expiry_index"]:
                    exit_price = close_prices[i]
                    exit_reason = "expiry"
                    # print(f'expiry close_price: {close_prices[i]}')
                elif i == close_prices.shape[0] - 1:
                    exit_price = close_prices[i]
                    exit_reason = "market_close"
                    # print(f'market_close close_price: {close_prices[i]}')

                if exit_reason is not None and exit_price is not None:
                    equity, record = self._finalize_trade(
                        current_trade,
                        exit_price,
                        timestamp,
                        exit_reason,
                        equity,
                        current_trade["position_size"],
                    )
                    # print(
                    #     f'close trade TP: {current_trade["target_price"]}, SL: {current_trade["stop_price"]}, low_price: {low_prices[i]}, high_price: {high_prices[i]}, exit_reason: {exit_reason}')
                    trade_records.append(record)
                    current_trade = None
                    exit_price = None
                    exit_reason = None

            if current_trade is None and self.signals[i] == 1 and i < close_prices.shape[0] - 1:
                entry_price = close_prices[i]
                target_price = float(tp_prices[i])
                stop_price = float(sl_prices[i])
                expiry_index = i + holding_period
                if risk_mode == "fixed_size":
                    cur_position_size = position_size
                elif risk_mode == "fixed_risk":
                    cur_position_size = position_size / \
                        (np.abs(stop_price/entry_price - 1) +
                         self.slippage + self.transaction_cost)

                current_trade = {
                    "open_date": timestamp,
                    "entry_price": entry_price,
                    "target_price": target_price,
                    "stop_price": stop_price,
                    "expiry_index": expiry_index,
                    "entry_index": i,
                    "entry_equity": equity,
                    "position_size": cur_position_size
                }
                # print(
                #     f'open trade TP: {current_trade["target_price"]}, SL: {current_trade["stop_price"]}')

            step_return = (equity - prev_equity) / \
                prev_equity if prev_equity else 0.0
            self.returns.append(step_return)
            self.equity_curve.append(equity)
            self.positions.append(1 if current_trade is not None else 0)

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
        position_size: float = 1.0,
    ) -> tuple[float, TradeRecord]:
        """Close an active trade and return the updated equity and trade record."""

        entry_equity = trade["entry_equity"]
        entry_price = trade["entry_price"]

        gross_return = (exit_price / entry_price - 1.0 -
                        self.slippage - self.transaction_cost) * position_size

        if self.compound:
            equity *= (1 + gross_return)
            position_size = position_size*equity
        else:
            equity += self.initial_equity * gross_return
            position_size = position_size*self.initial_equity

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
            position_size=position_size
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
            sharpe = np.sqrt(self.periods_per_year) * \
                returns_array.mean() / returns_std
        else:
            sharpe = 0.0

        volatility = returns_std * \
            np.sqrt(self.periods_per_year) if returns_std > 0 else 0.0

        cummax = np.maximum.accumulate(
            equity_array) if equity_array.size else np.array([])
        drawdown = (equity_array - cummax) / \
            cummax if cummax.size else np.array([0.0])
        max_drawdown = drawdown.min() if drawdown.size else 0.0
        total_return = (
            (equity_array[-1] / equity_array[0] - 1) *
            100 if equity_array.size else 0.0
        )

        metrics = {
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
            "volatility": volatility,
            "final_equity": equity_array[-1] if equity_array.size else 0,
        }

        return metrics
