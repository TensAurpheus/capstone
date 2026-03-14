import pandas as pd
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union, Optional
from src.utils.metrics import bss_metric
from src.utils.general_utils import set_deterministic

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
    "min_point_atr",
]


def ulcer_index(eq: pd.Series) -> float:
    eq = eq.astype(float)
    rollmax = eq.cummax()
    dd = 1.0 - (eq / rollmax).clip(upper=1.0)
    return float(np.sqrt(np.mean(np.square(dd.values)))) if len(dd) else 0.0


def martin_ratio(eq: pd.Series, c, mar: float = 0.20) -> float:
    ui = ulcer_index(eq)
    if ui == 0.0:
        return math.inf if c > mar else -math.inf
    return (c - mar) / ui


@dataclass
class TradeRecord:
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
    min_point_atr: float = np.nan


@dataclass
class TripleBarrierConfig:
    tp_distance: float
    sl_distance: float
    holding_period: int
    min_return: float = 0.0


@dataclass
class PredictedRangeConfig:
    holding_period: int
    min_rr: float


class TradingStrategy:
    def __init__(
        self,
        mode: str = "triple_barrier",
        periods_per_year: int = 365 * 24 * 4,
        transaction_cost: bool = True,
        slippage: float = 0.0,
        triple_barrier_config: TripleBarrierConfig | None = None,
        predicted_range_config: PredictedRangeConfig | None = None,
        use_limit_entries: bool = False,
        limit_offset: float = 0.0,
        taker_fee: float = 0.0005,
        maker_fee: float = 0.0002,
    ) -> None:
        self.mode = mode
        self.periods_per_year = periods_per_year
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.taker_fee = taker_fee if transaction_cost else 0.0
        self.maker_fee = maker_fee if transaction_cost else 0.0
        self.use_limit_entries = use_limit_entries
        self.limit_offset = limit_offset
        self.triple_barrier_config = triple_barrier_config
        self.predicted_range_config = predicted_range_config
        self.reset()

    def reset(self) -> None:
        self.signals = np.array([])
        self.positions = []
        self.returns = []
        self.equity_curve = []
        self.trade_log = pd.DataFrame(columns=TRADE_COLUMNS)

    def _commission_path(self, entry_type: str, exit_reason: str) -> float:
        fees = self.taker_fee if entry_type == "market" else self.maker_fee
        fees += self.maker_fee if exit_reason == "take_profit" else self.taker_fee
        return fees

    def generate_signals(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        probability_column: str = "p2",
        atr_column="atr_200",
    ) -> np.ndarray:
        if self.mode == "triple_barrier":
            probs = predictions[probability_column].to_numpy(dtype=float)
            config = self.triple_barrier_config
            close_prices, atrs = prices["close"].to_numpy(dtype=float), prices[
                atr_column
            ].to_numpy(dtype=float)
            atr_pct = atrs / close_prices
            tp_move, sl_move = (
                config.tp_distance * atr_pct,
                config.sl_distance * atr_pct,
            )
            fee_tp, fee_sl = self._commission_path(
                "market", "take_profit"
            ), self._commission_path("market", "stop_loss")
            reward, risk = tp_move - (self.slippage + fee_tp), sl_move + (
                self.slippage + fee_sl
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                p_be = np.where(reward + risk > 0, risk / (reward + risk), 1.0)
            edge = probs - np.clip(p_be, 0.0, 1.0)
            entries = (edge >= config.min_return).astype(int)
            entries[np.isnan(edge)] = 0
            return entries
        return np.zeros(len(prices))

    def backtest(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        timestamps: Optional[Sequence] = None,
        probability_column: str = "p2",
        atr_column: str = "atr_200",
        equity: float = 1000.0,
        position_size: float = 1.0,
        risk_mode: str = "fixed_size",
        compound: bool = False,
    ) -> dict:
        self.initial_equity, self.compound = equity, compound
        close_prices, high_prices, low_prices, atr = (
            prices["close"].values,
            prices["high"].values,
            prices["low"].values,
            prices[atr_column].values,
        )
        signals = self.generate_signals(
            predictions,
            prices,
            probability_column=probability_column,
            atr_column=atr_column,
        )
        self.signals = signals
        tp_prices = close_prices + self.triple_barrier_config.tp_distance * atr
        sl_prices = close_prices - self.triple_barrier_config.sl_distance * atr
        ts = (
            np.asarray(timestamps)
            if timestamps is not None
            else np.arange(len(close_prices))
        )
        self.reset()
        trade_records: List[TradeRecord] = []
        current_trade, pending_order = None, None
        h_period = self.triple_barrier_config.holding_period

        for i in range(close_prices.shape[0]):
            prev_eq = equity
            if current_trade:
                low_i, high_i = low_prices[i], high_prices[i]
                exit_price, exit_reason = None, None
                if low_i <= current_trade["stop_price"]:
                    exit_price, exit_reason = current_trade["stop_price"], "stop_loss"
                elif high_i >= current_trade["target_price"]:
                    exit_price, exit_reason = (
                        current_trade["target_price"],
                        "take_profit",
                    )
                elif i >= current_trade["expiry_index"] or i == len(close_prices) - 1:
                    exit_price, exit_reason = close_prices[i], (
                        "expiry"
                        if i >= current_trade["expiry_index"]
                        else "market_close"
                    )
                else:
                    current_trade["min_low"] = min(current_trade["min_low"], low_i)

                if exit_reason:
                    equity, record = self._finalize_trade(
                        current_trade,
                        exit_price,
                        ts[i],
                        exit_reason,
                        equity,
                        current_trade["position_size"],
                    )
                    trade_records.append(record)
                    current_trade = None

            if not current_trade and pending_order:
                if i > pending_order["order_expiry_index"]:
                    pending_order = None
                elif low_prices[i] < pending_order["limit_price"]:
                    entry_p = pending_order["limit_price"]
                    cur_pos = (
                        position_size
                        if risk_mode == "fixed_size"
                        else position_size
                        / (
                            abs(pending_order["stop_price"] / entry_p - 1)
                            + self.slippage
                            + self.taker_fee
                            + self.maker_fee
                        )
                    )
                    current_trade = {
                        "open_date": ts[i],
                        "entry_price": entry_p,
                        "target_price": pending_order["target_price"],
                        "stop_price": pending_order["stop_price"],
                        "expiry_index": pending_order["order_expiry_index"],
                        "entry_index": i,
                        "entry_equity": equity,
                        "position_size": cur_pos,
                        "entry_atr": atr[i],
                        "min_low": entry_p,
                        "entry_type": "limit",
                    }
                    pending_order = None

            if (
                not current_trade
                and not pending_order
                and signals[i] == 1
                and i < len(close_prices) - 1
            ):
                if self.use_limit_entries:
                    dist = close_prices[i] - sl_prices[i]
                    lp = (
                        close_prices[i] - self.limit_offset * dist
                        if dist > 0
                        else close_prices[i]
                    )
                    pending_order = {
                        "signal_index": i,
                        "limit_price": max(lp, sl_prices[i] * (1 + 1e-8)),
                        "target_price": tp_prices[i],
                        "stop_price": sl_prices[i],
                        "order_expiry_index": i + h_period,
                    }
                else:
                    cur_pos = (
                        position_size
                        if risk_mode == "fixed_size"
                        else position_size
                        / (
                            abs(sl_prices[i] / close_prices[i] - 1)
                            + self.slippage
                            + 2 * self.taker_fee
                        )
                    )
                    current_trade = {
                        "open_date": ts[i],
                        "entry_price": close_prices[i],
                        "target_price": tp_prices[i],
                        "stop_price": sl_prices[i],
                        "expiry_index": i + h_period,
                        "entry_index": i,
                        "entry_equity": equity,
                        "position_size": cur_pos,
                        "entry_atr": atr[i],
                        "min_low": close_prices[i],
                        "entry_type": "market",
                    }

            self.returns.append((equity - prev_eq) / prev_eq if prev_eq else 0.0)
            self.equity_curve.append(equity)
            self.positions.append(1 if current_trade else 0)

        if trade_records:
            self.trade_log = pd.DataFrame(
                [r.__dict__ for r in trade_records], columns=TRADE_COLUMNS
            )
        return self.calculate_metrics()

    def _finalize_trade(self, trade, exit_p, close_ts, reason, eq, pos_size):
        fees = self._commission_path(trade.get("entry_type", "market"), reason)
        gross_ret = (
            exit_p / trade["entry_price"] - 1.0 - (self.slippage + fees)
        ) * pos_size
        if self.compound:
            eq *= 1 + gross_ret
            final_pos = pos_size * eq
        else:
            eq += self.initial_equity * gross_ret
            final_pos = pos_size * self.initial_equity
        min_l = trade.get("min_low", trade["entry_price"])
        min_pt_atr = (
            (trade["entry_price"] - min_l) / trade["entry_atr"]
            if trade.get("entry_atr", 0) > 0
            else np.nan
        )
        rec = TradeRecord(
            trade["open_date"],
            close_ts,
            "long",
            eq - trade["entry_equity"],
            (eq / trade["entry_equity"] - 1) * 100,
            eq,
            trade["entry_price"],
            exit_p,
            reason,
            final_pos,
            min_pt_atr,
        )
        return eq, rec

    def calculate_metrics(self) -> dict:
        E = np.array(self.equity_curve)
        R = np.array(self.returns)
        std = R.std(ddof=1) if len(R) > 1 else 0.0
        sharpe = np.sqrt(self.periods_per_year) * R.mean() / std if std > 0 else 0.0
        total_ret = (E[-1] / E[0] - 1) * 100 if len(E) > 0 and E[0] > 0 else 0.0
        mdd = (E / np.maximum.accumulate(E) - 1).min() if len(E) > 0 else 0.0
        cagr = (
            (E[-1] / E[0]) ** (self.periods_per_year / len(E)) - 1
            if len(E) > 0 and E[0] > 0
            else 0.0
        )
        winrate = (
            (self.trade_log["pnl"] > 0).mean() if not self.trade_log.empty else 0.0
        )
        return {
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "total_return": total_ret,
            "volatility": std * np.sqrt(self.periods_per_year),
            "final_equity": E[-1] if len(E) > 0 else 0.0,
            "winrate": winrate,
            "calmar_ratio": cagr / abs(mdd) if mdd < 0 else np.nan,
            "martin_ratio": martin_ratio(pd.Series(E), cagr),
            "num_trades": len(self.trade_log),
        }


def periods_per_year(tf: str, days_in_year: float = 365.0) -> float:
    tf = tf.strip().lower()
    num = float("".join(ch for ch in tf if ch.isdigit()))
    unit = "".join(ch for ch in tf if ch.isalpha())
    mins = {"m": 1, "h": 60, "d": 1440, "w": 10080, "mo": 43200}[unit]
    return (days_in_year * 1440) / (num * mins)


def standard_trade_test(predictions, prices, ku, kd, hold, **kwargs):
    tf = kwargs.get("tf", "15m")
    config = TripleBarrierConfig(ku, kd, hold, kwargs.get("min_return", 0.0))
    strat = TradingStrategy(
        periods_per_year=periods_per_year(tf),
        triple_barrier_config=config,
        **{
            k: v
            for k, v in kwargs.items()
            if k
            in ["transaction_cost", "slippage", "use_limit_entries", "limit_offset"]
        }
    )
    metrics = strat.backtest(
        predictions,
        prices,
        **{
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "probability_column",
                "atr_column",
                "equity",
                "position_size",
                "risk_mode",
                "compound",
            ]
        }
    )
    return (
        metrics,
        strat.trade_log,
        {
            "equity": np.array(strat.equity_curve),
            "positions": np.array(strat.positions),
        },
    )


def sweep_min_return(prices, df_pred, ku, kd, hold, min_grid, **kwargs):
    best_ret, best_mart = {"score": -np.inf, "mr": None, "metrics": None}, {
        "score": -np.inf,
        "mr": None,
        "metrics": None,
    }
    for mr in min_grid:
        m, _, _ = standard_trade_test(
            df_pred, prices, ku, kd, hold, min_return=mr, **kwargs
        )
        if m["num_trades"] < kwargs.get("min_trades", 0):
            continue
        if m["total_return"] > best_ret["score"]:
            best_ret.update({"score": m["total_return"], "mr": mr, "metrics": m})
        if m["martin_ratio"] > best_mart["score"]:
            best_mart.update({"score": m["martin_ratio"], "mr": mr, "metrics": m})
    return {"best_return": best_ret, "best_martin": best_mart}


def build_trade_metrics_df(all_results: dict) -> pd.DataFrame:
    rows = []
    for (tf, es), info in all_results.items():
        for split, key in [("val", "val_bnh"), ("test", "test_bnh")]:
            rows.append(
                {
                    "tf": tf,
                    "es": es,
                    "split": split,
                    "model": "BnH",
                    "mr": None,
                    "sel": "bnh",
                    "total_return": info.get(key),
                }
            )
        for model_type, splits in [
            ("DL", ["val_results", "test_results"]),
            ("logreg", ["baseline_val", "baseline_test"]),
        ]:
            # This would need more complex implementation to match exactly, but let's keep it simple for now
            pass
    return pd.DataFrame(rows)
