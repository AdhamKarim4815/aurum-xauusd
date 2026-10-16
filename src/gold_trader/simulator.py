from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from gold_trader.config import RiskSettings, SimulationSettings
from gold_trader.domain import AccountSnapshot, OrderIntent, Side
from gold_trader.risk import RiskEngine
from gold_trader.strategies import BaseStrategy


@dataclass(frozen=True)
class TradeResult:
    trade_id: str
    strategy_id: str
    side: str
    entry_at: datetime
    exit_at: datetime
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    lots: float
    pnl: float
    commission: float
    swap: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in self.__dict__.items()
        }


class DeterministicSimulator:
    def __init__(
        self,
        risk: RiskSettings,
        simulation: SimulationSettings,
        starting_equity: float = 10_000,
    ) -> None:
        self.risk = risk
        self.simulation = simulation
        self.starting_equity = starting_equity
        self.risk_engine = RiskEngine(risk)

    def run(
        self, bars: pd.DataFrame, strategy: BaseStrategy
    ) -> tuple[list[TradeResult], dict[str, float]]:
        data = bars.sort_values("timestamp").reset_index(drop=True)
        balance = self.starting_equity
        peak = balance
        day_start = balance
        day_marker: date | None = None
        rolling_high = balance
        trades: list[TradeResult] = []
        index = strategy.warmup
        candidates = strategy.precompute(data)

        while index < len(data) - 2:
            timestamp = pd.Timestamp(data.iloc[index]["timestamp"]).to_pydatetime().astimezone(UTC)
            if day_marker != timestamp.date():
                day_marker = timestamp.date()
                day_start = balance
            signal = strategy.candidate_signal(data, candidates, index)
            if signal is None:
                index += 1
                continue
            intent = OrderIntent(
                signal_id=signal.event_id,
                side=signal.side,
                reference_price=signal.reference_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                requested_risk_fraction=self.risk.max_risk_fraction,
                strategy_hash=signal.strategy_hash,
            )
            account = AccountSnapshot(
                balance=balance,
                equity=balance,
                day_start_equity=day_start,
                rolling_five_day_high=rolling_high,
                peak_equity=peak,
                open_position_count=0,
            )
            decision = self.risk_engine.evaluate(intent, account)
            if not decision.approved:
                index += 1
                continue
            result, exit_index = self._execute(data, index + 1, signal, decision.lots)
            trades.append(result)
            balance += result.pnl
            peak = max(peak, balance)
            rolling_high = max(rolling_high, balance)
            index = max(exit_index + 1, index + 1)

        return trades, calculate_metrics(trades, self.starting_equity)

    def _execute(
        self, data: pd.DataFrame, entry_index: int, signal: Any, lots: float
    ) -> tuple[TradeResult, int]:
        entry_bar = data.iloc[entry_index]
        half_spread = self.simulation.spread_points / 2
        adverse = half_spread + self.simulation.slippage_points
        sign = 1 if signal.side is Side.BUY else -1
        entry_price = float(entry_bar["open"]) + sign * adverse
        stop_distance = abs(signal.reference_price - signal.stop_loss)
        target_distance = abs(signal.take_profit - signal.reference_price)
        stop = entry_price - sign * stop_distance
        target = entry_price + sign * target_distance
        exit_price = entry_price
        reason = "end-of-data"
        exit_index = len(data) - 1

        for cursor in range(entry_index, len(data)):
            bar = data.iloc[cursor]
            open_price = float(bar["open"])
            high = float(bar["high"])
            low = float(bar["low"])
            if signal.side is Side.BUY:
                if open_price <= stop:
                    exit_price, reason = open_price - adverse, "gap-through-stop"
                elif open_price >= target:
                    exit_price, reason = open_price - adverse, "gap-through-target"
                elif low <= stop:
                    exit_price, reason = stop - adverse, "stop-loss"
                elif high >= target:
                    exit_price, reason = target - adverse, "take-profit"
                else:
                    continue
            else:
                if open_price >= stop:
                    exit_price, reason = open_price + adverse, "gap-through-stop"
                elif open_price <= target:
                    exit_price, reason = open_price + adverse, "gap-through-target"
                elif high >= stop:
                    exit_price, reason = stop + adverse, "stop-loss"
                elif low <= target:
                    exit_price, reason = target + adverse, "take-profit"
                else:
                    continue
            exit_index = cursor
            break
        else:
            final = data.iloc[-1]
            exit_price = float(final["close"]) - sign * adverse

        entry_at = pd.Timestamp(entry_bar["timestamp"]).to_pydatetime().astimezone(UTC)
        exit_at = pd.Timestamp(data.iloc[exit_index]["timestamp"]).to_pydatetime().astimezone(UTC)
        held_days = max((exit_at - entry_at).total_seconds() / 86400, 0)
        swap = self.simulation.swap_per_lot_day * lots * held_days
        commission = self.simulation.commission_per_lot_round_trip * lots
        gross = (exit_price - entry_price) * sign * self.risk.contract_size_oz * lots
        pnl = gross - commission + swap
        return (
            TradeResult(
                trade_id=str(uuid4()),
                strategy_id=signal.strategy_id,
                side=signal.side.value,
                entry_at=entry_at,
                exit_at=exit_at,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_loss=stop,
                take_profit=target,
                lots=lots,
                pnl=pnl,
                commission=commission,
                swap=swap,
                reason=reason,
            ),
            exit_index,
        )


def calculate_metrics(trades: list[TradeResult], starting_equity: float) -> dict[str, float]:
    if not trades:
        return {
            "net_return": 0.0,
            "net_pnl": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0.0,
            "win_rate": 0.0,
        }
    pnl = np.array([trade.pnl for trade in trades], dtype=float)
    equity = starting_equity + np.cumsum(pnl)
    peaks = np.maximum.accumulate(np.concatenate(([starting_equity], equity)))[:-1]
    drawdowns = np.maximum((peaks - equity) / peaks, 0)
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    returns = pnl / np.concatenate(([starting_equity], equity[:-1]))
    sharpe = (
        float(np.sqrt(252) * returns.mean() / returns.std(ddof=1))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else 0.0
    )
    return {
        "net_return": float((equity[-1] - starting_equity) / starting_equity),
        "net_pnl": float(equity[-1] - starting_equity),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "sharpe": sharpe,
        "max_drawdown": float(drawdowns.max(initial=0)),
        "trade_count": float(len(trades)),
        "win_rate": float((pnl > 0).mean()),
    }
