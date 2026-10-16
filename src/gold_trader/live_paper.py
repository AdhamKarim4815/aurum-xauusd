from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from gold_trader.config import Settings
from gold_trader.data import LiveQuoteProvider, file_sha256, normalize_bars, resample_bars
from gold_trader.database import Store
from gold_trader.domain import (
    AccountSnapshot,
    Bar,
    Fill,
    OrderIntent,
    OrderStatus,
    PaperTradeClosed,
    Position,
    Quote,
    Side,
    Signal,
    SimulatedOrder,
)
from gold_trader.notifications import MacNotifier
from gold_trader.risk import RiskEngine
from gold_trader.strategies import BaseStrategy


def utc_now() -> datetime:
    return datetime.now(UTC)


class LivePaperEngine:
    """Continuous quote-to-simulated-fill loop. It has no real venue dependency."""

    def __init__(
        self,
        settings: Settings,
        store: Store,
        provider: LiveQuoteProvider,
        report: dict[str, Any],
        strategy: BaseStrategy,
        *,
        notifier: MacNotifier | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.settings = settings
        self.store = store
        self.provider = provider
        self.report = report
        self.strategy = strategy
        self.notifier = notifier or MacNotifier()
        self.clock = clock
        self.risk_engine = RiskEngine(settings.risk)
        self.base_bars = pd.DataFrame()
        self.current_candle: dict[str, Any] | None = None
        self.last_quote: Quote | None = None
        self.sequence = 0

    def _provider_name(self) -> str:
        return str(
            getattr(
                self.provider,
                "active_provider_name",
                getattr(self.provider, "name", "unknown"),
            )
            or "unknown"
        )

    @property
    def lineage(self) -> dict[str, str]:
        return {
            "data_hash": str(self.report["dataset_hash"]),
            "config_hash": str(self.report["configuration_hash"]),
            "strategy_hash": str(self.report["strategy_hash"]),
            "model_hash": str(self.report["model_hash"]),
        }

    def prepare(self, paper: dict[str, Any]) -> dict[str, Any]:
        dataset_value = self.report.get("data", {}).get("dataset_path")
        if not dataset_value:
            raise ValueError("approved candidate has no replay dataset for live warmup")
        dataset = Path(str(dataset_value)).resolve()
        if not dataset.is_file() or file_sha256(dataset) != self.report["dataset_hash"]:
            raise ValueError("approved live warmup dataset is missing or changed")
        raw = pd.read_parquet(dataset) if dataset.suffix == ".parquet" else pd.read_csv(dataset)
        normalized, quality = normalize_bars(raw)
        if not quality.valid:
            raise ValueError("approved live warmup dataset failed data quality")
        self.base_bars = normalized.tail(self.settings.live_paper.warmup_bars_5m).reset_index(
            drop=True
        )
        pending = paper.get("pending_signal") or {}
        self.sequence = int(pending.get("created_sequence", 0))
        balance = float(paper.get("balance", self.settings.starting_equity))
        now = self.clock().astimezone(UTC)
        paper.update(
            {
                "status": "running",
                "mode": "live",
                "feed_status": "connecting",
                "entries_allowed": False,
                "balance": balance,
                "equity": float(paper.get("equity", balance)),
                "day_start_equity": float(paper.get("day_start_equity", balance)),
                "day_marker": paper.get("day_marker", now.date().isoformat()),
                "rolling_five_day_high": float(paper.get("rolling_five_day_high", balance)),
                "peak_equity": float(paper.get("peak_equity", balance)),
                "open_position_count": 1 if paper.get("position") else 0,
                "observation_trade_count": int(paper.get("observation_trade_count", 0)),
                "pending_signal": paper.get("pending_signal"),
                "daily_history": paper.get("daily_history", []),
                "feed_failures": 0,
                "next_retry_seconds": self.settings.live_paper.poll_seconds,
                "last_decision_bar": paper.get("last_decision_bar"),
                "live_provider": self._provider_name(),
                "simulated_fills_only": True,
                "reconciliation": "warmup-loaded-no-retroactive-trades",
            }
        )
        self.store.set_state("paper", paper)
        return paper

    def poll_once(self) -> dict[str, Any]:
        paper = self.store.get_state("paper") or {}
        approval = self.store.valid_approval()
        lineage_matches = approval is not None and all(
            approval.get(key) == self.report.get(key)
            for key in (
                "candidate_hash",
                "dataset_hash",
                "configuration_hash",
                "strategy_hash",
                "model_hash",
            )
        )
        if not lineage_matches:
            paper.update(
                {
                    "status": "approval-invalidated",
                    "entries_allowed": False,
                    "feed_status": "blocked",
                    "pending_signal": None,
                }
            )
            self.store.set_state("paper", paper)
            self.notifier.notify("Aurum approval invalidated", "Live paper entries are blocked")
            return paper
        try:
            quote = asyncio.run(self.provider.quote("XAUUSD"))
        except Exception as exc:
            failures = int(paper.get("feed_failures", 0)) + 1
            retry = min(
                self.settings.live_paper.poll_seconds * 2 ** min(failures, 6),
                self.settings.live_paper.max_retry_seconds,
            )
            paper.update(
                {
                    "feed_status": "network-error",
                    "entries_allowed": False,
                    "feed_failures": failures,
                    "next_retry_seconds": retry,
                    "last_feed_error": type(exc).__name__,
                    "last_feed_error_at": self.clock().astimezone(UTC).isoformat(),
                }
            )
            self.store.set_state("paper", paper)
            if failures == 1:
                self.notifier.notify("Aurum feed failure", "New paper entries are blocked")
            return paper

        now = self.clock().astimezone(UTC)
        age = (now - quote.available_at).total_seconds()
        self.store.append_event(quote.model_copy(update=self.lineage))
        previous_received_at = paper.get("last_quote_received_at")
        paper["last_quote_at"] = quote.available_at.isoformat()
        paper["last_quote_received_at"] = now.isoformat()
        paper["last_bid"] = quote.bid
        paper["last_ask"] = quote.ask
        paper["quote_age_seconds"] = age
        paper["live_provider"] = self._provider_name()
        if age > self.settings.live_paper.stale_after_seconds or age < -30:
            paper.update(
                {
                    "feed_status": "stale",
                    "entries_allowed": False,
                    "last_feed_error": "quote timestamp outside the freshness window",
                }
            )
            self.store.set_state("paper", paper)
            self.notifier.notify("Aurum stale feed", "New paper entries are blocked")
            return paper

        if self.report.get("generation_five") and not self._generation5_session_open(
            quote.available_at
        ):
            paper.update(
                {
                    "status": "running",
                    "feed_status": "market-closed",
                    "entries_allowed": False,
                    "last_feed_error": None,
                }
            )
            self.store.set_state("paper", paper)
            return paper

        if int(paper.get("feed_failures", 0)):
            self._recover_gap(paper, quote, previous_received_at)
        paper.update(
            {
                "status": "running",
                "feed_status": "healthy",
                "feed_failures": 0,
                "next_retry_seconds": self.settings.live_paper.poll_seconds,
                "entries_allowed": not bool(paper.get("kill_switch")),
                "last_feed_error": None,
            }
        )
        self.sequence += 1
        self.last_quote = quote
        self._roll_day(paper, now.date())
        if paper.get("position"):
            self._mark_or_close_position(paper, quote, now)
        self._enforce_account_halts(paper)
        if paper.get("pending_signal") and paper.get("entries_allowed") and not paper.get(
            "position"
        ):
            pending_sequence = int(paper["pending_signal"].get("created_sequence", 0))
            if pending_sequence < self.sequence:
                self._execute_pending(paper, quote)
        completed_bar = self._update_candle(quote, now)
        if completed_bar is not None:
            self._evaluate_completed_bar(paper, completed_bar)
        self._append_account_snapshot(paper)
        self.store.set_state("paper", paper)
        return paper

    def stop(self, paper: dict[str, Any], reason: str, *, killed: bool = False) -> dict[str, Any]:
        paper["pending_signal"] = None
        paper["pending_orders_cancelled"] = True
        if paper.get("position"):
            now = self.clock().astimezone(UTC)
            bid = float(paper.get("last_bid", paper["position"]["current_price"]))
            ask = float(paper.get("last_ask", paper["position"]["current_price"]))
            quote = Quote(
                bid=bid,
                ask=max(ask, bid),
                provider="last-known-paper-mark",
                entitlement="simulated stop mark",
                available_at=now,
            )
            self._close_position(paper, quote, now, f"{reason}-last-known-quote")
        paper.update(
            {
                "status": "killed" if killed else "stopped",
                "feed_status": "stopped",
                "entries_allowed": False,
                "sleep_lease": False,
            }
        )
        self.store.set_state("paper", paper)
        return paper

    def _recover_gap(
        self, paper: dict[str, Any], quote: Quote, received_value: Any
    ) -> None:
        if not received_value:
            paper["reconciliation"] = "feed-recovered-no-gap"
            return
        start = datetime.fromisoformat(str(received_value)).astimezone(UTC)
        if (quote.available_at - start).total_seconds() < 300:
            paper["reconciliation"] = "feed-recovered-no-gap"
            return
        try:
            recovered = asyncio.run(self.provider.backfill("XAUUSD", start, quote.available_at))
        except Exception:
            paper["reconciliation"] = "feed-recovered-backfill-unavailable"
            return
        if not recovered.empty:
            self.base_bars = (
                pd.concat([self.base_bars, recovered], ignore_index=True)
                .drop_duplicates("timestamp", keep="last")
                .sort_values("timestamp")
                .tail(self.settings.live_paper.warmup_bars_5m)
                .reset_index(drop=True)
            )
        paper["reconciliation"] = "backfilled-no-retroactive-trades"
        paper["recovered_bar_count"] = len(recovered)

    def _roll_day(self, paper: dict[str, Any], current: date) -> None:
        marker = date.fromisoformat(str(paper.get("day_marker", current.isoformat())))
        if marker == current:
            return
        history = list(paper.get("daily_history", []))
        history.append(
            {
                "date": marker.isoformat(),
                "close_equity": float(paper.get("equity", paper["balance"])),
            }
        )
        history = history[-5:]
        paper["daily_history"] = history
        paper["day_marker"] = current.isoformat()
        paper["day_start_equity"] = float(paper.get("equity", paper["balance"]))
        values = [float(item["close_equity"]) for item in history] + [
            float(paper["day_start_equity"])
        ]
        paper["rolling_five_day_high"] = max(values)

    def _update_candle(self, quote: Quote, now: datetime) -> Bar | None:
        midpoint = (quote.bid + quote.ask) / 2
        bucket = pd.Timestamp(quote.available_at).floor("5min").to_pydatetime()
        if self.current_candle is None:
            self.current_candle = {
                "start": bucket,
                "open": midpoint,
                "high": midpoint,
                "low": midpoint,
                "close": midpoint,
            }
            return None
        if bucket == self.current_candle["start"]:
            self.current_candle["high"] = max(float(self.current_candle["high"]), midpoint)
            self.current_candle["low"] = min(float(self.current_candle["low"]), midpoint)
            self.current_candle["close"] = midpoint
            return None
        candle = self.current_candle
        end = pd.Timestamp(candle["start"]) + pd.Timedelta(minutes=5)
        completed = Bar(
            timeframe="5min",
            open=float(candle["open"]),
            high=float(candle["high"]),
            low=float(candle["low"]),
            close=float(candle["close"]),
            bid_close=quote.bid,
            ask_close=quote.ask,
            available_at=now,
            occurred_at=end.to_pydatetime(),
            **self.lineage,
        )
        self.store.append_event(completed)
        row = pd.DataFrame(
            [
                {
                    "timestamp": end,
                    "open": completed.open,
                    "high": completed.high,
                    "low": completed.low,
                    "close": completed.close,
                    "bid_close": completed.bid_close,
                    "ask_close": completed.ask_close,
                }
            ]
        )
        self.base_bars = (
            pd.concat([self.base_bars, row], ignore_index=True)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .tail(self.settings.live_paper.warmup_bars_5m)
            .reset_index(drop=True)
        )
        self.current_candle = {
            "start": bucket,
            "open": midpoint,
            "high": midpoint,
            "low": midpoint,
            "close": midpoint,
        }
        return completed

    def _evaluate_completed_bar(self, paper: dict[str, Any], completed: Bar) -> None:
        source_bars = self.base_bars
        if self.report.get("generation_five"):
            from gold_trader.generation5 import normalize_xau_session

            source_bars, _session_evidence = normalize_xau_session(source_bars)
        bars = resample_bars(source_bars, self.strategy.timeframe)
        cutoff = pd.Timestamp(completed.occurred_at)
        eligible = bars[pd.to_datetime(bars["timestamp"], utc=True) <= cutoff]
        if len(eligible) < self.strategy.warmup:
            return
        decision_at = pd.Timestamp(eligible.iloc[-1]["timestamp"]).isoformat()
        if paper.get("last_decision_bar") == decision_at:
            return
        paper["last_decision_bar"] = decision_at
        if not paper.get("entries_allowed") or paper.get("position") or paper.get(
            "pending_signal"
        ):
            return
        signal = self.strategy.generate(eligible)
        if signal is None:
            return
        signal = signal.model_copy(update=self.lineage)
        self.store.append_event(signal)
        paper["pending_signal"] = {
            "signal": signal.model_dump(mode="json"),
            "created_sequence": self.sequence,
            "created_at": self.clock().astimezone(UTC).isoformat(),
            "execution_policy": "next-fresh-quote",
        }

    def _account(self, paper: dict[str, Any]) -> AccountSnapshot:
        return AccountSnapshot(
            balance=float(paper["balance"]),
            equity=float(paper["equity"]),
            day_start_equity=float(paper["day_start_equity"]),
            rolling_five_day_high=float(paper["rolling_five_day_high"]),
            peak_equity=float(paper["peak_equity"]),
            open_position_count=1 if paper.get("position") else 0,
            kill_switch_active=bool(paper.get("kill_switch")),
            **self.lineage,
        )

    def _execute_pending(self, paper: dict[str, Any], quote: Quote) -> None:
        pending = dict(paper["pending_signal"])
        signal = Signal.model_validate(pending["signal"])
        midpoint = (quote.bid + quote.ask) / 2
        sign = 1 if signal.side is Side.BUY else -1
        stop_distance = abs(signal.reference_price - signal.stop_loss)
        target_distance = abs(signal.take_profit - signal.reference_price)
        intent = OrderIntent(
            signal_id=signal.event_id,
            side=signal.side,
            reference_price=midpoint,
            stop_loss=midpoint - sign * stop_distance,
            take_profit=midpoint + sign * target_distance,
            requested_risk_fraction=self.settings.risk.max_risk_fraction,
            causation_id=signal.event_id,
            **self.lineage,
        )
        self.store.append_event(intent)
        decision = self.risk_engine.evaluate(intent, self._account(paper)).model_copy(
            update={"causation_id": intent.event_id, **self.lineage}
        )
        self.store.append_event(decision)
        paper["pending_signal"] = None
        if not decision.approved:
            paper["last_risk_rejection"] = decision.reason
            return
        adverse = (
            self.settings.simulation.spread_points / 2
            + self.settings.simulation.slippage_points
        )
        entry = midpoint + sign * adverse
        stop = entry - sign * stop_distance
        target = entry + sign * target_distance
        order = SimulatedOrder(
            intent_id=intent.event_id,
            side=signal.side,
            lots=decision.lots,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            status=OrderStatus.FILLED,
            causation_id=decision.event_id,
            **self.lineage,
        )
        self.store.append_event(order)
        commission_half = self.settings.simulation.commission_per_lot_round_trip * decision.lots / 2
        fill = Fill(
            order_id=order.event_id,
            side=signal.side,
            lots=decision.lots,
            price=entry,
            commission=commission_half,
            slippage=self.settings.simulation.slippage_points,
            filled_at=quote.available_at,
            causation_id=order.event_id,
            **self.lineage,
        )
        self.store.append_event(fill)
        position = Position(
            side=signal.side,
            lots=decision.lots,
            entry_price=entry,
            current_price=midpoint,
            stop_loss=stop,
            take_profit=target,
            opened_at=quote.available_at,
            causation_id=fill.event_id,
            **self.lineage,
        )
        self.store.append_event(position)
        paper["position"] = position.model_dump(mode="json")
        paper["position"]["trade_id"] = str(uuid4())
        paper["open_position_count"] = 1
        paper["last_entry_at"] = quote.available_at.isoformat()

    def _mark_or_close_position(
        self, paper: dict[str, Any], quote: Quote, now: datetime
    ) -> None:
        position = dict(paper["position"])
        side = Side(position["side"])
        stop = float(position["stop_loss"])
        target = float(position["take_profit"])
        if side is Side.BUY:
            if quote.bid <= stop:
                self._close_position(paper, quote, now, "stop-loss")
                return
            if quote.bid >= target:
                self._close_position(paper, quote, now, "take-profit")
                return
            mark = quote.bid
            sign = 1
        else:
            if quote.ask >= stop:
                self._close_position(paper, quote, now, "stop-loss")
                return
            if quote.ask <= target:
                self._close_position(paper, quote, now, "take-profit")
                return
            mark = quote.ask
            sign = -1
        maximum_hours = self._maximum_holding_hours()
        opened_at = datetime.fromisoformat(str(position["opened_at"])).astimezone(UTC)
        if maximum_hours and (now - opened_at).total_seconds() >= maximum_hours * 3600:
            self._close_position(paper, quote, now, "maximum-holding-time")
            return
        unrealized = (
            (mark - float(position["entry_price"]))
            * sign
            * self.settings.risk.contract_size_oz
            * float(position["lots"])
        )
        position["current_price"] = mark
        position["unrealized_pnl"] = unrealized
        paper["position"] = position
        paper["equity"] = float(paper["balance"]) + unrealized
        paper["peak_equity"] = max(float(paper["peak_equity"]), float(paper["equity"]))

    @staticmethod
    def _generation5_session_open(timestamp: datetime) -> bool:
        selected = timestamp.astimezone(UTC)
        weekday = selected.weekday()
        if weekday <= 3:
            return True
        if weekday == 4:
            return selected.hour < 22
        return weekday == 6 and selected.hour >= 22

    def _maximum_holding_hours(self) -> float:
        bars = int(getattr(self.strategy, "max_holding_bars", 0) or 0)
        if not bars:
            return 0.0
        duration = {"5min": 1 / 12, "15min": 0.25, "1h": 1.0, "4h": 4.0}.get(
            self.strategy.timeframe
        )
        return bars * duration if duration is not None else 0.0

    def _close_position(
        self, paper: dict[str, Any], quote: Quote, now: datetime, reason: str
    ) -> None:
        position = dict(paper["position"])
        side = Side(position["side"])
        sign = 1 if side is Side.BUY else -1
        exit_side = Side.SELL if side is Side.BUY else Side.BUY
        slippage = self.settings.simulation.slippage_points
        exit_price = quote.bid - slippage if side is Side.BUY else quote.ask + slippage
        opened_at = datetime.fromisoformat(str(position["opened_at"])).astimezone(UTC)
        held_days = max((now - opened_at).total_seconds() / 86400, 0)
        lots = float(position["lots"])
        commission = self.settings.simulation.commission_per_lot_round_trip * lots
        swap = self.settings.simulation.swap_per_lot_day * lots * held_days
        gross = (
            (exit_price - float(position["entry_price"]))
            * sign
            * self.settings.risk.contract_size_oz
            * lots
        )
        pnl = gross - commission + swap
        fill = Fill(
            order_id=f"{position['trade_id']}:exit",
            side=exit_side,
            lots=lots,
            price=exit_price,
            commission=commission / 2,
            slippage=slippage,
            filled_at=now,
            **self.lineage,
        )
        self.store.append_event(fill)
        closed = PaperTradeClosed(
            trade_id=str(position["trade_id"]),
            side=side,
            lots=lots,
            entry_price=float(position["entry_price"]),
            exit_price=exit_price,
            opened_at=opened_at,
            closed_at=now,
            pnl=pnl,
            commission=commission,
            swap=swap,
            reason=reason,
            causation_id=fill.event_id,
            **self.lineage,
        )
        self.store.append_event(closed)
        paper["balance"] = float(paper["balance"]) + pnl
        paper["equity"] = float(paper["balance"])
        paper["peak_equity"] = max(float(paper["peak_equity"]), float(paper["equity"]))
        paper["position"] = None
        paper["open_position_count"] = 0
        paper["observation_trade_count"] = int(paper.get("observation_trade_count", 0)) + 1
        paper["last_trade"] = closed.model_dump(mode="json")

    def _enforce_account_halts(self, paper: dict[str, Any]) -> None:
        equity = float(paper["equity"])
        reason: str | None = None
        if equity <= float(paper["day_start_equity"]) * (
            1 - self.settings.risk.daily_loss_fraction
        ):
            reason = "daily-loss-halt"
        elif equity <= float(paper["rolling_five_day_high"]) * (
            1 - self.settings.risk.rolling_five_day_loss_fraction
        ):
            reason = "rolling-five-day-halt"
        elif equity <= float(paper["peak_equity"]) * (
            1 - self.settings.risk.peak_drawdown_fraction
        ):
            reason = "peak-drawdown-lock"
        if reason is not None:
            first = paper.get("risk_halt_reason") != reason
            paper.update(
                {
                    "status": "risk-halted",
                    "entries_allowed": False,
                    "risk_halt_reason": reason,
                }
            )
            if first:
                self.notifier.notify("Aurum risk halt", reason.replace("-", " "))

    def _append_account_snapshot(self, paper: dict[str, Any]) -> None:
        self.store.append_event(self._account(paper))
