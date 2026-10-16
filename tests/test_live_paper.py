from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from gold_trader.config import Settings
from gold_trader.data import file_sha256, generate_demo_data
from gold_trader.database import Store
from gold_trader.domain import Quote
from gold_trader.live_paper import LivePaperEngine
from gold_trader.service import ApplicationService
from gold_trader.strategies import BaseStrategy, StrategyParameters


class AlwaysBuyStrategy(BaseStrategy):
    strategy_id = "always-buy-test"
    timeframe = "5min"
    warmup = 1

    def __init__(self) -> None:
        super().__init__(StrategyParameters(lookback=1, stop_atr=1.0, reward_risk=2.0))

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "side": ["buy"] * len(bars),
                "reason": ["test signal"] * len(bars),
                "atr": [1.0] * len(bars),
            },
            index=bars.index,
        )


class TimedAlwaysBuyStrategy(AlwaysBuyStrategy):
    timeframe = "1h"
    max_holding_bars = 1


class FakeProvider:
    name = "fake"

    def __init__(self, quotes: list[Quote | Exception]) -> None:
        self.quotes = iter(quotes)
        self.backfills = 0

    async def quote(self, symbol: str = "XAUUSD") -> Quote:
        selected = next(self.quotes)
        if isinstance(selected, Exception):
            raise selected
        return selected

    async def backfill(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        self.backfills += 1
        return pd.DataFrame()


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def approved_report(dataset: Path, candidate_hash: str = "a" * 64) -> dict[str, object]:
    return {
        "run_id": "live-run",
        "created_at": "2026-01-01T00:00:00+00:00",
        "strategy_id": "always-buy-test",
        "timeframe": "5min",
        "candidate_hash": candidate_hash,
        "dataset_hash": file_sha256(dataset),
        "configuration_hash": "c" * 64,
        "strategy_hash": "s" * 64,
        "model_hash": "m" * 64,
        "status": "eligible",
        "eligible": True,
        "metrics": {"net_return": 0.1},
        "gates": {"all": True},
        "data": {"dataset_path": str(dataset)},
    }


def quote(at: datetime, price: float) -> Quote:
    return Quote(
        bid=price,
        ask=price,
        provider="fake",
        entitlement="test",
        available_at=at,
    )


def test_continuous_live_paper_executes_next_quote_and_closes_simulated_trade(
    tmp_path: Path, settings: Settings, store: Store
) -> None:
    dataset = tmp_path / "bars.parquet"
    generate_demo_data(rows=600).to_parquet(dataset, index=False)
    report = approved_report(dataset)
    store.save_research_run(report)
    ApplicationService(store, settings).approve_candidate("a" * 64, "a" * 64)
    start = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    times = [
        start,
        start + timedelta(minutes=5),
        start + timedelta(minutes=6),
        start + timedelta(minutes=7),
    ]
    provider = FakeProvider(
        [
            quote(times[0], 2400),
            quote(times[1], 2400),
            quote(times[2], 2400),
            quote(times[3], 2410),
        ]
    )
    clock = MutableClock(times[0])
    engine = LivePaperEngine(settings, store, provider, report, AlwaysBuyStrategy(), clock=clock)
    paper = {"status": "running", "kill_switch": False}
    engine.prepare(paper)

    engine.poll_once()
    clock.value = times[1]
    after_signal = engine.poll_once()
    assert after_signal["pending_signal"] is not None
    assert after_signal.get("position") is None
    assert after_signal["live_provider"] == "fake"
    clock.value = times[2]
    entered = engine.poll_once()
    assert entered["position"] is not None
    clock.value = times[3]
    completed = engine.poll_once()

    assert completed["position"] is None
    assert completed["observation_trade_count"] == 1
    assert completed["simulated_fills_only"] is True
    event_types = [event["type"] for event in store.audit(limit=100)]
    assert "RiskDecision" in event_types
    assert "SimulatedOrder" in event_types
    assert "PaperTradeClosed" in event_types


def test_stale_quote_blocks_new_entries(tmp_path: Path, settings: Settings, store: Store) -> None:
    dataset = tmp_path / "bars.parquet"
    generate_demo_data(rows=600).to_parquet(dataset, index=False)
    report = approved_report(dataset, "b" * 64)
    store.save_research_run(report)
    ApplicationService(store, settings).approve_candidate("b" * 64, "b" * 64)
    now = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    provider = FakeProvider([quote(now - timedelta(minutes=10), 2400)])
    engine = LivePaperEngine(
        settings,
        store,
        provider,
        report,
        AlwaysBuyStrategy(),
        clock=MutableClock(now),
    )
    engine.prepare({"status": "running", "kill_switch": False})

    paper = engine.poll_once()

    assert paper["feed_status"] == "stale"
    assert paper["entries_allowed"] is False
    assert paper.get("position") is None


def test_generation5_session_and_maximum_holding_time(
    tmp_path: Path, settings: Settings, store: Store
) -> None:
    dataset = tmp_path / "bars.parquet"
    generate_demo_data(rows=600).to_parquet(dataset, index=False)
    report = approved_report(dataset, "f" * 64)
    report["generation_five"] = {"adaptive_period_reuse": True}
    now = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    engine = LivePaperEngine(
        settings,
        store,
        FakeProvider([]),
        report,
        TimedAlwaysBuyStrategy(),
        clock=MutableClock(now),
    )
    paper = {
        "balance": 10_000.0,
        "equity": 10_000.0,
        "peak_equity": 10_000.0,
        "observation_trade_count": 0,
        "position": {
            "side": "buy",
            "lots": 0.01,
            "entry_price": 2400.0,
            "current_price": 2400.0,
            "stop_loss": 2300.0,
            "take_profit": 2500.0,
            "opened_at": (now - timedelta(hours=2)).isoformat(),
            "trade_id": "timed-paper-trade",
        },
    }

    engine._mark_or_close_position(paper, quote(now, 2401.0), now)

    assert paper["position"] is None
    assert paper["last_trade"]["reason"] == "maximum-holding-time"
    saturday = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
    sunday_open = datetime(2026, 7, 26, 22, 0, tzinfo=UTC)
    assert engine._generation5_session_open(saturday) is False
    assert engine._generation5_session_open(sunday_open) is True


def test_network_recovery_backfills_without_trades_and_invalidation_blocks(
    tmp_path: Path, settings: Settings, store: Store
) -> None:
    dataset = tmp_path / "bars.parquet"
    generate_demo_data(rows=600).to_parquet(dataset, index=False)
    report = approved_report(dataset, "c" * 64)
    store.save_research_run(report)
    ApplicationService(store, settings).approve_candidate("c" * 64, "c" * 64)
    now = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    provider = FakeProvider([RuntimeError("offline"), quote(now, 2400)])
    clock = MutableClock(now)
    engine = LivePaperEngine(settings, store, provider, report, AlwaysBuyStrategy(), clock=clock)
    engine.prepare(
        {
            "status": "running",
            "kill_switch": False,
            "day_marker": (now.date() - timedelta(days=1)).isoformat(),
        }
    )

    failed = engine.poll_once()
    assert failed["feed_status"] == "network-error"
    failed["last_quote_received_at"] = (now - timedelta(minutes=10)).isoformat()
    store.set_state("paper", failed)
    recovered = engine.poll_once()
    assert recovered["reconciliation"] == "backfilled-no-retroactive-trades"
    assert recovered["observation_trade_count"] == 0
    assert provider.backfills == 1
    assert len(recovered["daily_history"]) == 1

    store.invalidate_approvals()
    blocked = engine.poll_once()
    assert blocked["status"] == "approval-invalidated"
    assert blocked["pending_signal"] is None


def test_stop_flattens_open_simulated_position_at_last_mark(
    tmp_path: Path, settings: Settings, store: Store
) -> None:
    dataset = tmp_path / "bars.parquet"
    generate_demo_data(rows=600).to_parquet(dataset, index=False)
    report = approved_report(dataset, "d" * 64)
    now = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
    engine = LivePaperEngine(
        settings,
        store,
        FakeProvider([]),
        report,
        AlwaysBuyStrategy(),
        clock=MutableClock(now),
    )
    paper = engine.prepare(
        {
            "status": "running",
            "kill_switch": False,
            "last_bid": 2401.0,
            "last_ask": 2401.2,
            "position": {
                "trade_id": "paper-trade",
                "side": "buy",
                "lots": 0.1,
                "entry_price": 2400.0,
                "current_price": 2401.0,
                "stop_loss": 2395.0,
                "take_profit": 2410.0,
                "opened_at": (now - timedelta(hours=1)).isoformat(),
            },
        }
    )

    stopped = engine.stop(paper, "operator-stop")

    assert stopped["status"] == "stopped"
    assert stopped["position"] is None
    assert stopped["observation_trade_count"] == 1
    assert "last-known-quote" in stopped["last_trade"]["reason"]
