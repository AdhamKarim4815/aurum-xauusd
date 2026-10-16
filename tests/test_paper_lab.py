from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gold_trader.config import Settings
from gold_trader.data import file_sha256, generate_demo_data
from gold_trader.database import Store
from gold_trader.domain import Quote
from gold_trader.paper_lab import (
    PAPER_LAB_STATE,
    PaperLabStateStore,
    PaperLabWorker,
    SharedCycleQuoteProvider,
    paper_lab_account_key,
    validate_experimental_report,
)
from gold_trader.service import ApplicationService
from gold_trader.strategies import BaseStrategy, StrategyParameters
from gold_trader.web import create_app


class CountingProvider:
    name = "counting"

    def __init__(self, quote: Quote) -> None:
        self.value = quote
        self.calls = 0

    async def quote(self, symbol: str = "XAUUSD") -> Quote:
        self.calls += 1
        return self.value

    async def backfill(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return pd.DataFrame()


class RecoveringProvider:
    name = "recovering"

    def __init__(self, outcomes: list[Quote | Exception]) -> None:
        self.outcomes = iter(outcomes)
        self.calls = 0
        self.backfills = 0

    async def quote(self, symbol: str = "XAUUSD") -> Quote:
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def backfill(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        self.backfills += 1
        return pd.DataFrame()


class NoSignalStrategy(BaseStrategy):
    strategy_id = "paper-lab-test"
    timeframe = "5min"
    warmup = 1

    def __init__(self) -> None:
        super().__init__(StrategyParameters(lookback=1, stop_atr=1.0, reward_risk=2.0))

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"side": [None] * len(bars), "reason": [None] * len(bars)},
            index=bars.index,
        )


def lab_report(
    candidate_hash: str,
    dataset: Path | None = None,
    *,
    failed_gate: str = "trade_count",
    discovery_return: float = 0.2,
) -> dict[str, Any]:
    trial_hash = candidate_hash[::-1]
    report: dict[str, Any] = {
        "run_id": f"lab-{candidate_hash[:8]}",
        "created_at": "2026-01-01T00:00:00+00:00",
        "strategy_id": "time-series-momentum",
        "variant_id": f"lab-{candidate_hash[:8]}",
        "timeframe": "4h",
        "candidate_hash": candidate_hash,
        "dataset_hash": file_sha256(dataset) if dataset else "d" * 64,
        "configuration_hash": "c" * 64,
        "strategy_hash": "s" * 64,
        "model_hash": "m" * 64,
        "status": "ineligible",
        "eligible": False,
        "metrics": {
            "net_return": 0.06,
            "profit_factor": 2.0,
            "sharpe": 2.0,
            "max_drawdown": 0.02,
            "trade_count": 8,
        },
        "external_metrics": {"net_return": 0.04},
        "gates": {"positive_return": True, failed_gate: False},
        "generation_four": {
            "trial_spec": {
                "generation_id": "generation-4",
                "trial_hash": trial_hash,
                "family": "time-series-momentum",
                "parameters": {"timeframe": "4h", "lookback": 120},
            },
            "discovery_result": {"metrics": {"net_return": discovery_return, "trade_count": 100}},
        },
    }
    if dataset:
        report["data"] = {"dataset_path": str(dataset)}
    return report


def test_shared_cycle_provider_fetches_once_and_clones_event_ids() -> None:
    provider = CountingProvider(
        Quote(
            bid=2400,
            ask=2400.2,
            provider="test",
            entitlement="test",
            available_at=datetime.now(UTC),
        )
    )
    shared = SharedCycleQuoteProvider(provider)

    assert shared.name == "counting"
    assert shared.active_provider_name is None
    first = asyncio.run(shared.quote())
    second = asyncio.run(shared.quote())
    assert provider.calls == 1
    assert first.event_id != second.event_id

    shared.begin_cycle()
    asyncio.run(shared.quote())
    assert provider.calls == 2
    assert asyncio.run(shared.backfill("XAUUSD", datetime.now(UTC), datetime.now(UTC))).empty

    failing = CountingProvider(provider.value)

    async def fail(_symbol: str = "XAUUSD") -> Quote:
        failing.calls += 1
        raise RuntimeError("offline")

    failing.quote = fail  # type: ignore[method-assign]
    shared_failure = SharedCycleQuoteProvider(failing)
    with pytest.raises(RuntimeError, match="offline"):
        asyncio.run(shared_failure.quote())
    with pytest.raises(RuntimeError, match="offline"):
        asyncio.run(shared_failure.quote())
    assert failing.calls == 1


def test_experimental_override_only_accepts_activity_failures() -> None:
    validate_experimental_report(lab_report("a" * 64))
    validate_experimental_report(lab_report("b" * 64, failed_gate="monthly_activity"))
    with pytest.raises(ValueError, match="non-overridable"):
        validate_experimental_report(lab_report("c" * 64, failed_gate="drawdown"))


def test_namespaced_store_supplies_exact_lineage_and_correlates_events(
    store: Store,
) -> None:
    report = lab_report("a" * 64)
    wrapper = PaperLabStateStore(store, "a" * 64, report)
    wrapper.set_state("paper", {"status": "running"})
    wrapper.set_state("other", {"value": 1})

    assert wrapper.get_state("paper") == {"status": "running"}
    assert wrapper.get_state("other") == {"value": 1}
    assert wrapper.valid_approval()["candidate_hash"] == "a" * 64
    wrapper.append_event(
        Quote(
            bid=2400,
            ask=2400.2,
            provider="test",
            entitlement="test",
            available_at=datetime.now(UTC),
        )
    )
    event = store.audit(limit=1)[0]["payload"]
    assert event["correlation_id"].startswith("paper-lab:")


def test_service_requires_acknowledgement_and_exactly_four(
    settings: Settings, store: Store
) -> None:
    hashes = [str(index) * 64 for index in range(1, 5)]
    for index, candidate_hash in enumerate(hashes):
        store.save_research_run(lab_report(candidate_hash, discovery_return=0.4 - index * 0.01))
    service = ApplicationService(store, settings)

    with pytest.raises(ValueError, match="acknowledgement"):
        service.start_paper_lab(hashes, acknowledged=False)
    with pytest.raises(ValueError, match="exactly four"):
        service.start_paper_lab(hashes[:3], acknowledged=True)

    command_id = service.start_paper_lab(hashes, acknowledged=True)
    command = store.claim_next_command("paperlab.")
    assert command is not None and command["id"] == command_id
    assert command["payload"]["candidate_hashes"] == hashes

    store.set_state(PAPER_LAB_STATE, {"status": "running", "candidate_hashes": hashes})
    for candidate_hash in hashes:
        store.set_state(paper_lab_account_key(candidate_hash), {"candidate_hash": candidate_hash})
    status = service.paper_lab_status()
    assert status["status"] == "running"
    assert len(status["accounts"]) == 4


def test_worker_runs_four_isolated_accounts_with_one_quote_request(
    tmp_path: Path, settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "bars.parquet"
    generate_demo_data(rows=600).to_parquet(dataset, index=False)
    hashes = [str(index) * 64 for index in range(1, 5)]
    for candidate_hash in hashes:
        store.save_research_run(lab_report(candidate_hash, dataset))
    provider = CountingProvider(
        Quote(
            bid=2400,
            ask=2400.2,
            provider="test",
            entitlement="test",
            available_at=datetime.now(UTC),
        )
    )
    worker = PaperLabWorker(settings, store)
    monkeypatch.setattr(worker, "_build_live_provider", lambda: provider)
    monkeypatch.setattr(worker, "_resolve_strategy", lambda _report: NoSignalStrategy())
    monkeypatch.setattr(worker.sleep_lease, "acquire", lambda: False)

    started = worker.start(hashes)
    polled = worker.poll_once()

    assert started["strategy_count"] == 4
    assert provider.calls == 1
    assert polled["healthy_feeds"] == 4
    assert polled["entries_allowed"] == 4
    states = [store.get_state(paper_lab_account_key(value)) for value in hashes]
    assert all(state is not None and state["status"] == "running" for state in states)
    assert len({state["candidate_hash"] for state in states if state}) == 4

    stopped = worker.stop()
    assert stopped["status"] == "stopped"
    assert all(
        store.get_state(paper_lab_account_key(value))["status"] == "stopped"  # type: ignore[index]
        for value in hashes
    )


def test_worker_pauses_for_network_and_resumes_automatically(
    tmp_path: Path, settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "bars.parquet"
    generate_demo_data(rows=600).to_parquet(dataset, index=False)
    hashes = [str(index) * 64 for index in range(1, 5)]
    for candidate_hash in hashes:
        store.save_research_run(lab_report(candidate_hash, dataset))
    provider = RecoveringProvider(
        [
            ConnectionError("offline"),
            ConnectionError("still offline"),
            Quote(
                bid=2400,
                ask=2400.2,
                provider="test",
                entitlement="test",
                available_at=datetime.now(UTC),
            ),
        ]
    )
    worker = PaperLabWorker(settings, store)
    monkeypatch.setattr(worker, "_build_live_provider", lambda: provider)
    monkeypatch.setattr(worker, "_resolve_strategy", lambda _report: NoSignalStrategy())
    monkeypatch.setattr(worker.sleep_lease, "acquire", lambda: False)
    worker.start(hashes)

    first_pause = worker.poll_once()
    assert first_pause["status"] == "paused-network"
    assert first_pause["connection_status"] == "offline"
    assert first_pause["retry_seconds"] == 10
    assert first_pause["entries_allowed"] == 0
    assert first_pause["network_pause_count"] == 1
    assert provider.calls == 1
    assert all(
        store.get_state(paper_lab_account_key(value))["status"] == "paused-network"  # type: ignore[index]
        for value in hashes
    )

    second_pause = worker.poll_once()
    assert second_pause["status"] == "paused-network"
    assert second_pause["retry_seconds"] == 20
    assert second_pause["network_pause_count"] == 1
    assert provider.calls == 2

    recovered = worker.poll_once()
    assert recovered["status"] == "running"
    assert recovered["connection_status"] == "online"
    assert recovered["healthy_feeds"] == 4
    assert recovered["entries_allowed"] == 4
    assert recovered["retry_seconds"] is None
    assert recovered["resumed_at"]
    assert recovered["reconciliation"] == (
        "network-restored-backfill-checked-no-retroactive-trades"
    )
    assert provider.calls == 3
    assert all(
        store.get_state(paper_lab_account_key(value))["status"] == "running"  # type: ignore[index]
        for value in hashes
    )


def test_worker_restores_all_four_together_and_fails_closed(
    tmp_path: Path, settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "bars.parquet"
    generate_demo_data(rows=600).to_parquet(dataset, index=False)
    hashes = [str(index) * 64 for index in range(1, 5)]
    for candidate_hash in hashes:
        store.save_research_run(lab_report(candidate_hash, dataset))
    provider = CountingProvider(
        Quote(
            bid=2400,
            ask=2400.2,
            provider="test",
            entitlement="test",
            available_at=datetime.now(UTC),
        )
    )
    starter = PaperLabWorker(settings, store)
    monkeypatch.setattr(starter, "_build_live_provider", lambda: provider)
    monkeypatch.setattr(starter, "_resolve_strategy", lambda _report: NoSignalStrategy())
    monkeypatch.setattr(starter.sleep_lease, "acquire", lambda: False)
    starter.start(hashes)

    restored = PaperLabWorker(settings, store)
    monkeypatch.setattr(restored, "_build_live_provider", lambda: provider)
    monkeypatch.setattr(restored, "_resolve_strategy", lambda _report: NoSignalStrategy())
    monkeypatch.setattr(restored.sleep_lease, "acquire", lambda: False)
    restored.restore_if_needed()
    assert restored.status()["reconciliation"] == "worker-restored-no-duplicate-fills"
    assert len(restored.status()["accounts"]) == 4

    store.set_state(PAPER_LAB_STATE, {"status": "paused-network", "candidate_hashes": hashes})
    paused_restore = PaperLabWorker(settings, store)
    monkeypatch.setattr(paused_restore, "_build_live_provider", lambda: provider)
    monkeypatch.setattr(paused_restore, "_resolve_strategy", lambda _report: NoSignalStrategy())
    monkeypatch.setattr(paused_restore.sleep_lease, "acquire", lambda: False)
    paused_restore.restore_if_needed()
    assert len(paused_restore.status()["accounts"]) == 4

    store.set_state(PAPER_LAB_STATE, {"status": "running", "candidate_hashes": ["missing"] * 4})
    failed = PaperLabWorker(settings, store)
    failed.restore_if_needed()
    assert failed.status()["status"] == "recovery-required"
    assert failed.status()["reconciliation"].startswith("restore-failed:")


def test_worker_command_loop_processes_start_and_polls_once(
    tmp_path: Path, settings: Settings, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "bars.parquet"
    generate_demo_data(rows=600).to_parquet(dataset, index=False)
    hashes = [str(index) * 64 for index in range(1, 5)]
    for candidate_hash in hashes:
        store.save_research_run(lab_report(candidate_hash, dataset))
    provider = CountingProvider(
        Quote(
            bid=2400,
            ask=2400.2,
            provider="test",
            entitlement="test",
            available_at=datetime.now(UTC),
        )
    )
    worker = PaperLabWorker(settings, store)
    monkeypatch.setattr(worker, "_build_live_provider", lambda: provider)
    monkeypatch.setattr(worker, "_resolve_strategy", lambda _report: NoSignalStrategy())
    monkeypatch.setattr(worker.sleep_lease, "acquire", lambda: False)
    monkeypatch.setattr("gold_trader.paper_lab.signal.signal", lambda *_args: None)
    monkeypatch.setattr(
        "gold_trader.paper_lab.time.sleep",
        lambda _seconds: setattr(worker, "running", False),
    )
    command_id = store.enqueue_command("paperlab.start", {"candidate_hashes": hashes}, "loop-test")

    worker.run_forever()

    assert provider.calls == 1
    assert worker.status()["last_poll_at"] is not None
    assert store.claim_next_command("paperlab.") is None
    with store.sessions() as session:
        from gold_trader.database import CommandRow

        assert session.get(CommandRow, command_id).status == "completed"  # type: ignore[union-attr]


def test_paper_lab_dashboard_and_csrf_start(settings: Settings, store: Store) -> None:
    hashes = [str(index) * 64 for index in range(1, 5)]
    for index, candidate_hash in enumerate(hashes):
        store.save_research_run(lab_report(candidate_hash, discovery_return=0.4 - index * 0.01))
    client = TestClient(create_app(settings, store))

    page = client.get("/paper-lab")

    assert page.status_code == 200
    assert "up to four simultaneous XAUUSD positions" in page.text
    assert page.text.count('name="candidate_hashes"') == 4
    assert page.text.count("checked") == 4
    token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
    assert token is not None
    response = client.post(
        "/api/paper-lab/start",
        data={
            "csrf_token": token.group(1),
            "candidate_hashes": hashes,
            "acknowledge": "paper-only",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/paper-lab"
    assert client.get("/api/paper-lab").json()["status"] == "stopped"
    assert store.claim_next_command("paperlab.") is not None
