from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gold_trader.config import Settings
from gold_trader.data import file_sha256, generate_demo_data
from gold_trader.database import Store
from gold_trader.ml import FEATURE_COLUMNS, ChronologicalSignalMetaModel, MetaFilteredStrategy
from gold_trader.notifications import MacNotifier
from gold_trader.research import stable_hash
from gold_trader.service import ApplicationService
from gold_trader.sleep import SleepLease
from gold_trader.strategies import strategy_registry
from gold_trader.worker import PaperWorker

from .test_database_service import eligible_report


class FakeLease:
    active = False

    def acquire(self) -> bool:
        self.active = True
        return True

    def release(self) -> None:
        self.active = False


class FakeNotifier:
    messages: list[tuple[str, str]]

    def __init__(self) -> None:
        self.messages = []

    def notify(self, title: str, message: str) -> bool:
        self.messages.append((title, message))
        return True


class FakeLiveProvider:
    async def quote(self, symbol: str = "XAUUSD") -> object:
        raise RuntimeError("not polled during worker lifecycle test")

    async def backfill(self, symbol: str, start: object, end: object) -> pd.DataFrame:
        return pd.DataFrame()


def approved_worker(settings: Settings, store: Store) -> PaperWorker:
    report = eligible_report()
    report["configuration_hash"] = stable_hash(settings.model_dump())
    report["strategy_hash"] = strategy_registry()["trend-pullback"].strategy_hash
    store.save_research_run(report)
    ApplicationService(store).approve_candidate("a" * 64, "a" * 64)
    worker = PaperWorker(settings, store)
    worker.sleep_lease = FakeLease()  # type: ignore[assignment]
    worker.notifier = FakeNotifier()  # type: ignore[assignment]
    return worker


def test_worker_start_stop_and_kill(settings: Settings, store: Store) -> None:
    worker = approved_worker(settings, store)
    started = worker._handle("paper.start", {"candidate_hash": "a" * 64, "mode": "replay"})
    assert started["status"] == "running"
    assert started["sleep_lease"] is True
    stopped = worker._handle("paper.stop", {})
    assert stopped["status"] == "stopped"
    killed = worker._handle("paper.kill", {"reason": "worker test"})
    assert killed["status"] == "killed"
    assert killed["pending_orders_cancelled"] is True
    with pytest.raises(ValueError, match="unsupported command"):
        worker._handle("real.order", {})


def test_worker_completes_accelerated_historical_replay(
    tmp_path: Path, settings: Settings, store: Store
) -> None:
    dataset = tmp_path / "approved.csv"
    generate_demo_data(rows=5000).to_csv(dataset, index=False)
    report = eligible_report("b" * 64)
    report["configuration_hash"] = stable_hash(settings.model_dump())
    report["strategy_hash"] = strategy_registry()["trend-pullback"].strategy_hash
    report["data"] = {"dataset_path": str(dataset)}
    store.save_research_run(report)
    ApplicationService(store).approve_candidate("b" * 64, "b" * 64)
    worker = PaperWorker(settings, store)
    worker.sleep_lease = FakeLease()  # type: ignore[assignment]
    worker.notifier = FakeNotifier()  # type: ignore[assignment]
    completed = worker._handle("paper.start", {"candidate_hash": "b" * 64, "mode": "replay"})
    assert completed["status"] == "replay-complete"
    assert completed["historical_replay_only"] is True
    assert completed["observation_trade_count"] == 0
    assert completed["sleep_lease"] is False


def test_worker_replays_exact_registered_variant(
    tmp_path: Path, settings: Settings, store: Store
) -> None:
    dataset = tmp_path / "approved.csv"
    generate_demo_data(rows=5000).to_csv(dataset, index=False)
    report = eligible_report("c" * 64)
    report["strategy_id"] = "time-series-momentum"
    report["variant_id"] = "time-series-momentum-l20"
    report["configuration_hash"] = stable_hash(settings.model_dump())
    report["strategy_hash"] = strategy_registry()["time-series-momentum-l20"].strategy_hash
    report["data"] = {"dataset_path": str(dataset)}
    store.save_research_run(report)
    ApplicationService(store).approve_candidate("c" * 64, "c" * 64)
    worker = PaperWorker(settings, store)
    worker.sleep_lease = FakeLease()  # type: ignore[assignment]
    worker.notifier = FakeNotifier()  # type: ignore[assignment]

    completed = worker._handle("paper.start", {"candidate_hash": "c" * 64, "mode": "replay"})

    assert completed["status"] == "replay-complete"


def test_worker_loads_only_the_checksum_matched_approved_model(
    tmp_path: Path, settings: Settings, store: Store, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(Settings, "project_root", property(lambda _self: tmp_path))
    rows = 400
    events = pd.DataFrame(
        {
            column: [((index + position) % 17) / 17 for index in range(rows)]
            for position, column in enumerate(FEATURE_COLUMNS)
        }
    )
    events["timestamp"] = pd.date_range("2020-01-01", periods=rows, freq="h", tz="UTC")
    events["label"] = [index % 2 for index in range(rows)]
    model = ChronologicalSignalMetaModel().fit(events)
    candidate_hash = "f" * 64
    artifact = tmp_path / "artifacts/models" / f"{candidate_hash}.joblib"
    model.save(artifact)
    report = eligible_report(candidate_hash)
    report.update(
        {
            "candidate_hash": candidate_hash,
            "base_variant_id": "trend-pullback-ema30-80",
            "model": {"model": ChronologicalSignalMetaModel.model_id},
            "model_hash": model.model_hash,
            "model_artifact": {
                "path": str(artifact),
                "sha256": file_sha256(artifact),
            },
        }
    )
    worker = PaperWorker(settings, store)

    resolved = worker._resolve_approved_strategy(report)

    assert isinstance(resolved, MetaFilteredStrategy)
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        worker._resolve_approved_strategy(report)


def test_worker_starts_restores_stops_and_kills_continuous_live_paper(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "approved.parquet"
    generate_demo_data(rows=600).to_parquet(dataset, index=False)
    candidate_hash = "e" * 64
    report = eligible_report(candidate_hash)
    report.update(
        {
            "variant_id": "trend-pullback",
            "dataset_hash": file_sha256(dataset),
            "configuration_hash": stable_hash(settings.model_dump()),
            "strategy_hash": strategy_registry()["trend-pullback"].strategy_hash,
            "data": {"dataset_path": str(dataset)},
        }
    )
    store.save_research_run(report)
    ApplicationService(store, settings).approve_candidate(candidate_hash, candidate_hash)
    worker = PaperWorker(settings, store, provider_factory=lambda: FakeLiveProvider())
    worker.sleep_lease = FakeLease()  # type: ignore[assignment]
    worker.notifier = FakeNotifier()  # type: ignore[assignment]

    started = worker._handle("paper.start", {"candidate_hash": candidate_hash, "mode": "live"})
    assert started["status"] == "running"
    assert started["feed_status"] == "connecting"
    assert worker.live_engine is not None
    stopped = worker._handle("paper.stop", {})
    assert stopped["status"] == "stopped"
    assert worker.live_engine is None

    paper = store.get_state("paper") or {}
    paper.update({"status": "running", "mode": "live", "candidate_hash": candidate_hash})
    store.set_state("paper", paper)
    restored = PaperWorker(settings, store, provider_factory=lambda: FakeLiveProvider())
    restored.sleep_lease = FakeLease()  # type: ignore[assignment]
    restored.notifier = FakeNotifier()  # type: ignore[assignment]
    restored._restore_live_if_needed()
    assert restored.live_engine is not None
    assert store.get_state("paper")["reconciliation"] == (  # type: ignore[index]
        "worker-restored-no-duplicate-fills"
    )
    killed = restored._handle("paper.kill", {"reason": "test live kill"})
    assert killed["status"] == "killed"
    assert restored.live_engine is None

    paper = store.get_state("paper") or {}
    paper.update(
        {
            "status": "running",
            "mode": "live",
            "candidate_hash": candidate_hash,
            "kill_switch": False,
        }
    )
    store.set_state("paper", paper)
    store.invalidate_approvals()
    blocked = PaperWorker(settings, store, provider_factory=lambda: FakeLiveProvider())
    blocked._restore_live_if_needed()
    assert store.get_state("paper")["status"] == "recovery-required"  # type: ignore[index]


def test_sleep_and_notification_noop_off_mac(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("gold_trader.sleep.sys.platform", "linux")
    monkeypatch.setattr("gold_trader.notifications.sys.platform", "linux")
    lease = SleepLease()
    assert lease.acquire() is False
    assert lease.active is False
    lease.release()
    assert MacNotifier().notify("title", "message") is False
