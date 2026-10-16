from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import gold_trader.generation5 as generation5
from gold_trader.config import Settings
from gold_trader.data import file_sha256, generate_demo_data
from gold_trader.database import Store
from gold_trader.generation5 import (
    Generation5Coordinator,
    Generation5Simulator,
    Generation5Worker,
    create_generation5_manifest,
    generation5_trial_specs,
    load_generation5_manifest,
    normalize_xau_session,
    seal_generation5_manifest,
)
from gold_trader.service import ApplicationService
from gold_trader.simulator import TradeResult
from gold_trader.strategies import BaseStrategy, StrategyParameters
from gold_trader.worker import PaperWorker


class FrequentTestStrategy(BaseStrategy):
    strategy_id = "frequent-test"
    timeframe = "1h"
    warmup = 1
    max_holding_bars = 4

    def __init__(self) -> None:
        super().__init__(
            StrategyParameters(
                lookback=1,
                atr_period=1,
                stop_atr=10.0,
                reward_risk=2.0,
            )
        )

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "side": ["buy"] * len(bars),
                "reason": ["frequency test"] * len(bars),
                "atr": [1.0] * len(bars),
            },
            index=bars.index,
        )


def healthy(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "healthy": True,
        "checks": {"on_ac_power": True, "free_disk": True, "system_load": True},
        "free_disk_gb": 100.0,
        "load": 0.1,
    }


def prepare_parent(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: int = 20_000,
) -> tuple[Path, Path]:
    monkeypatch.setattr(Settings, "project_root", property(lambda _self: tmp_path))
    dataset = tmp_path / "var/data/bars.parquet"
    dataset.parent.mkdir(parents=True)
    bars = generate_demo_data(rows=rows)
    bars.to_parquet(dataset, index=False)
    timestamps = pd.to_datetime(bars["timestamp"], utc=True)
    holdout = timestamps.max() - pd.DateOffset(days=30)
    parent_manifest = {
        "dataset_path": str(dataset),
        "dataset_hash": file_sha256(dataset),
        "dataset_start": timestamps.min().isoformat(),
        "dataset_end": timestamps.max().isoformat(),
        "holdout_start": holdout.isoformat(),
    }
    store.save_generation(
        "generation-4-test",
        "p" * 64,
        "completed",
        parent_manifest,
        {"completed_trials": 4},
    )
    learning = tmp_path / "var/reports/generation-5-learning.json"
    manifest = tmp_path / "var/generations/generation-5.json"
    return learning, manifest


def test_session_normalization_removes_weekend_bars() -> None:
    start = datetime(2026, 7, 17, 20, tzinfo=UTC)
    timestamps = [start + timedelta(hours=offset) for offset in range(60)]
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 2400.0,
            "high": 2401.0,
            "low": 2399.0,
            "close": 2400.0,
        }
    )

    normalized, evidence = normalize_xau_session(bars)

    selected = pd.to_datetime(normalized["timestamp"], utc=True)
    assert not (selected.dt.dayofweek == 5).any()
    assert evidence["removed_rows"] > 0
    assert evidence["policy"] == "utc-sunday-22-through-friday-22"


def test_frequency_simulator_enforces_maximum_holding_time(settings: Settings) -> None:
    timestamps = pd.date_range("2026-01-05", periods=80, freq="1h", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 2400.0,
            "high": 2400.5,
            "low": 2399.5,
            "close": 2400.0,
        }
    )

    trades, metrics = Generation5Simulator(
        settings.risk, settings.simulation, settings.starting_equity
    ).run(bars, FrequentTestStrategy())

    assert metrics["trade_count"] >= 10
    assert all(trade.reason == "maximum-holding-time" for trade in trades[:-1])
    assert trades[-1].reason == "end-of-data"
    assert all((trade.exit_at - trade.entry_at) <= timedelta(hours=3) for trade in trades)


def test_generation5_manifest_is_deterministic_and_tamper_evident(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learning, path = prepare_parent(tmp_path, settings, store, monkeypatch)
    draft = create_generation5_manifest(
        settings,
        store,
        "generation-4-test",
        learning,
        path,
        workers=1,
        trials_per_family=2,
        batch_size=2,
    )
    assert set(draft.families) == {
        "compression-breakout",
        "time-series-momentum",
        "trend-pullback",
        "mean-reversion",
    }
    sealed = seal_generation5_manifest(path, store)
    first = generation5_trial_specs(sealed)
    second = generation5_trial_specs(load_generation5_manifest(path))
    assert [item.trial_hash for item in first] == [item.trial_hash for item in second]
    assert len(first) == 8

    report = json.loads(learning.read_text(encoding="utf-8"))
    report["lessons"].append("tampered")
    learning.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="learning report"):
        load_generation5_manifest(path)


def test_generation5_coordinator_and_command_worker_checkpoint(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learning, path = prepare_parent(tmp_path, settings, store, monkeypatch)
    create_generation5_manifest(
        settings,
        store,
        "generation-4-test",
        learning,
        path,
        workers=1,
        trials_per_family=1,
        batch_size=2,
    )
    manifest = seal_generation5_manifest(path, store)
    monkeypatch.setattr("gold_trader.generation5.resource_health", healthy)
    coordinator = Generation5Coordinator(settings, store)
    coordinator.sleep_lease.acquire = lambda: False  # type: ignore[method-assign]

    result = coordinator.run(path)

    assert result["status"] == "completed"
    assert result["completed_trials"] == 4
    assert len(store.list_trials(manifest.generation_id)) == 4

    store.save_generation(
        manifest.generation_id,
        manifest.manifest_hash,
        "completed",
        manifest.model_dump(mode="json"),
        {"manifest_path": str(path), "completed_trials": 4},
    )
    command_id = ApplicationService(store, settings).start_generation5(path)
    called: list[Path] = []

    def fake_run(_self: Generation5Coordinator, selected: Path) -> dict[str, Any]:
        called.append(selected)
        return {"status": "completed"}

    worker = Generation5Worker(settings, store)
    monkeypatch.setattr("gold_trader.generation5.Generation5Coordinator.run", fake_run)
    monkeypatch.setattr("time.sleep", lambda _seconds: setattr(worker, "running", False))
    worker.run_forever(poll_seconds=0)

    assert called == [path.resolve()]
    assert command_id > 0


def test_generation5_finalization_is_session_normalized_and_replayable(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learning, path = prepare_parent(tmp_path, settings, store, monkeypatch, rows=30_000)
    create_generation5_manifest(
        settings,
        store,
        "generation-4-test",
        learning,
        path,
        workers=1,
        trials_per_family=1,
        batch_size=1,
    )
    manifest = seal_generation5_manifest(path, store)
    spec = generation5_trial_specs(manifest)[0]
    store.save_trial(
        manifest.generation_id,
        spec.model_dump(mode="json"),
        {
            "status": "shortlisted",
            "score": 1.0,
            "fold_profitable_fraction": 0.8,
            "largest_profit_fraction": 0.3,
            "selection_statistics": {"deflated_sharpe_probability": 0.9},
        },
    )
    store.save_generation(
        manifest.generation_id,
        manifest.manifest_hash,
        "completed",
        manifest.model_dump(mode="json"),
        {"completed_trials": 1, "manifest_path": str(path)},
    )
    external = tmp_path / "var/data/external.parquet"
    generate_demo_data(rows=30_000, seed=77).to_parquet(external, index=False)

    report = ApplicationService(store, settings).finalize_strategy_choice(
        manifest.generation_id, spec.trial_hash, external
    )

    assert report["generation_five"]["holdout_opened_once"] is True
    assert report["generation_five"]["adaptive_period_reuse"] is True
    assert report["data"]["primary_session_normalization"]["policy"].startswith("utc-")
    assert report["data"]["external_session_normalization"]["removed_rows"] > 0
    assert "monthly_activity" in report["gates"]
    assert (
        ApplicationService(store, settings).finalize_strategy_choice(
            manifest.generation_id, spec.trial_hash, external
        )
        == report
    )
    resolved = PaperWorker(settings, store)._resolve_approved_strategy(report)
    assert resolved.strategy_hash == report["strategy_hash"]


def test_generation5_full_frequency_gate_evaluation(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learning, path = prepare_parent(tmp_path, settings, store, monkeypatch)
    draft = create_generation5_manifest(
        settings,
        store,
        "generation-4-test",
        learning,
        path,
        workers=1,
        trials_per_family=1,
        batch_size=1,
    )
    sealed = seal_generation5_manifest(path)
    spec = generation5_trial_specs(sealed)[0]
    anchor = sealed.holdout_start
    trades: list[TradeResult] = []
    for year_offset in range(5, 0, -1):
        year_start = pd.Timestamp(anchor) - pd.DateOffset(years=year_offset)
        for ordinal in range(50):
            entry = (year_start + pd.Timedelta(days=ordinal * 6)).to_pydatetime()
            trades.append(
                TradeResult(
                    trade_id=f"{year_offset}-{ordinal}",
                    strategy_id="frequency-fixture",
                    side="buy",
                    entry_at=entry,
                    exit_at=entry + timedelta(hours=2),
                    entry_price=2400.0,
                    exit_price=2401.0,
                    stop_loss=2390.0,
                    take_profit=2420.0,
                    lots=0.01,
                    pnl=1.0,
                    commission=0.01,
                    swap=0.0,
                    reason="take-profit",
                )
            )
    metrics = {
        "net_return": 0.50,
        "net_pnl": 5000.0,
        "profit_factor": 2.0,
        "sharpe": 1.5,
        "max_drawdown": 0.02,
        "trade_count": 400.0,
        "win_rate": 0.6,
    }

    monkeypatch.setattr(
        generation5.Generation5Simulator,
        "run",
        lambda _self, _bars, _strategy: (trades, metrics),
    )
    monkeypatch.setattr(generation5, "_fold_pnl", lambda *_args: [10, 11, 9, 12, 8, 10])
    generation5._initialize_worker(
        draft.dataset_path,
        draft.settings_snapshot,
        draft.holdout_start.isoformat(),
        draft.frequency_policy,
    )

    result = generation5._evaluate_trial(spec.model_dump(mode="json"))

    assert result["stage"] == 2
    assert result["annual_trade_counts"] == [50, 50, 50, 50, 50]
    assert result["gates"]["annual_trade_floor"] is True
    assert result["holding_statistics"]["median_hours"] == 2.0
