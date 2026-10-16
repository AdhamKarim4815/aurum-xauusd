from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import gold_trader.generation6 as generation6
from gold_trader.config import Settings
from gold_trader.data import file_sha256, generate_demo_data
from gold_trader.database import Store
from gold_trader.domain import Approval
from gold_trader.generation6 import (
    Generation6Coordinator,
    Generation6PaperWorker,
    Generation6Strategy,
    Generation6Worker,
    create_generation6_manifest,
    generation6_trial_specs,
    load_generation6_manifest,
    seal_generation6_manifest,
)
from gold_trader.service import ApplicationService
from gold_trader.simulator import TradeResult


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
    parent_manifest = {
        "dataset_path": str(dataset),
        "dataset_hash": file_sha256(dataset),
        "dataset_start": timestamps.min().isoformat(),
        "dataset_end": timestamps.max().isoformat(),
        "holdout_start": (timestamps.max() - pd.DateOffset(days=30)).isoformat(),
    }
    store.save_generation(
        "generation-5-test",
        "5" * 64,
        "futile",
        parent_manifest,
        {"completed_trials": 16, "shortlisted": 0},
    )
    store.save_trial(
        "generation-5-test",
        {
            "generation_id": "generation-5-test",
            "trial_hash": "a" * 64,
            "family": "compression-breakout",
            "ordinal": 0,
            "parameters": {"timeframe": "1h"},
        },
        {
            "status": "pruned",
            "score": 0.1,
            "metrics": {"net_return": 0.05, "trade_count": 40},
            "gate_details": [
                {"key": "recent_trade_count", "passed": False},
            ],
        },
    )
    return (
        tmp_path / "var/reports/generation-6-learning.json",
        tmp_path / "var/generations/generation-6.json",
    )


def test_generation6_trend_state_produces_persistent_signals() -> None:
    timestamps = pd.date_range("2026-01-05", periods=240, freq="15min", tz="UTC")
    close = pd.Series([2400 + index * 0.4 for index in range(len(timestamps))])
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        }
    )
    strategy = Generation6Strategy(
        "trend-state",
        {
            "timeframe": "15min",
            "atr_period": 10,
            "stop_atr": 1.25,
            "reward_risk": 1.5,
            "max_holding_bars": 8,
            "direction": "long-only",
            "fast_window": 8,
            "slow_window": 35,
            "slope_window": 3,
            "minimum_slope_atr": 0.25,
        },
        "t" * 64,
    )

    candidates = strategy.precompute(bars)

    assert candidates["side"].eq("buy").sum() > 100
    assert not candidates["side"].eq("sell").any()
    assert strategy.max_holding_bars == 8


def test_generation6_manifest_is_deterministic_and_tamper_evident(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learning, path = prepare_parent(tmp_path, settings, store, monkeypatch)
    draft = create_generation6_manifest(
        settings,
        store,
        "generation-5-test",
        learning,
        path,
        workers=1,
        trials_per_family=2,
        batch_size=2,
    )
    assert set(draft.families) == {
        "trend-state",
        "momentum-consensus",
        "rsi-recovery",
        "session-trend",
    }
    sealed = seal_generation6_manifest(path, store)
    first = generation6_trial_specs(sealed)
    second = generation6_trial_specs(load_generation6_manifest(path))
    assert [item.trial_hash for item in first] == [item.trial_hash for item in second]
    assert len(first) == 8

    report = json.loads(learning.read_text(encoding="utf-8"))
    report["lessons"].append("tampered")
    learning.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="learning report"):
        load_generation6_manifest(path)


def test_generation6_coordinator_checkpoints_all_small_search_trials(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learning, path = prepare_parent(tmp_path, settings, store, monkeypatch)
    create_generation6_manifest(
        settings,
        store,
        "generation-5-test",
        learning,
        path,
        workers=1,
        trials_per_family=1,
        batch_size=2,
    )
    manifest = seal_generation6_manifest(path, store)
    monkeypatch.setattr("gold_trader.generation6.resource_health", healthy)
    coordinator = Generation6Coordinator(settings, store)
    coordinator.sleep_lease.acquire = lambda: False  # type: ignore[method-assign]

    result = coordinator.run(path)

    assert result["status"] == "completed"
    assert result["completed_trials"] == 4
    assert len(store.list_trials(manifest.generation_id)) == 4


def test_generation6_uses_isolated_paper_command_and_strategy_resolution(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learning, path = prepare_parent(tmp_path, settings, store, monkeypatch)
    create_generation6_manifest(
        settings,
        store,
        "generation-5-test",
        learning,
        path,
        workers=1,
        trials_per_family=1,
        batch_size=1,
    )
    manifest = seal_generation6_manifest(path, store)
    spec = generation6_trial_specs(manifest)[0].model_dump(mode="json")
    strategy = Generation6Strategy(spec["family"], spec["parameters"], spec["trial_hash"])
    candidate_hash = "c" * 64
    report = {
        "run_id": "generation-6-paper-test",
        "strategy_id": strategy.strategy_id,
        "candidate_hash": candidate_hash,
        "dataset_hash": manifest.dataset_hash,
        "configuration_hash": "q" * 64,
        "strategy_hash": strategy.strategy_hash,
        "model_hash": "m" * 64,
        "status": "eligible",
        "eligible": True,
        "metrics": {"net_return": 0.1},
        "generation_six": {"trial_spec": spec},
    }
    store.save_research_run(report)
    store.create_approval(
        Approval(
            candidate_hash=candidate_hash,
            dataset_hash=manifest.dataset_hash,
            configuration_hash="q" * 64,
            strategy_hash=strategy.strategy_hash,
            model_hash="m" * 64,
        )
    )

    command_id = ApplicationService(store, settings).start_paper("replay")
    command = store.claim_next_command("paper6.")
    resolved = Generation6PaperWorker(settings, store)._resolve_approved_strategy(report)

    assert command_id > 0
    assert command is not None
    assert command["type"] == "paper6.start"
    assert resolved.strategy_hash == strategy.strategy_hash


def test_generation6_finalization_records_new_lineage_and_runtime_compatibility(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learning, path = prepare_parent(tmp_path, settings, store, monkeypatch, rows=30_000)
    create_generation6_manifest(
        settings,
        store,
        "generation-5-test",
        learning,
        path,
        workers=1,
        trials_per_family=1,
        batch_size=1,
    )
    manifest = seal_generation6_manifest(path, store)
    spec = generation6_trial_specs(manifest)[0]
    store.save_trial(
        manifest.generation_id,
        spec.model_dump(mode="json"),
        {
            "status": "shortlisted",
            "score": 1.0,
            "fold_profitable_fraction": 0.8,
            "largest_profit_fraction": 0.3,
            "selection_statistics": {
                "deflated_sharpe_probability": 0.9,
                "stationary_bootstrap_p_value": 0.05,
            },
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
    generate_demo_data(rows=30_000, seed=71).to_parquet(external, index=False)

    report = ApplicationService(store, settings).finalize_strategy_choice(
        manifest.generation_id, spec.trial_hash, external
    )

    assert report["generation_six"]["holdout_opened_once"] is True
    assert report["generation_six"]["new_hypothesis_lineage"] is True
    assert report["generation_five"]["runtime_compatibility"].startswith("shared-")
    assert "bootstrap_significance" in report["gates"]
    assert report["data"]["primary_session_normalization"]["policy"].startswith("utc-")
    resolved = Generation6PaperWorker(settings, store)._resolve_approved_strategy(report)
    assert resolved.strategy_hash == report["strategy_hash"]


def test_generation6_full_activity_gate_evaluation(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learning, path = prepare_parent(tmp_path, settings, store, monkeypatch)
    draft = create_generation6_manifest(
        settings,
        store,
        "generation-5-test",
        learning,
        path,
        workers=1,
        trials_per_family=1,
        batch_size=1,
    )
    sealed = seal_generation6_manifest(path)
    spec = generation6_trial_specs(sealed)[0]
    anchor = sealed.holdout_start
    trades: list[TradeResult] = []
    for year_offset in range(5, 0, -1):
        year_start = pd.Timestamp(anchor) - pd.DateOffset(years=year_offset)
        for ordinal in range(60):
            entry = (year_start + pd.Timedelta(days=ordinal * 5)).to_pydatetime()
            trades.append(
                TradeResult(
                    trade_id=f"{year_offset}-{ordinal}",
                    strategy_id="generation-6-fixture",
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
        "trade_count": 500.0,
        "win_rate": 0.6,
    }

    monkeypatch.setattr(
        generation6.Generation5Simulator,
        "run",
        lambda _self, _bars, _strategy: (trades, metrics),
    )
    monkeypatch.setattr(generation6, "_fold_pnl", lambda *_args: [10, 11, 9, 12, 8, 10])
    generation6._initialize_worker(
        draft.dataset_path,
        draft.settings_snapshot,
        draft.holdout_start.isoformat(),
        draft.activity_policy,
    )

    result = generation6._evaluate_trial(spec.model_dump(mode="json"))

    assert result["stage"] == 2
    assert result["annual_trade_counts"] == [60, 60, 60, 60, 60]
    assert result["gates"]["annual_trade_floor"] is True
    assert result["gates"]["trade_count"] is True
    assert result["holding_statistics"]["median_hours"] == 2.0


def test_generation6_command_workers_use_their_isolated_queues(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learning, path = prepare_parent(tmp_path, settings, store, monkeypatch)
    create_generation6_manifest(
        settings,
        store,
        "generation-5-test",
        learning,
        path,
        workers=1,
        trials_per_family=1,
        batch_size=1,
    )
    manifest = seal_generation6_manifest(path, store)
    command_id = ApplicationService(store, settings).start_generation6(path)
    called: list[Path] = []

    def fake_run(_self: Generation6Coordinator, selected: Path) -> dict[str, Any]:
        called.append(selected)
        return {"status": "completed"}

    research_worker = Generation6Worker(settings, store)
    monkeypatch.setattr("gold_trader.generation6.Generation6Coordinator.run", fake_run)
    monkeypatch.setattr(
        "gold_trader.generation6.time.sleep",
        lambda _seconds: setattr(research_worker, "running", False),
    )
    research_worker.run_forever(poll_seconds=0)

    assert command_id > 0
    assert called == [path.resolve()]
    assert store.claim_next_command("generation6.") is None
    assert manifest.manifest_hash

    store.set_state(
        "paper",
        {
            "status": "stopped",
            "mode": "replay",
            "entries_allowed": False,
            "sleep_lease": False,
        },
    )
    store.enqueue_command("paper6.stop", {}, "generation-6-paper-stop")
    paper_worker = Generation6PaperWorker(settings, store)
    monkeypatch.setattr(
        "gold_trader.generation6.time.sleep",
        lambda _seconds: setattr(paper_worker, "running", False),
    )

    paper_worker.run_forever(poll_seconds=0)

    paper = store.get_state("paper")
    assert paper is not None
    assert paper["status"] == "stopped"
    assert paper["worker_prefix"] == "paper6."
