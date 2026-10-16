from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gold_trader.config import Settings
from gold_trader.data import generate_demo_data
from gold_trader.database import Store
from gold_trader.generation import (
    FAMILY_SPACES,
    GenerationCoordinator,
    GenerationStrategy,
    _bootstrap_p_value,
    _deflated_sharpe_probability,
    _probability_of_backtest_overfit,
    create_manifest,
    explain_trial_result,
    load_sealed_manifest,
    resource_health,
    seal_manifest,
    trial_specs,
)
from gold_trader.worker import PaperWorker


class FakeLease:
    active = False

    def acquire(self) -> bool:
        self.active = True
        return True

    def release(self) -> None:
        self.active = False


def healthy(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "healthy": True,
        "checks": {"on_ac_power": True, "free_disk": True, "system_load": True},
        "free_disk_gb": 100.0,
        "load": 0.1,
    }


def test_manifest_is_deterministic_sealed_and_tamper_evident(
    tmp_path: Path, settings: Settings, store: Store
) -> None:
    dataset = tmp_path / "bars.csv"
    generate_demo_data(rows=1000).to_csv(dataset, index=False)
    path = tmp_path / "generation.json"
    draft = create_manifest(settings, dataset, path, workers=1, trials_per_family=2, batch_size=2)
    assert draft.status == "draft"
    assert "test-session-secret" not in path.read_text(encoding="utf-8")
    sealed = seal_manifest(path, store)
    loaded = load_sealed_manifest(path)
    assert loaded.manifest_hash == sealed.manifest_hash
    first = trial_specs(loaded)
    second = trial_specs(loaded)
    assert [item.trial_hash for item in first] == [item.trial_hash for item in second]
    assert len({item.trial_hash for item in first}) == len(first) == 12
    assert {item.family for item in first[:6]} == set(FAMILY_SPACES)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["batch_size"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        load_sealed_manifest(path)


def test_coordinator_checkpoints_trials_and_command_routing(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "bars.csv"
    generate_demo_data(rows=1500).to_csv(dataset, index=False)
    path = tmp_path / "generation.json"
    create_manifest(settings, dataset, path, workers=2, trials_per_family=1, batch_size=2)
    seal_manifest(path, store)
    monkeypatch.setattr("gold_trader.generation.resource_health", healthy)
    coordinator = GenerationCoordinator(settings, store)
    coordinator.sleep_lease = FakeLease()  # type: ignore[assignment]

    result = coordinator.run(path)

    assert result["status"] == "completed"
    assert result["completed_trials"] == 6
    stored = store.list_trials()
    assert len(stored) == 6
    store.save_trial(
        "generation-3",
        stored[0]["spec"],
        {"status": "rejected", "score": 0, "metrics": {"net_return": -0.2}},
    )
    store.save_trial(
        "generation-3",
        stored[1]["spec"],
        {"status": "shortlisted", "score": 1, "metrics": {"net_return": 0.3}},
    )
    leaderboard = store.list_trials(limit=6, sort_by_return=True)
    returns = [item["result"]["metrics"]["net_return"] for item in leaderboard]
    assert returns == sorted(returns, reverse=True)
    assert leaderboard[0]["spec"]["trial_hash"] == stored[1]["spec"]["trial_hash"]
    research_id = store.enqueue_command("research.generation.run", {}, "research-only")
    paper_id = store.enqueue_command("paper.stop", {}, "paper-only")
    assert store.claim_next_command("paper.")["id"] == paper_id  # type: ignore[index]
    assert store.claim_next_command("research.")["id"] == research_id  # type: ignore[index]


def test_finalize_opens_holdout_once_and_builds_replayable_evidence(
    tmp_path: Path, settings: Settings, store: Store
) -> None:
    primary = tmp_path / "primary.parquet"
    external = tmp_path / "external.parquet"
    generate_demo_data(rows=8000, seed=11).to_parquet(primary, index=False)
    generate_demo_data(rows=8000, seed=12).to_parquet(external, index=False)
    path = tmp_path / "generation.json"
    create_manifest(settings, primary, path, workers=1, trials_per_family=2, batch_size=2)
    manifest = seal_manifest(path, store)
    specs = trial_specs(manifest)
    discovery = {
        "status": "shortlisted",
        "stage": 2,
        "score": 1.0,
        "metrics": {"net_return": 0.1},
        "stressed_metrics": {"net_return": 0.05},
        "fold_pnl": [10, 11, 9, 12, 8, 10],
        "fold_profitable_fraction": 1.0,
        "largest_profit_fraction": 0.2,
        "selection_statistics": {
            "deflated_sharpe_probability": 0.9,
            "stationary_bootstrap_p_value": 0.05,
        },
    }
    store.save_trial(manifest.generation_id, specs[0].model_dump(mode="json"), discovery)
    coordinator = GenerationCoordinator(settings, store)

    report = coordinator.finalize(path, specs[0].trial_hash, external)

    assert report["generation_three"]["holdout_opened_once"] is True
    assert store.get_research_run(report["candidate_hash"]) is not None
    assert coordinator.finalize(path, specs[0].trial_hash, external) == report
    resolved = PaperWorker(settings, store)._resolve_approved_strategy(report)
    assert resolved.strategy_hash == report["strategy_hash"]

    store.save_trial(manifest.generation_id, specs[1].model_dump(mode="json"), discovery)
    with pytest.raises(ValueError, match="independently sourced"):
        coordinator.finalize(path, specs[1].trial_hash, primary)


def test_selection_statistics_and_resource_diagnostics_are_deterministic(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = [float(index - 3) for index in range(20)]
    assert _bootstrap_p_value(values, 42) == _bootstrap_p_value(values, 42)
    assert _bootstrap_p_value(values[:3], 42) == 1.0
    assert 0 <= _deflated_sharpe_probability(1.2, 100, 200) <= 1
    assert _deflated_sharpe_probability(1.0, 2, 10) == 0.0
    trials = [
        {"result": {"fold_pnl": [offset + value for value in range(6)]}}
        for offset in (0.0, 1.0, -1.0)
    ]
    assert 0 <= _probability_of_backtest_overfit(trials) <= 1
    assert _probability_of_backtest_overfit([]) == 1.0
    monkeypatch.setattr("gold_trader.generation.sys.platform", "linux")
    diagnostics = resource_health(settings, workers=6)
    assert set(diagnostics["checks"]) == {"on_ac_power", "free_disk", "system_load"}


def test_generation_strategy_rejects_an_unknown_family() -> None:
    with pytest.raises(ValueError, match="unknown Generation 3 family"):
        GenerationStrategy(
            "unbounded-transformer",
            {"timeframe": "5min", "lookback": 10},
            "f" * 64,
        )


def test_rejected_trial_explains_the_failed_doubled_cost_gate() -> None:
    evidence = explain_trial_result(
        {"parameters": {"timeframe": "4h"}},
        {
            "status": "rejected",
            "stage": 2,
            "metrics": {
                "net_return": 0.0447,
                "profit_factor": 1.075,
                "sharpe": 0.618,
                "max_drawdown": 0.0503,
                "trade_count": 268,
            },
            "stressed_metrics": {"net_return": -0.02269},
            "fold_profitable_fraction": 0.667,
            "largest_profit_fraction": 0.417,
            "gates": {
                "positive_return": True,
                "positive_stressed_return": False,
                "profit_factor": True,
                "sharpe": True,
                "drawdown": True,
                "trade_count": True,
                "fold_stability": True,
                "profit_concentration": True,
            },
        },
    )

    assert evidence["failed_reasons"] == ["Doubled-cost net return was -2.27%; required > 0.00%."]
    assert "Rejected because" in evidence["summary"]
