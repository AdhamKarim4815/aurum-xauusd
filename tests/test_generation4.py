from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gold_trader.config import Settings
from gold_trader.data import generate_demo_data
from gold_trader.database import Store
from gold_trader.generation import create_manifest, seal_manifest, trial_specs
from gold_trader.generation4 import (
    Generation4Coordinator,
    Generation4Worker,
    create_generation4_manifest,
    generation4_trial_specs,
    load_generation4_manifest,
    seal_generation4_manifest,
)
from gold_trader.service import ApplicationService
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


def completed_generation3(
    tmp_path: Path, settings: Settings, store: Store, *, rows: int = 1500
) -> tuple[Path, str]:
    dataset = tmp_path / "bars.parquet"
    generate_demo_data(rows=rows).to_parquet(dataset, index=False)
    path = tmp_path / "generation-3.json"
    create_manifest(
        settings,
        dataset,
        path,
        generation_id="generation-3-parent",
        workers=1,
        trials_per_family=1,
        batch_size=2,
    )
    manifest = seal_manifest(path, store)
    specs = trial_specs(manifest)
    result = {
        "status": "shortlisted",
        "stage": 2,
        "score": 1.0,
        "metrics": {
            "net_return": 0.2,
            "profit_factor": 1.3,
            "sharpe": 1.0,
            "max_drawdown": 0.05,
            "trade_count": 80,
        },
        "stressed_metrics": {"net_return": 0.1},
        "fold_profitable_fraction": 0.8,
        "largest_profit_fraction": 0.3,
    }
    for family in ("compression-breakout", "time-series-momentum"):
        spec = next(item for item in specs if item.family == family)
        store.save_trial(manifest.generation_id, spec.model_dump(mode="json"), result)
    store.save_generation(
        manifest.generation_id,
        manifest.manifest_hash,
        "completed",
        manifest.model_dump(mode="json"),
        {"completed_trials": 6, "shortlisted": 2, "manifest_path": str(path)},
    )
    return path, manifest.generation_id


def test_generation4_learns_seals_and_detects_tampering(
    tmp_path: Path, settings: Settings, store: Store
) -> None:
    _parent_path, parent_id = completed_generation3(tmp_path, settings, store)
    learning = tmp_path / "learning.json"
    path = tmp_path / "generation-4.json"
    draft = create_generation4_manifest(
        settings,
        store,
        parent_id,
        learning,
        path,
        workers=1,
        trials_per_family=2,
        batch_size=2,
    )
    assert set(draft.families) == {"compression-breakout", "time-series-momentum"}
    assert draft.parent_generation_id == parent_id
    assert draft.holdout_start is not None
    report = json.loads(learning.read_text(encoding="utf-8"))
    assert report["holdout_opened"] is False
    assert report["buy_hold_discovery"]

    sealed = seal_generation4_manifest(path, store)
    loaded = load_generation4_manifest(path)
    first = generation4_trial_specs(loaded)
    second = generation4_trial_specs(loaded)
    assert sealed.manifest_hash == loaded.manifest_hash
    assert [item.trial_hash for item in first] == [item.trial_hash for item in second]
    assert len(first) == 4

    learning_payload = json.loads(learning.read_text(encoding="utf-8"))
    learning_payload["lessons"].append("tampered")
    learning.write_text(json.dumps(learning_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="learning report"):
        load_generation4_manifest(path)


def test_generation4_coordinator_checkpoints_locally(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _parent_path, parent_id = completed_generation3(tmp_path, settings, store)
    learning = tmp_path / "learning.json"
    path = tmp_path / "generation-4.json"
    create_generation4_manifest(
        settings,
        store,
        parent_id,
        learning,
        path,
        workers=1,
        trials_per_family=1,
        batch_size=1,
    )
    seal_generation4_manifest(path, store)
    monkeypatch.setattr("gold_trader.generation4.resource_health", healthy)
    coordinator = Generation4Coordinator(settings, store)
    coordinator.sleep_lease = FakeLease()  # type: ignore[assignment]

    result = coordinator.run(path)

    assert result["status"] == "completed"
    assert result["completed_trials"] == 2
    assert len(store.list_trials("generation-4")) == 2


def test_generation4_finalization_is_one_time_and_replayable(
    tmp_path: Path, settings: Settings, store: Store
) -> None:
    _parent_path, parent_id = completed_generation3(tmp_path, settings, store, rows=8000)
    learning = tmp_path / "learning.json"
    path = tmp_path / "generation-4.json"
    create_generation4_manifest(
        settings,
        store,
        parent_id,
        learning,
        path,
        workers=1,
        trials_per_family=1,
        batch_size=1,
    )
    manifest = seal_generation4_manifest(path, store)
    spec = generation4_trial_specs(manifest)[0]
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
    store.save_trial(manifest.generation_id, spec.model_dump(mode="json"), discovery)
    external = tmp_path / "external.parquet"
    generate_demo_data(rows=8000, seed=22).to_parquet(external, index=False)
    coordinator = Generation4Coordinator(settings, store)

    report = coordinator.finalize(path, spec.trial_hash, external)

    assert report["generation_four"]["holdout_opened_once"] is True
    assert report["benchmark"]["years"] > 0
    assert store.get_research_run(report["candidate_hash"]) is not None
    assert coordinator.finalize(path, spec.trial_hash, external) == report
    strategy = PaperWorker(settings, store)._resolve_approved_strategy(report)
    assert strategy.strategy_hash == report["strategy_hash"]
    store.save_generation(
        manifest.generation_id,
        manifest.manifest_hash,
        "completed",
        manifest.model_dump(mode="json"),
        {"completed_trials": 2, "manifest_path": str(path)},
    )
    assert (
        ApplicationService(store, settings).finalize_strategy_choice(
            manifest.generation_id, spec.trial_hash, external
        )
        == report
    )


def test_generation4_rejects_primary_dataset_as_external(
    tmp_path: Path, settings: Settings, store: Store
) -> None:
    _parent_path, parent_id = completed_generation3(tmp_path, settings, store)
    learning = tmp_path / "learning.json"
    path = tmp_path / "generation-4.json"
    create_generation4_manifest(
        settings,
        store,
        parent_id,
        learning,
        path,
        workers=1,
        trials_per_family=1,
        batch_size=1,
    )
    manifest = seal_generation4_manifest(path, store)
    spec = generation4_trial_specs(manifest)[0]
    store.save_trial(
        manifest.generation_id,
        spec.model_dump(mode="json"),
        {"status": "shortlisted", "score": 1.0},
    )

    with pytest.raises(ValueError, match="independently sourced"):
        Generation4Coordinator(settings, store).finalize(
            path, spec.trial_hash, Path(manifest.dataset_path)
        )


def test_generation4_resource_pause_and_command_worker(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _parent_path, parent_id = completed_generation3(tmp_path, settings, store)
    learning = tmp_path / "learning.json"
    path = tmp_path / "generation-4.json"
    create_generation4_manifest(
        settings,
        store,
        parent_id,
        learning,
        path,
        workers=1,
        trials_per_family=1,
        batch_size=1,
    )
    manifest = seal_generation4_manifest(path, store)
    monkeypatch.setattr(
        "gold_trader.generation4.resource_health",
        lambda *_args, **_kwargs: {"healthy": False, "checks": {}},
    )
    paused = Generation4Coordinator(settings, store).run(path)
    assert paused["status"] == "resource-paused"

    service = ApplicationService(store, settings)
    service.control_generation4("pause")
    service.control_generation4("stop")
    with pytest.raises(ValueError, match="must be pause or stop"):
        service.control_generation4("invalid")
    command_id = service.start_generation4(path)
    called: list[Path] = []

    def fake_run(_self: Generation4Coordinator, manifest_path: Path) -> dict[str, Any]:
        called.append(manifest_path)
        return {"status": "completed"}

    worker = Generation4Worker(settings, store)
    monkeypatch.setattr("gold_trader.generation4.Generation4Coordinator.run", fake_run)
    monkeypatch.setattr("time.sleep", lambda _seconds: setattr(worker, "running", False))
    worker.run_forever(poll_seconds=0)

    assert called == [path]
    assert store.claim_next_command("generation4.") is None
    assert command_id > 0
    assert manifest.manifest_hash
