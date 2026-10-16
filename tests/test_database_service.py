from __future__ import annotations

from pathlib import Path

from gold_trader.config import Settings
from gold_trader.data import generate_demo_data
from gold_trader.database import Store
from gold_trader.generation import create_manifest, seal_manifest
from gold_trader.service import ApplicationService


def eligible_report(candidate_hash: str = "a" * 64) -> dict[str, object]:
    return {
        "run_id": "run-1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "strategy_id": "trend-pullback",
        "timeframe": "1h",
        "candidate_hash": candidate_hash,
        "dataset_hash": "d" * 64,
        "configuration_hash": "c" * 64,
        "strategy_hash": "s" * 64,
        "model_hash": "m" * 64,
        "status": "eligible",
        "eligible": True,
        "metrics": {"net_return": 0.1},
        "gates": {"all": True},
    }


def test_command_queue_is_idempotent(store: Store) -> None:
    first = store.enqueue_command("paper.stop", {}, "same-key")
    second = store.enqueue_command("paper.stop", {}, "same-key")
    assert first == second
    command = store.claim_next_command()
    assert command is not None and command["id"] == first


def test_interrupted_commands_recover_only_to_their_worker_queue(store: Store) -> None:
    paper = store.enqueue_command("paper.stop", {}, "paper-recovery")
    research = store.enqueue_command("research.generation.run", {}, "research-recovery")
    assert store.claim_next_command("paper.")["id"] == paper  # type: ignore[index]
    assert store.claim_next_command("research.")["id"] == research  # type: ignore[index]
    assert store.recover_processing_commands("paper.") == 1
    recovered = store.claim_next_command("paper.")
    assert recovered is not None and recovered["id"] == paper
    assert store.claim_next_command("paper.") is None


def test_candidate_requires_exact_hash_and_eligibility(store: Store) -> None:
    service = ApplicationService(store)
    report = eligible_report()
    store.save_research_run(report)
    approval = service.approve_candidate("a" * 64, "a" * 64)
    assert approval.candidate_hash == "a" * 64
    assert store.valid_approval() is not None


def test_kill_switch_blocks_start(store: Store) -> None:
    service = ApplicationService(store)
    store.save_research_run(eligible_report())
    service.approve_candidate("a" * 64, "a" * 64)
    service.activate_kill_switch("test")
    try:
        service.start_paper()
    except ValueError as exc:
        assert "kill switch" in str(exc)
    else:
        raise AssertionError("start unexpectedly passed an active kill switch")


def test_generation_command_is_bound_to_the_exact_sealed_manifest(
    tmp_path: Path, settings: Settings, store: Store
) -> None:
    dataset = tmp_path / "bars.csv"
    manifest_path = tmp_path / "generation.json"
    generate_demo_data(rows=1000).to_csv(dataset, index=False)
    create_manifest(settings, dataset, manifest_path, workers=1, trials_per_family=1)
    manifest = seal_manifest(manifest_path, store)

    command_id = ApplicationService(store, settings).start_generation(manifest_path)
    command = store.claim_next_command("research.")

    assert command is not None and command["id"] == command_id
    assert command["payload"]["generation_id"] == manifest.generation_id
    assert command["payload"]["manifest_hash"] == manifest.manifest_hash


def test_running_generations_are_marked_interrupted_on_worker_restart(store: Store) -> None:
    store.save_generation("orphan", "h" * 64, "running", {}, {"completed_trials": 12})

    assert store.interrupt_running_generations("test restart") == 1
    generation = store.get_generation("orphan")
    assert generation is not None
    assert generation["status"] == "interrupted"
    assert generation["summary"]["interruption_reason"] == "test restart"
