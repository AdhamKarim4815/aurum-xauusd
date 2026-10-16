from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gold_trader.config import Settings
from gold_trader.database import Store
from gold_trader.service import ApplicationService
from gold_trader.validation import ValidationWorker, validation_state_key
from gold_trader.web import create_app


def prepare_shortlist(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, str, Path]:
    monkeypatch.setattr(Settings, "project_root", property(lambda _self: tmp_path))
    generation_id = "generation-4-test"
    trial_hash = "t" * 64
    manifest = tmp_path / "var/generations/generation-4.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    external = tmp_path / "var/data/external.parquet"
    external.parent.mkdir(parents=True)
    external.write_bytes(b"independent fixture")
    store.save_generation(
        generation_id,
        "g" * 64,
        "completed",
        {},
        {"manifest_path": str(manifest)},
    )
    store.save_trial(
        generation_id,
        {
            "generation_id": generation_id,
            "trial_hash": trial_hash,
            "family": "compression-breakout",
            "ordinal": 1,
            "parameters": {},
        },
        {"status": "shortlisted", "score": 1.0},
    )
    return generation_id, trial_hash, external


def test_validation_queue_progress_and_durable_result(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_id, trial_hash, external = prepare_shortlist(tmp_path, settings, store, monkeypatch)
    service = ApplicationService(store, settings)
    command_id = service.queue_strategy_validation(generation_id, trial_hash, external)
    assert service.queue_strategy_validation(generation_id, trial_hash, external) == command_id
    queued = store.get_state(validation_state_key(trial_hash))
    assert queued is not None
    assert queued["status"] == "queued"
    assert queued["progress"] == 0

    def fake_finalize(
        _self: ApplicationService,
        selected_generation: str,
        selected_trial: str,
        selected_external: Path,
    ) -> dict[str, Any]:
        assert (selected_generation, selected_trial, selected_external) == (
            generation_id,
            trial_hash,
            external,
        )
        return {
            "candidate_hash": "c" * 64,
            "eligible": False,
            "gates": {"trade_count": False, "sharpe": True},
        }

    monkeypatch.setattr(ApplicationService, "finalize_strategy_choice", fake_finalize)
    assert ValidationWorker(settings, store).process_next() is True
    assert ValidationWorker(settings, store).process_next() is False

    completed = store.get_state(validation_state_key(trial_hash))
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["progress"] == 100
    assert completed["failed_gates"] == ["trade_count"]
    listed = store.list_states("validation:")
    assert listed[0]["candidate_hash"] == "c" * 64
    assert listed[0]["key"] == validation_state_key(trial_hash)


def test_validation_failure_is_visible_and_retryable(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_id, trial_hash, external = prepare_shortlist(tmp_path, settings, store, monkeypatch)
    service = ApplicationService(store, settings)
    first_command = service.queue_strategy_validation(generation_id, trial_hash, external)

    def fail_finalize(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise ValueError("external dataset is too short")

    monkeypatch.setattr(ApplicationService, "finalize_strategy_choice", fail_finalize)
    assert ValidationWorker(settings, store).process_next() is True
    failed = store.get_state(validation_state_key(trial_hash))
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["error"] == "external dataset is too short"

    second_command = service.queue_strategy_validation(generation_id, trial_hash, external)
    assert second_command != first_command
    assert store.get_state(validation_state_key(trial_hash))["status"] == "queued"  # type: ignore[index]


def test_validation_queue_rejects_invalid_preconditions(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Settings, "project_root", property(lambda _self: tmp_path))
    service = ApplicationService(store, settings)
    with pytest.raises(ValueError, match="unknown strategy generation"):
        service.queue_strategy_validation("missing", "t" * 64, tmp_path / "missing.csv")

    store.save_generation("generation-4-running", "g" * 64, "running", {}, {})
    with pytest.raises(ValueError, match="must finish"):
        service.queue_strategy_validation(
            "generation-4-running", "t" * 64, tmp_path / "missing.csv"
        )


def test_validation_dashboard_queues_and_exposes_progress(
    tmp_path: Path,
    settings: Settings,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_id, trial_hash, external = prepare_shortlist(tmp_path, settings, store, monkeypatch)
    client = TestClient(create_app(settings, store))
    page = client.get("/")
    token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
    assert token is not None

    response = client.post(
        f"/api/strategies/{generation_id}/{trial_hash}/finalize",
        data={
            "csrf_token": token.group(1),
            "external_dataset": str(external.relative_to(tmp_path)),
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/strategies"
    payload = client.get("/api/validations").json()
    assert payload["jobs"][0]["trial_hash"] == trial_hash
    assert payload["jobs"][0]["status"] == "queued"
