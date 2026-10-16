from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from gold_trader.cli import app
from gold_trader.config import get_settings
from gold_trader.database import Store
from gold_trader.generation import load_sealed_manifest, trial_specs


def test_cli_offline_workflow(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    database = tmp_path / "cli.sqlite3"
    monkeypatch.setenv("GOLD_TRADER_DATABASE_URL", f"sqlite:///{database}")
    monkeypatch.setenv("GOLD_TRADER_SESSION_SECRET", "cli-test-secret")
    get_settings.cache_clear()
    runner = CliRunner()

    assert runner.invoke(app, ["version"]).exit_code == 0
    assert runner.invoke(app, ["init"]).exit_code == 0
    doctor = runner.invoke(app, ["doctor"])
    assert doctor.exit_code == 0
    assert '"live_execution_absent": true' in doctor.stdout

    dataset = tmp_path / "demo.csv"
    demo = runner.invoke(app, ["demo-data", "--output", str(dataset), "--rows", "5000"])
    assert demo.exit_code == 0
    assert dataset.exists()

    strategies = runner.invoke(app, ["research", "list-strategies"])
    assert strategies.exit_code == 0
    assert "trend-pullback" in strategies.stdout

    research = runner.invoke(
        app,
        [
            "research",
            "run",
            "--dataset",
            str(dataset),
            "--strategy",
            "trend-pullback",
        ],
    )
    assert research.exit_code == 0
    assert '"status": "ineligible"' in research.stdout
    assert runner.invoke(app, ["research", "runs"]).exit_code == 0

    imported = tmp_path / "imported.parquet"
    data_import = runner.invoke(
        app,
        ["data", "import", str(dataset), "--destination", str(imported)],
    )
    assert data_import.exit_code == 0
    assert imported.exists()

    comparison = tmp_path / "comparison.json"
    summarized = runner.invoke(
        app,
        [
            "research",
            "summarize",
            "--dataset",
            str(dataset),
            "--output",
            str(comparison),
        ],
    )
    assert summarized.exit_code == 0
    assert comparison.exists()

    manifest = tmp_path / "generation.json"
    generation = runner.invoke(
        app,
        [
            "research",
            "generation",
            "create",
            "--dataset",
            str(imported),
            "--manifest",
            str(manifest),
            "--workers",
            "1",
            "--trials-per-family",
            "1",
            "--batch-size",
            "2",
        ],
    )
    assert generation.exit_code == 0
    assert (
        runner.invoke(
            app, ["research", "generation", "seal", "--manifest", str(manifest)]
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["research", "generation", "status"]).exit_code == 0
    assert runner.invoke(app, ["research", "generation", "pause"]).exit_code == 0
    assert runner.invoke(app, ["research", "generation", "stop"]).exit_code == 0
    assert (
        runner.invoke(
            app, ["research", "generation", "start", "--manifest", str(manifest)]
        ).exit_code
        == 0
    )

    sealed = load_sealed_manifest(manifest)
    database_store = Store(get_settings().database_url)
    shortlist_result = {
        "status": "shortlisted",
        "score": 1.0,
        "metrics": {
            "net_return": 0.1,
            "profit_factor": 1.2,
            "sharpe": 0.9,
            "max_drawdown": 0.05,
            "trade_count": 50,
        },
        "stressed_metrics": {"net_return": 0.05},
        "fold_profitable_fraction": 0.8,
        "largest_profit_fraction": 0.3,
    }
    specs = trial_specs(sealed)
    for family in ("compression-breakout", "time-series-momentum"):
        spec = next(item for item in specs if item.family == family)
        database_store.save_trial(
            sealed.generation_id, spec.model_dump(mode="json"), shortlist_result
        )
    database_store.save_generation(
        sealed.generation_id,
        sealed.manifest_hash,
        "completed",
        sealed.model_dump(mode="json"),
        {"completed_trials": 6, "shortlisted": 2, "manifest_path": str(manifest)},
    )
    generation4_manifest = tmp_path / "generation4.json"
    learning_report = tmp_path / "learning.json"
    generation4 = runner.invoke(
        app,
        [
            "research",
            "generation4",
            "create",
            "--parent-generation-id",
            sealed.generation_id,
            "--learning-report",
            str(learning_report),
            "--manifest",
            str(generation4_manifest),
            "--workers",
            "1",
            "--trials-per-family",
            "1",
            "--batch-size",
            "1",
        ],
    )
    assert generation4.exit_code == 0
    assert (
        runner.invoke(
            app,
            ["research", "generation4", "seal", "--manifest", str(generation4_manifest)],
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["research", "generation4", "status"]).exit_code == 0
    assert runner.invoke(app, ["research", "generation4", "pause"]).exit_code == 0
    assert runner.invoke(app, ["research", "generation4", "stop"]).exit_code == 0
    assert (
        runner.invoke(
            app,
            ["research", "generation4", "start", "--manifest", str(generation4_manifest)],
        ).exit_code
        == 0
    )

    assert runner.invoke(app, ["paper", "status"]).exit_code == 0
    assert runner.invoke(app, ["paper", "stop"]).exit_code == 0
    assert runner.invoke(app, ["paper", "kill", "--reason", "test"]).exit_code == 0
    assert runner.invoke(app, ["paper", "reset-kill"]).exit_code == 0

    readiness = tmp_path / "readiness.json"
    report = runner.invoke(app, ["report", "--output", str(readiness)])
    assert report.exit_code == 0
    assert readiness.exists()
    get_settings.cache_clear()


def test_cli_rejects_unknown_strategy(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("GOLD_TRADER_DATABASE_URL", f"sqlite:///{tmp_path / 'bad.sqlite3'}")
    get_settings.cache_clear()
    dataset = tmp_path / "demo.csv"
    runner = CliRunner()
    runner.invoke(app, ["demo-data", "--output", str(dataset), "--rows", "500"])
    result = runner.invoke(
        app, ["research", "run", "--dataset", str(dataset), "--strategy", "unknown"]
    )
    assert result.exit_code != 0
    assert "unknown strategy" in (result.stdout + result.stderr)
    baseline_meta = runner.invoke(
        app,
        [
            "research",
            "run-meta",
            "--dataset",
            str(dataset),
            "--strategy",
            "trend-pullback",
        ],
    )
    assert baseline_meta.exit_code != 0
    assert "preregistered only" in (baseline_meta.stdout + baseline_meta.stderr)
    get_settings.cache_clear()
