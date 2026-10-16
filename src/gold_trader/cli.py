from __future__ import annotations

import asyncio
import json
import multiprocessing
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from gold_trader import __version__
from gold_trader.config import Settings, get_settings
from gold_trader.data import (
    file_sha256,
    generate_demo_data,
    import_market_file,
)
from gold_trader.database import Store
from gold_trader.generation import (
    GenerationCoordinator,
    create_manifest,
    research_worker_main,
    seal_manifest,
)
from gold_trader.generation4 import (
    Generation4Coordinator,
    create_generation4_manifest,
    generation4_worker_main,
    seal_generation4_manifest,
)
from gold_trader.generation5 import (
    Generation5Coordinator,
    create_generation5_manifest,
    generation5_worker_main,
    seal_generation5_manifest,
)
from gold_trader.generation6 import (
    Generation6Coordinator,
    create_generation6_manifest,
    generation6_paper_worker_main,
    generation6_worker_main,
    seal_generation6_manifest,
)
from gold_trader.meta_research import MetaResearchRunner
from gold_trader.ml import optional_ml_backends
from gold_trader.paper_lab import paper_lab_worker_main
from gold_trader.research import ResearchRunner
from gold_trader.secrets import get_secret, set_secret
from gold_trader.service import ApplicationService
from gold_trader.strategies import GENERATION_TWO_VARIANTS, strategy_registry
from gold_trader.validation import validation_worker_main
from gold_trader.worker import PROVIDER_REGISTRY, worker_main

app = typer.Typer(help="Aurum XAUUSD research and paper-trading system", no_args_is_help=True)
data_app = typer.Typer(help="Import and inspect point-in-time market data")
research_app = typer.Typer(help="Run reproducible strategy research")
generation_app = typer.Typer(help="Run the sealed Generation 3 local coordinator")
generation4_app = typer.Typer(help="Run the adaptive sealed Generation 4 coordinator")
generation5_app = typer.Typer(help="Run the frequency-first sealed Generation 5 coordinator")
generation6_app = typer.Typer(help="Run the high-activity sealed Generation 6 coordinator")
paper_app = typer.Typer(help="Control the paper worker; real orders do not exist")
secret_app = typer.Typer(help="Store provider credentials in macOS Keychain")
app.add_typer(data_app, name="data")
app.add_typer(research_app, name="research")
research_app.add_typer(generation_app, name="generation")
research_app.add_typer(generation4_app, name="generation4")
research_app.add_typer(generation5_app, name="generation5")
research_app.add_typer(generation6_app, name="generation6")
app.add_typer(paper_app, name="paper")
app.add_typer(secret_app, name="secret")


def runtime() -> tuple[Settings, Store, ApplicationService]:
    settings = get_settings()
    store = Store(settings.database_url)
    store.initialize()
    return settings, store, ApplicationService(store, settings)


@app.command()
def version() -> None:
    """Show the installed foundation version."""
    typer.echo(__version__)


@app.command()
def init() -> None:
    """Create local runtime directories and initialize the audit database."""
    settings, store, _service = runtime()
    typer.echo(f"Initialized {settings.project_root}")
    typer.echo(f"Database: {store.engine.url.render_as_string(hide_password=True)}")


@app.command()
def doctor() -> None:
    """Check the local safety and runtime prerequisites."""
    settings, store, _service = runtime()
    checks = {
        "python_3_12": sys.version_info[:2] == (3, 12),
        "database": store.get_state("paper") is not None,
        "localhost_binding": settings.host in {"127.0.0.1", "::1", "localhost"},
        "caffeinate": shutil.which("caffeinate") is not None if sys.platform == "darwin" else True,
        "live_execution_absent": True,
    }
    typer.echo(json.dumps({"checks": checks, "ml_backends": optional_ml_backends()}, indent=2))
    if not all(checks.values()):
        raise typer.Exit(code=1)


@app.command("demo-data")
def demo_data(
    output: Annotated[Path, typer.Option()] = Path("var/data/demo_xauusd_5m.csv"),
    rows: Annotated[int, typer.Option(min=500, max=100_000)] = 5000,
) -> None:
    """Create deterministic synthetic bars for workflow testing only."""
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_demo_data(rows=rows)
    frame.to_csv(output, index=False)
    typer.echo(f"Wrote {len(frame)} synthetic bars to {output}")
    typer.echo("Synthetic data is never eligible for a profitability claim.")


@data_app.command("import")
def import_data(
    source: Annotated[Path, typer.Argument(exists=True, readable=True)],
    destination: Annotated[Path, typer.Option()] = Path("var/data/xauusd_5m.parquet"),
    source_name: Annotated[str, typer.Option()] = "user-import",
) -> None:
    """Normalize CSV/Parquet bars and write an immutable manifest."""
    settings, _store, _service = runtime()
    manifest = import_market_file(source, destination, source_name=source_name)
    manifest_path = settings.project_root / "var/data" / f"{manifest.dataset_id}.manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(manifest.model_dump_json(indent=2))


@data_app.command("quote")
def live_delayed_quote(
    provider: Annotated[
        str,
        typer.Option(help="Provider name, e.g. twelve-data, alpha-vantage, metals-dev, finnhub"),
    ] = "",
) -> None:
    """Fetch one provider-labeled XAU/USD quote; this never submits an order."""
    settings, _store, _service = runtime()
    name = provider or settings.live_paper.provider_priority[0]
    provider_class = PROVIDER_REGISTRY.get(name)
    if provider_class is None:
        raise typer.BadParameter(
            f"unknown provider {name!r}; choose one of {', '.join(PROVIDER_REGISTRY)}"
        )
    api_key = get_secret(name)
    if not api_key:
        raise typer.BadParameter(f"missing Keychain secret {name!r}; use secret set")
    try:
        quote = asyncio.run(provider_class(api_key).quote())
    except RuntimeError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(quote.model_dump_json(indent=2))


@research_app.command("list-strategies")
def list_strategies() -> None:
    for strategy_id, strategy in strategy_registry().items():
        typer.echo(f"{strategy_id:36} {strategy.timeframe:5} {strategy.strategy_hash[:12]}")


@research_app.command("run")
def run_research(
    dataset: Annotated[Path, typer.Option(exists=True, readable=True)],
    strategy: Annotated[str, typer.Option()] = "trend-pullback",
) -> None:
    settings, store, _service = runtime()
    strategies = strategy_registry()
    if strategy not in strategies:
        raise typer.BadParameter(f"unknown strategy; choose from {', '.join(strategies)}")
    report = ResearchRunner(settings, store).run(dataset, strategies[strategy])
    typer.echo(
        json.dumps(
            {
                "candidate_hash": report["candidate_hash"],
                "status": report["status"],
                "metrics": report["metrics"],
                "failed_gates": [key for key, passed in report["gates"].items() if not passed],
            },
            indent=2,
        )
    )


@research_app.command("run-meta")
def run_meta_research(
    dataset: Annotated[Path, typer.Option(exists=True, readable=True)],
    strategy: Annotated[str, typer.Option()] = "trend-pullback-ema30-80",
) -> None:
    """Run the fixed Generation-2 elastic-net signal filter protocol."""
    settings, store, _service = runtime()
    strategies = strategy_registry()
    if strategy not in GENERATION_TWO_VARIANTS:
        raise typer.BadParameter(
            "meta research is preregistered only for: " + ", ".join(GENERATION_TWO_VARIANTS)
        )
    try:
        report = MetaResearchRunner(settings, store).run(dataset, strategies[strategy])
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "candidate_hash": report["candidate_hash"],
                "parent_candidate_hash": report["parent_candidate_hash"],
                "status": report["status"],
                "metrics": report["metrics"],
                "probability_evidence": report["ml_evidence"]["holdout_probability"],
                "failed_gates": [key for key, passed in report["gates"].items() if not passed],
                "model_card": report["model_card_path"],
            },
            indent=2,
        )
    )


@research_app.command("runs")
def research_runs() -> None:
    _settings, store, _service = runtime()
    typer.echo(json.dumps(store.list_research_runs(), indent=2, default=str))


@research_app.command("summarize")
def summarize_research(
    dataset: Annotated[Path, typer.Option(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()] = Path("var/reports/research-comparison.json"),
) -> None:
    """Write a compact comparison for every lineage tied to one dataset."""
    _settings, store, _service = runtime()
    dataset_hash = file_sha256(dataset)
    runs = [
        run for run in store.list_research_runs(limit=500) if run["dataset_hash"] == dataset_hash
    ]
    runs.sort(key=lambda run: (run.get("variant_id", run["strategy_id"]), run["model_hash"]))
    comparison = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset.resolve()),
        "dataset_hash": dataset_hash,
        "candidate_count": len(runs),
        "eligible_count": sum(bool(run.get("eligible")) for run in runs),
        "approval_allowed": any(bool(run.get("eligible")) for run in runs),
        "candidates": [
            {
                "variant_id": run.get("variant_id", run["strategy_id"]),
                "candidate_hash": run["candidate_hash"],
                "parent_candidate_hash": run.get("parent_candidate_hash"),
                "model": run.get("model", {}).get("model", "rule-only"),
                "status": run["status"],
                "metrics": run["metrics"],
                "failed_gates": [gate for gate, passed in run["gates"].items() if not passed],
                "holdout_probability": run.get("ml_evidence", {}).get("holdout_probability"),
            }
            for run in runs
        ],
        "disclaimer": (
            "Simulation only. No candidate is approved automatically and profitability is "
            "neither assumed nor promised."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    typer.echo(output.resolve())


@generation_app.command("create")
def generation_create(
    dataset: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/data/xauusd_5m.parquet"
    ),
    manifest: Annotated[Path, typer.Option()] = Path("var/generations/generation-3.json"),
    workers: Annotated[int | None, typer.Option(min=1, max=12)] = None,
    trials_per_family: Annotated[int | None, typer.Option(min=1)] = None,
    batch_size: Annotated[int | None, typer.Option(min=1)] = None,
    generation_id: Annotated[str, typer.Option()] = "generation-3",
) -> None:
    """Create a draft manifest; the untouched final 12 months are excluded from search."""
    settings, _store, _service = runtime()
    created = create_manifest(
        settings,
        dataset,
        manifest,
        generation_id=generation_id,
        workers=workers,
        trials_per_family=trials_per_family,
        batch_size=batch_size,
    )
    typer.echo(created.model_dump_json(indent=2))


@generation_app.command("seal")
def generation_seal(
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-3.json"
    ),
) -> None:
    """Seal data, search space, settings, and implementation hashes against mutation."""
    _settings, store, _service = runtime()
    sealed = seal_manifest(manifest, store)
    typer.echo(f"Sealed {sealed.generation_id}: {sealed.manifest_hash}")


@generation_app.command("start")
def generation_start(
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-3.json"
    ),
) -> None:
    """Queue the generation for the isolated research worker used by `serve`."""
    _settings, _store, service = runtime()
    command_id = service.start_generation(manifest)
    typer.echo(f"Queued Generation 3 command {command_id}")


@generation_app.command("run-now")
def generation_run_now(
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-3.json"
    ),
) -> None:
    """Run in this terminal with checkpoints; Ctrl-C leaves completed trials durable."""
    settings, store, _service = runtime()
    result = GenerationCoordinator(settings, store).run(manifest)
    typer.echo(json.dumps(result, indent=2, default=str))


@generation_app.command("pause")
def generation_pause() -> None:
    settings, store, _service = runtime()
    GenerationCoordinator(settings, store).request_control("pause")
    typer.echo("Generation 3 will pause after the active batch")


@generation_app.command("stop")
def generation_stop() -> None:
    settings, store, _service = runtime()
    GenerationCoordinator(settings, store).request_control("stop")
    typer.echo("Generation 3 will stop after the active batch; checkpoints remain resumable")


@generation_app.command("status")
def generation_status() -> None:
    _settings, store, _service = runtime()
    typer.echo(json.dumps(store.get_generation(), indent=2, default=str))


@generation_app.command("finalize")
def generation_finalize(
    trial_hash: Annotated[str, typer.Option()],
    external_dataset: Annotated[Path, typer.Option(exists=True, readable=True)],
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-3.json"
    ),
) -> None:
    """Open the primary holdout once; external data is required for eligibility."""
    settings, store, _service = runtime()
    report = GenerationCoordinator(settings, store).finalize(manifest, trial_hash, external_dataset)
    typer.echo(
        json.dumps(
            {
                "candidate_hash": report["candidate_hash"],
                "status": report["status"],
                "failed_gates": [key for key, value in report["gates"].items() if not value],
            },
            indent=2,
        )
    )


@generation4_app.command("create")
def generation4_create(
    parent_generation_id: Annotated[str, typer.Option()] = "generation-3-r4",
    learning_report: Annotated[Path, typer.Option()] = Path(
        "var/reports/generation-3-learning.json"
    ),
    manifest: Annotated[Path, typer.Option()] = Path("var/generations/generation-4.json"),
    workers: Annotated[int | None, typer.Option(min=1, max=12)] = None,
    trials_per_family: Annotated[int, typer.Option(min=1)] = 512,
    batch_size: Annotated[int | None, typer.Option(min=1)] = None,
    generation_id: Annotated[str, typer.Option()] = "generation-4",
) -> None:
    """Learn a bounded Generation 4 search space from completed Generation 3 discovery."""
    settings, store, _service = runtime()
    created = create_generation4_manifest(
        settings,
        store,
        parent_generation_id,
        learning_report,
        manifest,
        generation_id=generation_id,
        workers=workers,
        trials_per_family=trials_per_family,
        batch_size=batch_size,
    )
    typer.echo(created.model_dump_json(indent=2))


@generation4_app.command("seal")
def generation4_seal(
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-4.json"
    ),
) -> None:
    """Seal the learned space, parent evidence, data, settings, and implementation."""
    _settings, store, _service = runtime()
    sealed = seal_generation4_manifest(manifest, store)
    typer.echo(f"Sealed {sealed.generation_id}: {sealed.manifest_hash}")


@generation4_app.command("start")
def generation4_start(
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-4.json"
    ),
) -> None:
    """Queue Generation 4 for the isolated worker started by `serve`."""
    _settings, _store, service = runtime()
    command_id = service.start_generation4(manifest)
    typer.echo(f"Queued Generation 4 command {command_id}")


@generation4_app.command("run-now")
def generation4_run_now(
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-4.json"
    ),
) -> None:
    """Run Generation 4 in this terminal with durable batch checkpoints."""
    settings, store, _service = runtime()
    result = Generation4Coordinator(settings, store).run(manifest)
    typer.echo(json.dumps(result, indent=2, default=str))


@generation4_app.command("pause")
def generation4_pause() -> None:
    settings, store, _service = runtime()
    Generation4Coordinator(settings, store).request_control("pause")
    typer.echo("Generation 4 will pause after the active batch")


@generation4_app.command("stop")
def generation4_stop() -> None:
    settings, store, _service = runtime()
    Generation4Coordinator(settings, store).request_control("stop")
    typer.echo("Generation 4 will stop after the active batch; checkpoints remain resumable")


@generation4_app.command("status")
def generation4_status(
    generation_id: Annotated[str, typer.Option()] = "generation-4",
) -> None:
    _settings, store, _service = runtime()
    typer.echo(json.dumps(store.get_generation(generation_id), indent=2, default=str))


@generation4_app.command("finalize")
def generation4_finalize(
    trial_hash: Annotated[str, typer.Option()],
    external_dataset: Annotated[Path, typer.Option(exists=True, readable=True)],
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-4.json"
    ),
) -> None:
    """Run the one-time final holdout and independent-source validation."""
    settings, store, _service = runtime()
    report = Generation4Coordinator(settings, store).finalize(
        manifest, trial_hash, external_dataset
    )
    typer.echo(
        json.dumps(
            {
                "candidate_hash": report["candidate_hash"],
                "status": report["status"],
                "failed_gates": [key for key, value in report["gates"].items() if not value],
            },
            indent=2,
        )
    )


@generation5_app.command("create")
def generation5_create(
    parent_generation_id: Annotated[str, typer.Option()] = "generation-4",
    learning_report: Annotated[Path, typer.Option()] = Path(
        "var/reports/generation-5-learning.json"
    ),
    manifest: Annotated[Path, typer.Option()] = Path("var/generations/generation-5.json"),
    workers: Annotated[int | None, typer.Option(min=1, max=12)] = None,
    trials_per_family: Annotated[int, typer.Option(min=1)] = 384,
    batch_size: Annotated[int | None, typer.Option(min=1)] = None,
    generation_id: Annotated[str, typer.Option()] = "generation-5",
) -> None:
    """Create the frequency-first lineage from completed Generation 4 evidence."""
    settings, store, _service = runtime()
    created = create_generation5_manifest(
        settings,
        store,
        parent_generation_id,
        learning_report,
        manifest,
        generation_id=generation_id,
        workers=workers,
        trials_per_family=trials_per_family,
        batch_size=batch_size,
    )
    typer.echo(created.model_dump_json(indent=2))


@generation5_app.command("seal")
def generation5_seal(
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-5.json"
    ),
) -> None:
    """Seal Generation 5 data, policy, search space, settings, and implementation."""
    _settings, store, _service = runtime()
    sealed = seal_generation5_manifest(manifest, store)
    typer.echo(f"Sealed {sealed.generation_id}: {sealed.manifest_hash}")


@generation5_app.command("start")
def generation5_start(
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-5.json"
    ),
) -> None:
    """Queue Generation 5 for the isolated worker started by `serve`."""
    _settings, _store, service = runtime()
    command_id = service.start_generation5(manifest)
    typer.echo(f"Queued Generation 5 command {command_id}")


@generation5_app.command("run-now")
def generation5_run_now(
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-5.json"
    ),
) -> None:
    """Run Generation 5 in this terminal with durable batch checkpoints."""
    settings, store, _service = runtime()
    result = Generation5Coordinator(settings, store).run(manifest)
    typer.echo(json.dumps(result, indent=2, default=str))


@generation5_app.command("pause")
def generation5_pause() -> None:
    settings, store, _service = runtime()
    Generation5Coordinator(settings, store).request_control("pause")
    typer.echo("Generation 5 will pause after the active batch")


@generation5_app.command("stop")
def generation5_stop() -> None:
    settings, store, _service = runtime()
    Generation5Coordinator(settings, store).request_control("stop")
    typer.echo("Generation 5 will stop after the active batch; checkpoints remain resumable")


@generation5_app.command("status")
def generation5_status(
    generation_id: Annotated[str, typer.Option()] = "generation-5",
) -> None:
    _settings, store, _service = runtime()
    typer.echo(json.dumps(store.get_generation(generation_id), indent=2, default=str))


@generation5_app.command("finalize")
def generation5_finalize(
    trial_hash: Annotated[str, typer.Option()],
    external_dataset: Annotated[Path, typer.Option(exists=True, readable=True)],
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-5.json"
    ),
) -> None:
    """Run one-time session-normalized final validation for a Generation 5 shortlist."""
    settings, store, _service = runtime()
    report = Generation5Coordinator(settings, store).finalize(
        manifest, trial_hash, external_dataset
    )
    typer.echo(
        json.dumps(
            {
                "candidate_hash": report["candidate_hash"],
                "status": report["status"],
                "failed_gates": [key for key, value in report["gates"].items() if not value],
            },
            indent=2,
        )
    )


@generation6_app.command("create")
def generation6_create(
    parent_generation_id: Annotated[str, typer.Option()] = "generation-5",
    learning_report: Annotated[Path, typer.Option()] = Path(
        "var/reports/generation-6-learning.json"
    ),
    manifest: Annotated[Path, typer.Option()] = Path("var/generations/generation-6.json"),
    workers: Annotated[int | None, typer.Option(min=1, max=12)] = None,
    trials_per_family: Annotated[int, typer.Option(min=1)] = 384,
    batch_size: Annotated[int | None, typer.Option(min=1)] = None,
    generation_id: Annotated[str, typer.Option()] = "generation-6",
) -> None:
    """Create Generation 6 from terminal Generation 5 evidence."""
    settings, store, _service = runtime()
    created = create_generation6_manifest(
        settings,
        store,
        parent_generation_id,
        learning_report,
        manifest,
        generation_id=generation_id,
        workers=workers,
        trials_per_family=trials_per_family,
        batch_size=batch_size,
    )
    typer.echo(created.model_dump_json(indent=2))


@generation6_app.command("seal")
def generation6_seal(
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-6.json"
    ),
) -> None:
    """Seal Generation 6 data, hypotheses, policy, settings, and implementation."""
    _settings, store, _service = runtime()
    sealed = seal_generation6_manifest(manifest, store)
    typer.echo(f"Sealed {sealed.generation_id}: {sealed.manifest_hash}")


@generation6_app.command("start")
def generation6_start(
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-6.json"
    ),
) -> None:
    """Queue Generation 6 for the isolated worker started by `serve`."""
    _settings, _store, service = runtime()
    command_id = service.start_generation6(manifest)
    typer.echo(f"Queued Generation 6 command {command_id}")


@generation6_app.command("run-now")
def generation6_run_now(
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-6.json"
    ),
) -> None:
    """Run Generation 6 in this terminal with durable batch checkpoints."""
    settings, store, _service = runtime()
    result = Generation6Coordinator(settings, store).run(manifest)
    typer.echo(json.dumps(result, indent=2, default=str))


@generation6_app.command("pause")
def generation6_pause() -> None:
    settings, store, _service = runtime()
    Generation6Coordinator(settings, store).request_control("pause")
    typer.echo("Generation 6 will pause after the active batch")


@generation6_app.command("stop")
def generation6_stop() -> None:
    settings, store, _service = runtime()
    Generation6Coordinator(settings, store).request_control("stop")
    typer.echo("Generation 6 will stop after the active batch; checkpoints remain resumable")


@generation6_app.command("status")
def generation6_status(
    generation_id: Annotated[str, typer.Option()] = "generation-6",
) -> None:
    _settings, store, _service = runtime()
    typer.echo(json.dumps(store.get_generation(generation_id), indent=2, default=str))


@generation6_app.command("finalize")
def generation6_finalize(
    trial_hash: Annotated[str, typer.Option()],
    external_dataset: Annotated[Path, typer.Option(exists=True, readable=True)],
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "var/generations/generation-6.json"
    ),
) -> None:
    """Run one-time session-normalized final validation for a Generation 6 shortlist."""
    settings, store, _service = runtime()
    report = Generation6Coordinator(settings, store).finalize(
        manifest, trial_hash, external_dataset
    )
    typer.echo(
        json.dumps(
            {
                "candidate_hash": report["candidate_hash"],
                "status": report["status"],
                "failed_gates": [key for key, value in report["gates"].items() if not value],
            },
            indent=2,
        )
    )


@paper_app.command("start")
def paper_start(mode: Annotated[str, typer.Option()] = "replay") -> None:
    _settings, _store, service = runtime()
    command_id = service.start_paper(mode)
    typer.echo(f"Queued safe paper start command {command_id}")


@paper_app.command("stop")
def paper_stop() -> None:
    _settings, _store, service = runtime()
    command_id = service.stop_paper()
    typer.echo(f"Queued paper stop command {command_id}")


@paper_app.command("kill")
def paper_kill(reason: Annotated[str, typer.Option()] = "operator CLI kill switch") -> None:
    _settings, _store, service = runtime()
    command_id = service.activate_kill_switch(reason)
    typer.echo(f"Kill switch active; cancellation command {command_id}")


@paper_app.command("reset-kill")
def reset_kill() -> None:
    _settings, _store, service = runtime()
    service.reset_kill_switch()
    typer.echo("Kill switch reset after operator review")


@paper_app.command("status")
def paper_status() -> None:
    _settings, _store, service = runtime()
    typer.echo(json.dumps(service.status(), indent=2, default=str))


@secret_app.command("set")
def secret_set(name: str) -> None:
    typer.echo("Paste the value once; hidden input displays no characters. Then press Enter.")
    value = typer.prompt("Secret value", hide_input=True)
    for repetitions in range(2, 5):
        part = value[: len(value) // repetitions]
        if len(value) % repetitions == 0 and value == part * repetitions:
            raise typer.BadParameter(
                f"secret appears to have been pasted {repetitions} times; rerun and paste once"
            )
    set_secret(name, value)
    typer.echo(f"Stored {name!r} in the operating-system keychain")


@secret_app.command("check")
def secret_check(name: str) -> None:
    typer.echo("present" if get_secret(name) else "missing")


@app.command()
def report(
    output: Annotated[Path, typer.Option()] = Path("var/reports/readiness.json"),
) -> None:
    _settings, store, service = runtime()
    runs = store.list_research_runs()
    approval = store.valid_approval()
    report_data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "paper": service.status()["paper"],
        "approval": approval,
        "eligible_research_runs": [run for run in runs if run.get("eligible")],
        "future_live_ready": False,
        "blocking_reason": "30 trading days and 50 closed paper trades are not yet evidenced",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report_data, indent=2, default=str), encoding="utf-8")
    typer.echo(output)


@app.command(hidden=True)
def worker() -> None:
    worker_main()


@app.command(hidden=True)
def research_worker() -> None:
    research_worker_main()


@app.command(hidden=True)
def generation5_worker() -> None:
    generation5_worker_main()


@app.command(hidden=True)
def generation6_worker() -> None:
    generation6_worker_main()


@app.command(hidden=True)
def paper_lab_worker() -> None:
    paper_lab_worker_main()


@app.command()
def serve() -> None:
    """Start the localhost dashboard and isolated command worker."""
    settings, _store, _service = runtime()
    if settings.host not in {"127.0.0.1", "::1", "localhost"}:
        raise typer.BadParameter("dashboard must bind to a loopback address")
    paper_process = multiprocessing.Process(target=worker_main, name="gold-trader-paper-worker")
    research_process = multiprocessing.Process(
        target=research_worker_main, name="gold-trader-research-worker"
    )
    generation4_process = multiprocessing.Process(
        target=generation4_worker_main, name="gold-trader-generation4-worker"
    )
    generation5_process = multiprocessing.Process(
        target=generation5_worker_main, name="gold-trader-generation5-worker"
    )
    generation6_process = multiprocessing.Process(
        target=generation6_worker_main, name="gold-trader-generation6-worker"
    )
    generation6_paper_process = multiprocessing.Process(
        target=generation6_paper_worker_main,
        name="gold-trader-generation6-paper-worker",
    )
    validation_process = multiprocessing.Process(
        target=validation_worker_main, name="gold-trader-validation-worker"
    )
    paper_lab_process = multiprocessing.Process(
        target=paper_lab_worker_main, name="gold-trader-paper-lab-worker"
    )
    paper_process.start()
    research_process.start()
    generation4_process.start()
    generation5_process.start()
    generation6_process.start()
    generation6_paper_process.start()
    validation_process.start()
    paper_lab_process.start()
    try:
        from gold_trader.web import create_app

        uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    finally:
        paper_process.terminate()
        research_process.terminate()
        generation4_process.terminate()
        generation5_process.terminate()
        generation6_process.terminate()
        generation6_paper_process.terminate()
        validation_process.terminate()
        paper_lab_process.terminate()
        paper_process.join(timeout=5)
        research_process.join(timeout=5)
        generation4_process.join(timeout=5)
        generation5_process.join(timeout=5)
        generation6_process.join(timeout=5)
        generation6_paper_process.join(timeout=5)
        validation_process.join(timeout=5)
        paper_lab_process.join(timeout=5)


if __name__ == "__main__":
    app()
