from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response

from gold_trader.config import Settings, get_settings
from gold_trader.database import Store
from gold_trader.generation import explain_trial_result
from gold_trader.generation4 import explain_generation4_result
from gold_trader.generation5 import explain_generation5_result
from gold_trader.generation6 import explain_generation6_result
from gold_trader.paper_lab import validate_experimental_report
from gold_trader.service import ApplicationService


class LocalOnlyMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        client = request.client.host if request.client else ""
        if client not in {"127.0.0.1", "::1", "testclient"}:
            return JSONResponse({"detail": "localhost access only"}, status_code=403)
        return await call_next(request)


def create_app(settings: Settings | None = None, store: Store | None = None) -> FastAPI:
    selected = settings or get_settings()
    selected_store = store or Store(selected.database_url)
    selected_store.initialize()
    service = ApplicationService(selected_store, selected)
    root = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=root / "templates")

    app = FastAPI(
        title="Aurum XAUUSD Paper Trader",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    app.add_middleware(
        SessionMiddleware, secret_key=selected.effective_session_secret, same_site="strict"
    )
    app.add_middleware(LocalOnlyMiddleware)
    app.mount("/static", StaticFiles(directory=root / "static"), name="static")

    def presented_trials(
        generation_id: str | None = None, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        items = selected_store.list_trials(
            generation_id=generation_id, limit=limit, sort_by_return=True
        )
        for item in items:
            item_generation = str(item["spec"].get("generation_id", ""))
            if item_generation.startswith("generation-6"):
                item["result"]["evidence"] = explain_generation6_result(item["result"])
            elif item_generation.startswith("generation-5"):
                item["result"]["evidence"] = explain_generation5_result(item["result"])
            elif item_generation.startswith("generation-4"):
                item["result"]["evidence"] = explain_generation4_result(item["result"])
            else:
                item["result"]["evidence"] = explain_trial_result(item["spec"], item["result"])
        return items

    def strategy_choices() -> list[dict[str, Any]]:
        choices: list[dict[str, Any]] = []
        for generation in selected_store.list_generations(limit=50):
            generation_id = str(generation["generation_id"])
            if (
                generation_id != "generation-3-r4"
                and not generation_id.startswith("generation-4")
                and not generation_id.startswith("generation-5")
                and not generation_id.startswith("generation-6")
            ):
                continue
            for item in presented_trials(generation_id, limit=100_000):
                if item["result"].get("status") != "shortlisted":
                    continue
                item["generation_id"] = generation_id
                item["generation_status"] = generation["status"]
                choices.append(item)
        choices.sort(
            key=lambda item: float(item["result"].get("metrics", {}).get("net_return", 0)),
            reverse=True,
        )
        return choices

    def strategy_finalizations() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        finalized: list[dict[str, Any]] = []
        by_trial: dict[str, dict[str, Any]] = {}
        for run in selected_store.list_research_runs(limit=500):
            generation_evidence = (
                run.get("generation_six")
                or run.get("generation_five")
                or run.get("generation_four")
                or run.get("generation_three")
            )
            if generation_evidence:
                trial_hash = str(generation_evidence.get("trial_spec", {}).get("trial_hash", ""))
                if trial_hash:
                    by_trial[trial_hash] = run
                finalized.append(run)
            elif run.get("eligible"):
                finalized.append(run)
        return finalized, by_trial

    def paper_lab_candidates() -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for run in selected_store.list_research_runs(limit=500):
            try:
                validate_experimental_report(run)
            except ValueError:
                continue
            evidence = (
                run.get("generation_six")
                or run.get("generation_five")
                or run.get("generation_four")
                or run.get("generation_three")
                or {}
            )
            trial = evidence.get("trial_spec", {})
            discovery = evidence.get("discovery_result", {})
            candidates.append(
                {
                    "candidate_hash": run["candidate_hash"],
                    "strategy_id": run["strategy_id"],
                    "variant_id": run.get("variant_id", run["strategy_id"]),
                    "timeframe": run["timeframe"],
                    "eligible": bool(run.get("eligible")),
                    "failed_gates": [
                        str(name) for name, passed in run.get("gates", {}).items() if not passed
                    ],
                    "final_metrics": run.get("metrics", {}),
                    "external_metrics": run.get("external_metrics", {}),
                    "discovery_metrics": discovery.get("metrics", {}),
                    "generation_id": trial.get("generation_id", "generation"),
                    "trial_hash": trial.get("trial_hash", ""),
                    "parameters": trial.get("parameters", {}),
                }
            )
        candidates.sort(
            key=lambda item: float(item["discovery_metrics"].get("net_return", 0)),
            reverse=True,
        )
        return candidates

    def validation_jobs() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        jobs = selected_store.list_states("validation:")
        return jobs, {str(job.get("trial_hash", "")): job for job in jobs}

    def external_datasets() -> list[str]:
        data_root = (selected.project_root / "var/data").resolve()
        primary_paths = {
            Path(str(generation.get("manifest", {}).get("dataset_path", ""))).resolve()
            for generation in (
                selected_store.get_generation(item["generation_id"])
                for item in selected_store.list_generations(limit=50)
            )
            if generation is not None
        }
        return [
            str(path.relative_to(selected.project_root))
            for path in sorted(data_root.glob("*"))
            if path.is_file()
            and path.suffix.lower() in {".csv", ".parquet"}
            and path.resolve() not in primary_paths
            and not path.name.startswith("demo_")
        ]

    def page_context(request: Request, **extra: Any) -> dict[str, Any]:
        token = request.session.get("csrf_token")
        if token is None:
            token = secrets.token_urlsafe(24)
            request.session["csrf_token"] = token
        return {
            "request": request,
            "csrf_token": token,
            "status": service.status(),
            **extra,
        }

    def check_csrf(request: Request, token: str) -> None:
        expected = request.session.get("csrf_token")
        if expected is None or not secrets.compare_digest(expected, token):
            raise HTTPException(status_code=403, detail="invalid CSRF token")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> Response:
        return templates.TemplateResponse(request, "dashboard.html", page_context(request))

    @app.get("/research", response_class=HTMLResponse)
    async def research(request: Request) -> Response:
        return templates.TemplateResponse(
            request,
            "research.html",
            page_context(request, runs=selected_store.list_research_runs()),
        )

    @app.get("/strategies", response_class=HTMLResponse)
    async def strategies(request: Request) -> Response:
        finalized_runs, finalizations_by_trial = strategy_finalizations()
        jobs, jobs_by_trial = validation_jobs()
        return templates.TemplateResponse(
            request,
            "strategies.html",
            page_context(
                request,
                choices=strategy_choices(),
                finalized_runs=finalized_runs,
                finalizations_by_trial=finalizations_by_trial,
                validation_jobs=jobs,
                validation_jobs_by_trial=jobs_by_trial,
                external_datasets=external_datasets(),
            ),
        )

    @app.get("/generation", response_class=HTMLResponse)
    async def generation(request: Request, generation_id: str | None = None) -> Response:
        selected_generation = selected_store.get_generation(generation_id)
        if generation_id is not None and selected_generation is None:
            raise HTTPException(status_code=404, detail="unknown generation")
        return templates.TemplateResponse(
            request,
            "generation.html",
            page_context(
                request,
                generation=selected_generation,
                generations=selected_store.list_generations(),
                trials=presented_trials(generation_id),
            ),
        )

    @app.get("/paper-lab", response_class=HTMLResponse)
    async def paper_lab(request: Request) -> Response:
        return templates.TemplateResponse(
            request,
            "paper_lab.html",
            page_context(
                request,
                lab=service.paper_lab_status(),
                candidates=paper_lab_candidates(),
            ),
        )

    @app.get("/audit", response_class=HTMLResponse)
    async def audit(request: Request) -> Response:
        return templates.TemplateResponse(
            request, "audit.html", page_context(request, events=selected_store.audit())
        )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request) -> Response:
        return templates.TemplateResponse(
            request,
            "settings.html",
            page_context(request, configuration=selected.model_dump()),
        )

    @app.get("/api/status")
    async def api_status() -> dict[str, Any]:
        return service.status()

    @app.get("/api/paper-lab")
    async def api_paper_lab() -> dict[str, Any]:
        return service.paper_lab_status()

    @app.get("/api/research")
    async def api_research() -> list[dict[str, Any]]:
        return selected_store.list_research_runs()

    @app.get("/api/reports/{candidate_hash}")
    async def api_report(candidate_hash: str) -> dict[str, Any]:
        report = selected_store.get_research_run(candidate_hash)
        if report is None:
            raise HTTPException(status_code=404, detail="unknown candidate report")
        return report

    @app.get("/api/strategies")
    async def api_strategies() -> dict[str, Any]:
        return {
            "choices": strategy_choices(),
            "approval": selected_store.valid_approval(),
            "external_datasets": external_datasets(),
        }

    @app.get("/api/validations")
    async def api_validations() -> dict[str, Any]:
        jobs, _by_trial = validation_jobs()
        return {"jobs": jobs}

    @app.get("/api/generation")
    async def api_generation(generation_id: str | None = None) -> dict[str, Any]:
        selected_generation = selected_store.get_generation(generation_id)
        if generation_id is not None and selected_generation is None:
            raise HTTPException(status_code=404, detail="unknown generation")
        trials = presented_trials(generation_id)
        return {
            "generation": selected_generation,
            "generations": selected_store.list_generations(),
            "trials": [
                {
                    "spec": item["spec"],
                    "result": {
                        key: item["result"].get(key)
                        for key in (
                            "status",
                            "stage",
                            "score",
                            "metrics",
                            "stressed_metrics",
                            "gates",
                            "evidence",
                            "selection_statistics",
                            "completed_at",
                        )
                    },
                }
                for item in trials
            ],
        }

    @app.get("/api/audit")
    async def api_audit() -> list[dict[str, Any]]:
        return selected_store.audit()

    @app.post("/api/candidates/{candidate_hash}/approve")
    async def approve(
        request: Request,
        candidate_hash: str,
        csrf_token: str = Form(...),
    ) -> Response:
        check_csrf(request, csrf_token)
        service.approve_candidate(candidate_hash)
        return RedirectResponse("/strategies", status_code=303)

    @app.post("/api/strategies/{generation_id}/{trial_hash}/finalize")
    async def finalize_strategy(
        request: Request,
        generation_id: str,
        trial_hash: str,
        external_dataset: str = Form(...),
        csrf_token: str = Form(...),
    ) -> Response:
        check_csrf(request, csrf_token)
        data_root = (selected.project_root / "var/data").resolve()
        dataset = (selected.project_root / external_dataset).resolve()
        if data_root != dataset.parent or dataset.suffix.lower() not in {".csv", ".parquet"}:
            raise ValueError("external validation dataset must be a CSV or Parquet in var/data")
        service.queue_strategy_validation(generation_id, trial_hash, dataset)
        return RedirectResponse("/strategies", status_code=303)

    @app.post("/api/paper/start")
    async def start_paper(
        request: Request,
        mode: str = Form("replay"),
        csrf_token: str = Form(...),
    ) -> Response:
        check_csrf(request, csrf_token)
        service.start_paper(mode)
        return RedirectResponse("/", status_code=303)

    @app.post("/api/paper/stop")
    async def stop_paper(request: Request, csrf_token: str = Form(...)) -> Response:
        check_csrf(request, csrf_token)
        service.stop_paper()
        return RedirectResponse("/", status_code=303)

    @app.post("/api/paper-lab/start")
    async def start_paper_lab(
        request: Request,
        candidate_hashes: list[str] = Form(...),
        acknowledge: str = Form(...),
        csrf_token: str = Form(...),
    ) -> Response:
        check_csrf(request, csrf_token)
        service.start_paper_lab(candidate_hashes, acknowledged=acknowledge == "paper-only")
        return RedirectResponse("/paper-lab", status_code=303)

    @app.post("/api/paper-lab/stop")
    async def stop_paper_lab(request: Request, csrf_token: str = Form(...)) -> Response:
        check_csrf(request, csrf_token)
        service.stop_paper_lab()
        return RedirectResponse("/paper-lab", status_code=303)

    @app.post("/api/paper-lab/kill")
    async def kill_paper_lab(
        request: Request,
        reason: str = Form("operator activated four-strategy lab kill"),
        csrf_token: str = Form(...),
    ) -> Response:
        check_csrf(request, csrf_token)
        service.kill_paper_lab(reason)
        return RedirectResponse("/paper-lab", status_code=303)

    @app.post("/api/generation/start")
    async def start_generation(
        request: Request,
        manifest_path: str = Form("var/generations/generation-3.json"),
        csrf_token: str = Form(...),
    ) -> Response:
        check_csrf(request, csrf_token)
        path = Path(manifest_path)
        if not path.is_absolute():
            path = selected.project_root / path
        resolved = path.resolve()
        allowed = (selected.project_root / "var/generations").resolve()
        if allowed not in resolved.parents:
            raise ValueError("generation manifest must be inside var/generations")
        service.start_generation(resolved)
        return RedirectResponse("/generation", status_code=303)

    @app.post("/api/generation/{action}")
    async def control_generation(
        request: Request, action: str, csrf_token: str = Form(...)
    ) -> Response:
        check_csrf(request, csrf_token)
        service.control_generation(action)
        return RedirectResponse("/generation", status_code=303)

    @app.post("/api/generation4/start")
    async def start_generation4(
        request: Request,
        manifest_path: str = Form("var/generations/generation-4.json"),
        csrf_token: str = Form(...),
    ) -> Response:
        check_csrf(request, csrf_token)
        path = Path(manifest_path)
        if not path.is_absolute():
            path = selected.project_root / path
        resolved = path.resolve()
        allowed = (selected.project_root / "var/generations").resolve()
        if allowed not in resolved.parents:
            raise ValueError("generation manifest must be inside var/generations")
        service.start_generation4(resolved)
        return RedirectResponse("/generation", status_code=303)

    @app.post("/api/generation4/{action}")
    async def control_generation4(
        request: Request, action: str, csrf_token: str = Form(...)
    ) -> Response:
        check_csrf(request, csrf_token)
        service.control_generation4(action)
        return RedirectResponse("/generation", status_code=303)

    @app.post("/api/generation5/start")
    async def start_generation5(
        request: Request,
        manifest_path: str = Form("var/generations/generation-5.json"),
        csrf_token: str = Form(...),
    ) -> Response:
        check_csrf(request, csrf_token)
        path = Path(manifest_path)
        if not path.is_absolute():
            path = selected.project_root / path
        resolved = path.resolve()
        allowed = (selected.project_root / "var/generations").resolve()
        if allowed not in resolved.parents:
            raise ValueError("generation manifest must be inside var/generations")
        service.start_generation5(resolved)
        return RedirectResponse("/generation", status_code=303)

    @app.post("/api/generation5/{action}")
    async def control_generation5(
        request: Request, action: str, csrf_token: str = Form(...)
    ) -> Response:
        check_csrf(request, csrf_token)
        service.control_generation5(action)
        return RedirectResponse("/generation", status_code=303)

    @app.post("/api/generation6/start")
    async def start_generation6(
        request: Request,
        manifest_path: str = Form("var/generations/generation-6.json"),
        csrf_token: str = Form(...),
    ) -> Response:
        check_csrf(request, csrf_token)
        path = Path(manifest_path)
        if not path.is_absolute():
            path = selected.project_root / path
        resolved = path.resolve()
        allowed = (selected.project_root / "var/generations").resolve()
        if allowed not in resolved.parents:
            raise ValueError("generation manifest must be inside var/generations")
        service.start_generation6(resolved)
        return RedirectResponse("/generation", status_code=303)

    @app.post("/api/generation6/{action}")
    async def control_generation6(
        request: Request, action: str, csrf_token: str = Form(...)
    ) -> Response:
        check_csrf(request, csrf_token)
        service.control_generation6(action)
        return RedirectResponse("/generation", status_code=303)

    @app.post("/api/risk/kill")
    async def kill(
        request: Request,
        reason: str = Form("operator activated from dashboard"),
        csrf_token: str = Form(...),
    ) -> Response:
        check_csrf(request, csrf_token)
        service.activate_kill_switch(reason)
        return RedirectResponse("/", status_code=303)

    @app.post("/api/risk/reset")
    async def reset_kill(request: Request, csrf_token: str = Form(...)) -> Response:
        check_csrf(request, csrf_token)
        service.reset_kill_switch()
        return RedirectResponse("/", status_code=303)

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc), "request_id": str(uuid4())}, status_code=409)

    return app


app = create_app()
