from __future__ import annotations

import re

from fastapi.testclient import TestClient

from gold_trader.config import Settings
from gold_trader.database import Store
from gold_trader.web import create_app


def test_dashboard_and_read_apis(settings: Settings, store: Store) -> None:
    client = TestClient(create_app(settings, store))
    response = client.get("/")
    assert response.status_code == 200
    assert "No live execution venue" in response.text
    assert client.get("/api/status").json()["live_execution_available"] is False
    assert client.get("/api/audit").status_code == 200


def test_write_requires_csrf(settings: Settings, store: Store) -> None:
    client = TestClient(create_app(settings, store))
    response = client.post("/api/risk/kill", data={"csrf_token": "wrong"})
    assert response.status_code == 403


def test_csrf_protected_kill(settings: Settings, store: Store) -> None:
    client = TestClient(create_app(settings, store))
    page = client.get("/")
    match = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
    assert match is not None
    response = client.post(
        "/api/risk/kill",
        data={"csrf_token": match.group(1), "reason": "http test"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert store.get_state("paper")["kill_switch"] is True  # type: ignore[index]


def test_no_manual_or_live_order_route(settings: Settings, store: Store) -> None:
    app = create_app(settings, store)
    paths = {route.path for route in app.routes}
    assert not any("order" in path or "trade" in path or "live-execution" in path for path in paths)


def test_generation_card_shows_rejection_reasons_and_thresholds(
    settings: Settings, store: Store
) -> None:
    store.save_generation("generation-test", "g" * 64, "stopped", {}, {})
    store.save_trial(
        "generation-test",
        {
            "trial_hash": "t" * 64,
            "family": "time-series-momentum",
            "ordinal": 10,
            "parameters": {"timeframe": "4h", "lookback": 80},
        },
        {
            "status": "rejected",
            "stage": 2,
            "score": 0.01,
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
            "gates": {"positive_stressed_return": False},
        },
    )
    store.save_trial(
        "generation-test",
        {
            "trial_hash": "p" * 64,
            "family": "mean-reversion",
            "ordinal": 11,
            "parameters": {"timeframe": "15min", "lookback": 20},
        },
        {
            "status": "pruned",
            "stage": 1,
            "score": -0.2,
            "metrics": {
                "net_return": -0.03,
                "profit_factor": 0.8,
                "sharpe": -0.5,
                "max_drawdown": 0.04,
                "trade_count": 12,
            },
            "gates": {"stage_one": False},
        },
    )

    client = TestClient(create_app(settings, store))
    response = client.get("/generation?generation_id=generation-test")

    assert response.status_code == 200
    assert "Why it was rejected" in response.text
    assert "Doubled-cost net return was -2.27%; required &gt; 0.00%." in response.text
    assert "2\u00d7 cost return" in response.text
    assert "Not run" in response.text
    assert "generation-test" in response.text
    assert client.get("/generation?generation_id=missing").status_code == 404


def test_strategy_library_and_one_click_approval(settings: Settings, store: Store) -> None:
    store.save_generation(
        "generation-3-r4",
        "g" * 64,
        "completed",
        {"dataset_path": str(settings.project_root / "var/data/xauusd_5m.parquet")},
        {"completed_trials": 1, "shortlisted": 1},
    )
    store.save_trial(
        "generation-3-r4",
        {
            "generation_id": "generation-3-r4",
            "trial_hash": "t" * 64,
            "family": "time-series-momentum",
            "ordinal": 0,
            "parameters": {"timeframe": "4h", "lookback": 80},
        },
        {
            "status": "shortlisted",
            "stage": 2,
            "score": 0.5,
            "metrics": {
                "net_return": 0.12,
                "profit_factor": 1.3,
                "sharpe": 1.1,
                "max_drawdown": 0.04,
                "trade_count": 50,
            },
            "stressed_metrics": {"net_return": 0.08},
            "fold_profitable_fraction": 0.8,
            "largest_profit_fraction": 0.3,
            "gates": {"positive_return": True},
        },
    )
    candidate_hash = "a" * 64
    store.save_research_run(
        {
            "run_id": "eligible-web-run",
            "created_at": "2026-01-01T00:00:00+00:00",
            "strategy_id": "time-series-momentum",
            "variant_id": "g3-test",
            "timeframe": "4h",
            "candidate_hash": candidate_hash,
            "dataset_hash": "d" * 64,
            "configuration_hash": "c" * 64,
            "strategy_hash": "s" * 64,
            "model_hash": "m" * 64,
            "status": "eligible",
            "eligible": True,
            "metrics": {
                "net_return": 0.07,
                "profit_factor": 1.2,
                "sharpe": 0.9,
                "max_drawdown": 0.05,
                "trade_count": 44,
            },
            "gates": {"all": True},
        }
    )

    client = TestClient(create_app(settings, store))
    page = client.get("/strategies")
    assert page.status_code == 200
    assert "Generation 3 through Generation 6" in page.text
    assert "12.00%" in page.text
    assert "Approve strategy" in page.text
    assert f"/api/reports/{candidate_hash}" in page.text
    assert "Type the full candidate hash" not in page.text
    assert client.get(f"/api/reports/{candidate_hash}").json()["candidate_hash"] == candidate_hash
    assert client.get(f"/api/reports/{'z' * 64}").status_code == 404
    token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
    assert token is not None

    response = client.post(
        f"/api/candidates/{candidate_hash}/approve",
        data={"csrf_token": token.group(1)},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/strategies"
    assert store.valid_approval()["candidate_hash"] == candidate_hash  # type: ignore[index]


def test_generation5_shortlist_and_controls_are_visible(settings: Settings, store: Store) -> None:
    store.save_generation(
        "generation-5-test",
        "5" * 64,
        "completed",
        {},
        {
            "manifest_path": "var/generations/generation-5.json",
            "completed_trials": 1,
            "total_trials": 1,
            "shortlisted": 1,
        },
    )
    store.save_trial(
        "generation-5-test",
        {
            "generation_id": "generation-5-test",
            "trial_hash": "5" * 64,
            "family": "trend-pullback",
            "ordinal": 0,
            "parameters": {
                "timeframe": "15min",
                "max_holding_bars": 16,
            },
        },
        {
            "status": "shortlisted",
            "stage": 2,
            "score": 1.0,
            "metrics": {
                "net_return": 0.2,
                "profit_factor": 1.4,
                "sharpe": 1.1,
                "max_drawdown": 0.03,
                "trade_count": 500,
            },
            "stressed_metrics": {"net_return": 0.1},
            "gate_details": [],
        },
    )
    client = TestClient(create_app(settings, store))

    strategies = client.get("/strategies")
    generation = client.get("/generation?generation_id=generation-5-test")

    assert strategies.status_code == 200
    assert "Generation 3 through Generation 6" in strategies.text
    assert "generation-5-test" in strategies.text
    assert generation.status_code == 200
    assert 'action="/api/generation5/start"' in generation.text
    assert 'action="/api/generation5/pause"' in generation.text


def test_generation6_shortlist_and_controls_are_visible(settings: Settings, store: Store) -> None:
    store.save_generation(
        "generation-6-test",
        "6" * 64,
        "completed",
        {},
        {
            "manifest_path": "var/generations/generation-6.json",
            "completed_trials": 1,
            "total_trials": 1,
            "shortlisted": 1,
        },
    )
    store.save_trial(
        "generation-6-test",
        {
            "generation_id": "generation-6-test",
            "trial_hash": "6" * 64,
            "family": "trend-state",
            "ordinal": 0,
            "parameters": {"timeframe": "15min", "max_holding_bars": 8},
        },
        {
            "status": "shortlisted",
            "stage": 2,
            "score": 1.0,
            "metrics": {
                "net_return": 0.25,
                "profit_factor": 1.4,
                "sharpe": 1.2,
                "max_drawdown": 0.03,
                "trade_count": 600,
            },
            "stressed_metrics": {"net_return": 0.12},
            "gate_details": [],
        },
    )
    client = TestClient(create_app(settings, store))

    strategies = client.get("/strategies")
    generation = client.get("/generation?generation_id=generation-6-test")

    assert strategies.status_code == 200
    assert "generation-6-test" in strategies.text
    assert generation.status_code == 200
    assert 'action="/api/generation6/start"' in generation.text
    assert 'action="/api/generation6/pause"' in generation.text
