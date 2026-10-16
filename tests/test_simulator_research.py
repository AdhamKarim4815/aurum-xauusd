from __future__ import annotations

from pathlib import Path

from gold_trader.config import Settings
from gold_trader.data import generate_demo_data, normalize_bars, resample_bars
from gold_trader.database import Store
from gold_trader.research import ResearchRunner
from gold_trader.simulator import DeterministicSimulator
from gold_trader.strategies import strategy_registry


def test_simulator_is_deterministic(settings: Settings) -> None:
    normalized, _ = normalize_bars(generate_demo_data(rows=5000))
    bars = resample_bars(normalized, "1h")
    strategy = strategy_registry()["trend-pullback"]
    simulator = DeterministicSimulator(settings.risk, settings.simulation)
    first_trades, first_metrics = simulator.run(bars, strategy)
    second_trades, second_metrics = simulator.run(bars, strategy)
    assert first_metrics == second_metrics
    assert [trade.as_dict() | {"trade_id": None} for trade in first_trades] == [
        trade.as_dict() | {"trade_id": None} for trade in second_trades
    ]


def test_small_synthetic_dataset_is_never_eligible(
    tmp_path: Path, settings: Settings, store: Store
) -> None:
    dataset = tmp_path / "demo.csv"
    generate_demo_data(rows=5000).to_csv(dataset, index=False)
    report = ResearchRunner(settings, store).run(dataset, strategy_registry()["trend-pullback"])
    assert not report["eligible"]
    assert not report["gates"]["data_sufficiency"]
    assert report["walk_forward"] == []
    assert store.get_research_run(report["candidate_hash"]) is not None
    repeated = ResearchRunner(settings, store).run(dataset, strategy_registry()["trend-pullback"])
    assert repeated["run_id"] == report["run_id"]
