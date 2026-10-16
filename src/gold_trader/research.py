from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from gold_trader import __version__
from gold_trader.config import Settings
from gold_trader.data import file_sha256, normalize_bars, resample_bars
from gold_trader.database import Store
from gold_trader.simulator import DeterministicSimulator, calculate_metrics
from gold_trader.strategies import BaseStrategy


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class ResearchRunner:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store

    def run(
        self,
        dataset_path: Path,
        strategy: BaseStrategy,
        *,
        model_metadata: dict[str, Any] | None = None,
        walk_forward_override: list[dict[str, Any]] | None = None,
        extra_gates: dict[str, bool] | None = None,
        report_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = (
            pd.read_parquet(dataset_path)
            if dataset_path.suffix == ".parquet"
            else pd.read_csv(dataset_path)
        )
        normalized, quality = normalize_bars(raw)
        if not quality.valid:
            raise ValueError(f"dataset failed quality gates: {quality.as_dict()}")
        bars = resample_bars(normalized, strategy.timeframe)
        if len(bars) < strategy.warmup + 10:
            raise ValueError("dataset is too small for the strategy warmup")

        dataset_hash = file_sha256(dataset_path)
        configuration_hash = stable_hash(self.settings.model_dump())
        strategy_hash = strategy.strategy_hash
        model_descriptor = model_metadata or {"model": "rule-only", "version": 1}
        model_hash = stable_hash(model_descriptor)
        implementation_hash = stable_hash(
            {"package_version": __version__, "research_schema": 3, "simulator_schema": 2}
        )
        candidate_hash = stable_hash(
            {
                "dataset": dataset_hash,
                "configuration": configuration_hash,
                "strategy": strategy_hash,
                "model": model_hash,
                "implementation": implementation_hash,
            }
        )
        existing = self.store.get_research_run(candidate_hash)
        if existing is not None:
            return existing

        latest = pd.Timestamp(bars["timestamp"].max())
        holdout_start = latest - pd.DateOffset(months=self.settings.research.holdout_months)
        holdout = bars[pd.to_datetime(bars["timestamp"], utc=True) >= holdout_start].reset_index(
            drop=True
        )
        simulator = DeterministicSimulator(
            self.settings.risk,
            self.settings.simulation,
            self.settings.starting_equity,
        )
        trades, metrics = simulator.run(holdout, strategy)
        stressed_simulation = self.settings.simulation.model_copy(
            update={
                "spread_points": self.settings.simulation.spread_points * 2,
                "slippage_points": self.settings.simulation.slippage_points * 2,
                "commission_per_lot_round_trip": (
                    self.settings.simulation.commission_per_lot_round_trip * 2
                ),
            }
        )
        stressed_trades, stressed_metrics = DeterministicSimulator(
            self.settings.risk,
            stressed_simulation,
            self.settings.starting_equity,
        ).run(holdout, strategy)

        duration_years = (
            pd.Timestamp(bars["timestamp"].max()) - pd.Timestamp(bars["timestamp"].min())
        ).total_seconds() / (365.25 * 86400)
        intraday = strategy.timeframe in {"5min", "15min"}
        minimum_years = 4 if intraday else 8
        minimum_trades = (
            self.settings.research.intraday_min_trades
            if intraday
            else self.settings.research.swing_min_trades
        )
        walk_forward = (
            walk_forward_override
            if walk_forward_override is not None
            else self._walk_forward(bars, holdout_start, strategy, intraday)
        )
        fold_stats = self._fold_statistics(walk_forward)
        gates = {
            "data_sufficiency": duration_years >= minimum_years,
            "positive_base_return": metrics["net_return"] > 0,
            "positive_stressed_return": stressed_metrics["net_return"] > 0,
            "profit_factor": metrics["profit_factor"] >= self.settings.research.min_profit_factor,
            "sharpe": metrics["sharpe"] >= self.settings.research.min_sharpe,
            "drawdown": metrics["max_drawdown"] <= self.settings.research.max_drawdown,
            "trade_count": metrics["trade_count"] >= minimum_trades,
            "fold_stability": fold_stats["profitable_fraction"]
            >= self.settings.research.min_profitable_fold_fraction,
            "profit_concentration": fold_stats["largest_profit_fraction"] <= 0.50,
            "data_quality": quality.valid,
        }
        gates.update(extra_gates or {})
        eligible = all(gates.values())
        report = {
            "run_id": str(uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
            "strategy_id": strategy.strategy_id,
            "variant_id": strategy.variant_id,
            "timeframe": strategy.timeframe,
            "candidate_hash": candidate_hash,
            "dataset_hash": dataset_hash,
            "configuration_hash": configuration_hash,
            "strategy_hash": strategy_hash,
            "model_hash": model_hash,
            "model": model_descriptor,
            "implementation_hash": implementation_hash,
            "status": "eligible" if eligible else "ineligible",
            "eligible": eligible,
            "metrics": metrics,
            "stressed_metrics": stressed_metrics,
            "gates": gates,
            "folds": fold_stats,
            "data": {
                "dataset_path": str(dataset_path.resolve()),
                "rows": len(bars),
                "duration_years": duration_years,
                "holdout_start": holdout_start.isoformat(),
                "holdout_rows": len(holdout),
                "quality": quality.as_dict(),
            },
            "trades": [trade.as_dict() for trade in trades],
            "walk_forward": walk_forward,
            "stressed_trade_count": len(stressed_trades),
            "disclaimer": "Simulation only; no profitability claim and no live-order capability.",
        }
        report.update(report_metadata or {})
        self.store.save_research_run(report)
        report_path = self.settings.project_root / "var/reports" / f"{candidate_hash}.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report

    def _walk_forward(
        self,
        bars: pd.DataFrame,
        holdout_start: pd.Timestamp,
        strategy: BaseStrategy,
        intraday: bool,
    ) -> list[dict[str, Any]]:
        timestamps = pd.to_datetime(bars["timestamp"], utc=True)
        first = pd.Timestamp(timestamps.min())
        train_offset = pd.DateOffset(months=24 if intraday else 60)
        validation_offset = pd.DateOffset(months=3 if intraday else 12)
        step_offset = pd.DateOffset(months=3 if intraday else 6)
        validation_start = first + train_offset
        folds: list[dict[str, Any]] = []
        simulator = DeterministicSimulator(
            self.settings.risk,
            self.settings.simulation,
            self.settings.starting_equity,
        )
        while validation_start + validation_offset <= holdout_start:
            validation_end = validation_start + validation_offset
            validation_mask = (timestamps >= validation_start) & (timestamps < validation_end)
            validation_indices = bars.index[validation_mask]
            if len(validation_indices) == 0:
                validation_start += step_offset
                continue
            context_start = max(int(validation_indices.min()) - strategy.warmup, 0)
            context_end = int(validation_indices.max()) + 1
            fold_bars = bars.iloc[context_start:context_end].reset_index(drop=True)
            fold_trades, _ = simulator.run(fold_bars, strategy)
            validation_start_time = validation_start.to_pydatetime()
            validation_end_time = validation_end.to_pydatetime()
            validation_trades = [
                trade
                for trade in fold_trades
                if validation_start_time <= trade.entry_at < validation_end_time
            ]
            folds.append(
                {
                    "train_start": first.isoformat(),
                    "train_end": validation_start.isoformat(),
                    "validation_start": validation_start.isoformat(),
                    "validation_end": validation_end.isoformat(),
                    "purge_bars": strategy.parameters.lookback,
                    "embargo_bars": strategy.parameters.lookback,
                    "metrics": calculate_metrics(validation_trades, self.settings.starting_equity),
                }
            )
            validation_start += step_offset
        return folds

    @staticmethod
    def _fold_statistics(folds: list[dict[str, Any]]) -> dict[str, float]:
        if not folds:
            return {"count": 0.0, "profitable_fraction": 0.0, "largest_profit_fraction": 1.0}
        pnl = pd.Series([float(fold["metrics"]["net_pnl"]) for fold in folds])
        profitable = pnl[pnl > 0]
        total_profit = float(profitable.sum())
        concentration = float(profitable.max() / total_profit) if total_profit > 0 else 1.0
        return {
            "count": float(len(folds)),
            "profitable_fraction": float((pnl > 0).mean()),
            "largest_profit_fraction": concentration,
        }
