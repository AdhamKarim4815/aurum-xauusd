from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from gold_trader.config import Settings
from gold_trader.data import file_sha256, normalize_bars, resample_bars
from gold_trader.database import Store
from gold_trader.ml import (
    FEATURE_COLUMNS,
    META_ELIGIBILITY_THRESHOLD,
    META_LABEL_HORIZONS,
    ChronologicalSignalMetaModel,
    MetaFilteredStrategy,
    build_rule_signal_meta_events,
    write_model_card,
)
from gold_trader.research import ResearchRunner
from gold_trader.simulator import DeterministicSimulator, calculate_metrics
from gold_trader.strategies import GENERATION_TWO_VARIANTS, BaseStrategy


class MetaResearchRunner:
    """Train only before validation boundaries and evaluate the final test once."""

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store

    def run(self, dataset_path: Path, base_strategy: BaseStrategy) -> dict[str, Any]:
        if base_strategy.variant_id not in GENERATION_TWO_VARIANTS:
            raise ValueError("ML research is preregistered only for Generation-2 variants")
        if base_strategy.timeframe not in META_LABEL_HORIZONS:
            raise ValueError("no preregistered meta-label horizon for strategy timeframe")
        baseline = ResearchRunner(self.settings, self.store).run(dataset_path, base_strategy)
        raw = (
            pd.read_parquet(dataset_path)
            if dataset_path.suffix == ".parquet"
            else pd.read_csv(dataset_path)
        )
        normalized, quality = normalize_bars(raw)
        if not quality.valid:
            raise ValueError("dataset failed quality gates")
        bars = resample_bars(normalized, base_strategy.timeframe)
        timestamps = pd.to_datetime(bars["timestamp"], utc=True)
        holdout_start = pd.Timestamp(timestamps.max()) - pd.DateOffset(
            months=self.settings.research.holdout_months
        )
        horizon = META_LABEL_HORIZONS[base_strategy.timeframe]
        purge_bars = max(base_strategy.parameters.lookback, horizon)
        pre_holdout = bars.index[timestamps < holdout_start]
        if len(pre_holdout) <= purge_bars:
            raise ValueError("not enough development bars before the purged final holdout")
        development = bars.iloc[: int(pre_holdout.max()) + 1 - purge_bars].reset_index(drop=True)
        development_events = build_rule_signal_meta_events(development, base_strategy, horizon)
        model = ChronologicalSignalMetaModel().fit(development_events)
        filtered_strategy = MetaFilteredStrategy(base_strategy, model)

        all_events = build_rule_signal_meta_events(bars, base_strategy, horizon)
        holdout_events = all_events.loc[
            pd.to_datetime(all_events["timestamp"], utc=True) >= holdout_start
        ].copy()
        probability_evidence = self._probability_evidence(model, holdout_events)
        walk_forward = self._walk_forward(bars, holdout_start, base_strategy, horizon)

        holdout = bars.loc[timestamps >= holdout_start].reset_index(drop=True)
        _, filtered_metrics = DeterministicSimulator(
            self.settings.risk,
            self.settings.simulation,
            self.settings.starting_equity,
        ).run(holdout, filtered_strategy)
        baseline_metrics = baseline["metrics"]
        baseline_improvement = bool(
            filtered_metrics["sharpe"] > float(baseline_metrics["sharpe"])
            or filtered_metrics["max_drawdown"] < float(baseline_metrics["max_drawdown"])
        )
        report = ResearchRunner(self.settings, self.store).run(
            dataset_path,
            filtered_strategy,
            model_metadata=model.descriptor,
            walk_forward_override=walk_forward,
            extra_gates={
                "positive_probability_skill": probability_evidence["brier_skill"] > 0,
                "baseline_improvement": baseline_improvement,
            },
            report_metadata={
                "parent_candidate_hash": baseline["candidate_hash"],
                "base_variant_id": base_strategy.variant_id,
                "ml_evidence": {
                    "label_horizon_bars": horizon,
                    "threshold": META_ELIGIBILITY_THRESHOLD,
                    "holdout_probability": probability_evidence,
                    "baseline_metrics": baseline_metrics,
                    "filtered_metrics": filtered_metrics,
                },
            },
        )
        artifact = (
            self.settings.project_root / "artifacts/models" / f"{report['candidate_hash']}.joblib"
        )
        model.save(artifact)
        card = (
            self.settings.project_root
            / "artifacts/model-cards"
            / f"{report['candidate_hash']}.json"
        )
        write_model_card(
            card,
            {
                "model_id": model.model_id,
                "candidate_hash": report["candidate_hash"],
                "dataset_hash": report["dataset_hash"],
                "feature_schema": FEATURE_COLUMNS,
                "label_horizon": horizon,
                "calibration": model.training_summary,
                "metrics": probability_evidence,
                "limitations": [
                    "Rule-signal filter only; it cannot create signals or bypass risk.",
                    (
                        "Generation-1 market regimes were inspected before Generation-2 "
                        "preregistration."
                    ),
                    "Simulation and provider prices may differ from a future broker.",
                ],
            },
        )
        report["model_artifact"] = {
            "path": str(artifact.resolve()),
            "sha256": file_sha256(artifact),
            "format": "joblib-local-trusted-only",
        }
        report["model_card_path"] = str(card.resolve())
        self.store.save_research_run(report)
        report_path = (
            self.settings.project_root / "var/reports" / f"{report['candidate_hash']}.json"
        )
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return report

    def _walk_forward(
        self,
        bars: pd.DataFrame,
        holdout_start: pd.Timestamp,
        base_strategy: BaseStrategy,
        horizon: int,
    ) -> list[dict[str, Any]]:
        timestamps = pd.to_datetime(bars["timestamp"], utc=True)
        first = pd.Timestamp(timestamps.min())
        intraday = base_strategy.timeframe == "15min"
        train_offset = pd.DateOffset(months=24 if intraday else 60)
        validation_offset = pd.DateOffset(months=3 if intraday else 12)
        step_offset = pd.DateOffset(months=3 if intraday else 6)
        validation_start = first + train_offset
        folds: list[dict[str, Any]] = []
        while validation_start + validation_offset <= holdout_start:
            validation_end = validation_start + validation_offset
            validation_mask = (timestamps >= validation_start) & (timestamps < validation_end)
            validation_indices = bars.index[validation_mask]
            if len(validation_indices) == 0:
                validation_start += step_offset
                continue
            purge_bars = max(base_strategy.parameters.lookback, horizon)
            pre_validation = bars.index[timestamps < validation_start]
            if len(pre_validation) <= purge_bars:
                validation_start += step_offset
                continue
            training = bars.iloc[: int(pre_validation.max()) + 1 - purge_bars].reset_index(
                drop=True
            )
            training_events = build_rule_signal_meta_events(training, base_strategy, horizon)
            fold_report: dict[str, Any] = {
                "train_start": first.isoformat(),
                "train_end": validation_start.isoformat(),
                "validation_start": validation_start.isoformat(),
                "validation_end": validation_end.isoformat(),
                "purge_bars": purge_bars,
                "embargo_bars": purge_bars,
            }
            try:
                fold_model = ChronologicalSignalMetaModel().fit(training_events)
                fold_strategy = MetaFilteredStrategy(base_strategy, fold_model)
                context = max(base_strategy.warmup, 60)
                context_start = max(int(validation_indices.min()) - context, 0)
                context_end = int(validation_indices.max()) + 1
                fold_bars = bars.iloc[context_start:context_end].reset_index(drop=True)
                fold_trades, _ = DeterministicSimulator(
                    self.settings.risk,
                    self.settings.simulation,
                    self.settings.starting_equity,
                ).run(fold_bars, fold_strategy)
                start_at = validation_start.to_pydatetime()
                end_at = validation_end.to_pydatetime()
                validation_trades = [
                    trade for trade in fold_trades if start_at <= trade.entry_at < end_at
                ]
                fold_report["metrics"] = calculate_metrics(
                    validation_trades, self.settings.starting_equity
                )
                fold_report["model_status"] = "trained"
                fold_report["training_events"] = len(training_events)
            except ValueError as error:
                fold_report["metrics"] = calculate_metrics([], self.settings.starting_equity)
                fold_report["model_status"] = "ineligible"
                fold_report["model_reason"] = str(error)
                fold_report["training_events"] = len(training_events)
            folds.append(fold_report)
            validation_start += step_offset
        return folds

    @staticmethod
    def _probability_evidence(
        model: ChronologicalSignalMetaModel, events: pd.DataFrame
    ) -> dict[str, float]:
        clean = events.dropna(subset=[*FEATURE_COLUMNS, "label"])
        if len(clean) < 20 or clean["label"].nunique() < 2:
            return {
                "events": float(len(clean)),
                "brier": 1.0,
                "baseline_brier": 1.0,
                "brier_skill": -1.0,
                "roc_auc": 0.0,
            }
        probability = model.predict_events(clean)
        labels = clean["label"].astype(int).to_numpy()
        brier = float(brier_score_loss(labels, probability))
        prevalence = float(model.training_summary["fit_prevalence"])
        baseline_brier = float(np.mean((labels - prevalence) ** 2))
        return {
            "events": float(len(clean)),
            "brier": brier,
            "baseline_brier": baseline_brier,
            "brier_skill": 1 - brier / baseline_brier if baseline_brier > 0 else -1.0,
            "roc_auc": float(roc_auc_score(labels, probability)),
        }
