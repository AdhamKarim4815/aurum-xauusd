from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gold_trader.data import generate_demo_data
from gold_trader.domain import Side
from gold_trader.ml import (
    FEATURE_COLUMNS,
    ChronologicalSignalMetaModel,
    LightGBMMetaModel,
    LocalExperimentTracker,
    LogisticMetaModel,
    MetaFilteredStrategy,
    PurgedWalkForwardSplitter,
    TemporalConvolutionalMetaModel,
    build_rule_signal_meta_events,
    feature_ablation_sets,
    optional_ml_backends,
    triple_barrier_labels,
    write_model_card,
)
from gold_trader.strategies import BaseStrategy, StrategyParameters


class FixedSignalStrategy(BaseStrategy):
    strategy_id = "fixed-signal"
    timeframe = "1h"
    warmup = 0

    def __init__(self, signal_indices: tuple[int, ...] = (100,)) -> None:
        super().__init__(StrategyParameters(lookback=20, stop_atr=1.0, reward_risk=1.0))
        self.signal_indices = signal_indices

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=bars.index, data={"side": None, "reason": None, "atr": 1.0})
        for index in self.signal_indices:
            if index < len(result):
                result.loc[index, ["side", "reason"]] = [Side.BUY.value, "fixed test signal"]
        return result


def chronological_events(rows: int = 400) -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=rows, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            column: np.sin(np.arange(rows) / (position + 2)) + np.arange(rows) * 0.0001
            for position, column in enumerate(FEATURE_COLUMNS)
        }
    )
    frame["timestamp"] = timestamps
    frame["label"] = np.arange(rows) % 2
    return frame


def test_purged_walk_forward_splitter_enforces_embargo() -> None:
    folds = PurgedWalkForwardSplitter(train_size=100, validation_size=25, embargo=5).split(250)
    assert folds
    for fold in folds:
        assert fold.train[-1] + 5 < fold.validation[0]
        assert len(fold.validation) == 25


def test_logistic_meta_model_fit_predict_and_save(tmp_path: Path) -> None:
    bars = generate_demo_data(rows=650)
    labels = triple_barrier_labels(bars, horizon=8)
    model = LogisticMetaModel().fit(bars, labels)
    probability = model.predict_probability(bars)
    assert 0 <= probability <= 1
    path = tmp_path / "model.joblib"
    model.save(str(path))
    assert path.exists()


def test_signal_meta_label_is_stop_first_and_ignores_bars_after_horizon() -> None:
    bars = generate_demo_data(rows=180)
    entry = float(bars.iloc[101]["open"])
    bars.loc[101, "low"] = entry - 2.0
    bars.loc[101, "high"] = entry + 2.0
    strategy = FixedSignalStrategy()

    original = build_rule_signal_meta_events(bars, strategy, horizon=5)
    changed = bars.copy()
    changed.loc[106:, ["open", "high", "low", "close"]] = 1_000_000.0
    after_horizon_changed = build_rule_signal_meta_events(changed, strategy, horizon=5)

    assert original.loc[0, "label"] == 0
    assert after_horizon_changed.loc[0, "label"] == original.loc[0, "label"]
    assert original.loc[0, "timestamp"] == after_horizon_changed.loc[0, "timestamp"]


def test_meta_filter_can_only_remove_parent_signals() -> None:
    bars = generate_demo_data(rows=140)
    parent = FixedSignalStrategy((100, 110))

    class FixedProbabilities:
        def predict_frame(self, values: pd.DataFrame) -> pd.Series:
            probabilities = pd.Series(0.99, index=values.index)
            probabilities.loc[100] = 0.10
            return probabilities

    filtered = MetaFilteredStrategy(parent, FixedProbabilities())  # type: ignore[arg-type]
    parent_candidates = parent.precompute(bars)
    filtered_candidates = filtered.precompute(bars)

    assert set(filtered_candidates.index[filtered_candidates["side"].notna()]).issubset(
        set(parent_candidates.index[parent_candidates["side"].notna()])
    )
    assert pd.isna(filtered_candidates.loc[100, "side"])
    assert filtered_candidates.loc[110, "side"] == Side.BUY.value


def test_chronological_meta_model_keeps_fit_before_evaluation() -> None:
    model = ChronologicalSignalMetaModel().fit(chronological_events())

    assert model.training_summary["fit_events"] == 280
    assert model.training_summary["calibration_events"] == 60
    assert model.training_summary["evaluation_events"] == 60
    assert model.training_summary["fit_end"] < model.training_summary["evaluation_end"]
    probabilities = model.predict_events(chronological_events().iloc[-20:])
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_optional_backend_diagnostics_are_boolean() -> None:
    availability = optional_ml_backends()
    assert set(availability) == {"lightgbm", "torch", "transformers", "optuna", "shap", "mlflow"}
    assert all(isinstance(value, bool) for value in availability.values())


def test_feature_ablations_and_model_card_contract(tmp_path: Path) -> None:
    sets = feature_ablation_sets()
    assert set(sets) == {"price_only", "cross_market", "macro_positioning", "news"}
    card = tmp_path / "model-card.json"
    payload = {
        "model_id": "elastic-net-logistic-v1",
        "candidate_hash": "a" * 64,
        "dataset_hash": "d" * 64,
        "feature_schema": sets["price_only"],
        "label_horizon": 12,
        "calibration": {"method": "sigmoid"},
        "metrics": {"brier": 0.2},
        "limitations": ["simulation only"],
    }
    write_model_card(card, payload)
    assert card.exists()
    with pytest.raises(ValueError, match="missing"):
        write_model_card(tmp_path / "bad.json", {"model_id": "incomplete"})


@pytest.mark.optional_ml
def test_lightgbm_backend_smoke() -> None:
    pytest.importorskip("lightgbm")
    bars = generate_demo_data(rows=650)
    labels = triple_barrier_labels(bars, horizon=8)
    lightgbm = LightGBMMetaModel(n_estimators=20).fit(bars, labels)
    assert 0 <= lightgbm.predict_probability(bars) <= 1


@pytest.mark.optional_ml
def test_tcn_backend_smoke() -> None:
    pytest.importorskip("torch")
    bars = generate_demo_data(rows=220)
    labels = triple_barrier_labels(bars, horizon=8)
    tcn = TemporalConvolutionalMetaModel(sequence_length=16, epochs=1, device_preference="cpu").fit(
        bars, labels
    )
    assert 0 <= tcn.predict_probability(bars) <= 1


@pytest.mark.optional_ml
def test_mlflow_backend_smoke(tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    tracker = LocalExperimentTracker(tmp_path / "mlflow.sqlite3", tmp_path / "artifacts")
    run_id = tracker.log(
        "optional-smoke",
        {"model": "fixture"},
        {"probability": 0.5},
        {"scope": "test"},
    )
    assert run_id
