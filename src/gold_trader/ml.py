from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gold_trader.domain import Side
from gold_trader.features import build_feature_frame
from gold_trader.strategies import BaseStrategy

FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_12",
    "volatility_12",
    "volatility_48",
    "ema_fast_gap",
    "ema_slow_gap",
    "rsi_14",
    "atr_14",
    "range_fraction",
    "hour_sin",
    "hour_cos",
    "weekday",
]

META_LABEL_HORIZONS = {"15min": 24, "1h": 24, "4h": 12}
META_ELIGIBILITY_THRESHOLD = 0.55


@dataclass(frozen=True)
class Fold:
    train: np.ndarray[Any, np.dtype[np.int64]]
    validation: np.ndarray[Any, np.dtype[np.int64]]


class PurgedWalkForwardSplitter:
    def __init__(self, train_size: int, validation_size: int, embargo: int) -> None:
        self.train_size = train_size
        self.validation_size = validation_size
        self.embargo = embargo

    def split(self, length: int) -> list[Fold]:
        folds: list[Fold] = []
        validation_start = self.train_size + self.embargo
        while validation_start + self.validation_size <= length:
            train_end = validation_start - self.embargo
            train_start = max(0, train_end - self.train_size)
            folds.append(
                Fold(
                    train=np.arange(train_start, train_end, dtype=np.int64),
                    validation=np.arange(
                        validation_start,
                        validation_start + self.validation_size,
                        dtype=np.int64,
                    ),
                )
            )
            validation_start += self.validation_size
        return folds


def triple_barrier_labels(
    bars: pd.DataFrame, horizon: int = 12, stop_atr: float = 1.5, target_atr: float = 2.0
) -> pd.Series:
    features = build_feature_frame(bars)
    labels = pd.Series(np.nan, index=bars.index, dtype=float)
    for index in range(len(bars) - horizon):
        price = float(bars.iloc[index]["close"])
        volatility = float(features.iloc[index]["atr_14"])
        if np.isnan(volatility):
            continue
        upper = price + volatility * target_atr
        lower = price - volatility * stop_atr
        future = bars.iloc[index + 1 : index + horizon + 1]
        for _, row in future.iterrows():
            if float(row["low"]) <= lower:
                labels.iloc[index] = 0
                break
            if float(row["high"]) >= upper:
                labels.iloc[index] = 1
                break
        else:
            labels.iloc[index] = float(future.iloc[-1]["close"] > price)
    return labels


class LogisticMetaModel:
    model_id = "elastic-net-logistic-v1"

    def __init__(self) -> None:
        base = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.25,
                        class_weight="balanced",
                        random_state=42,
                        max_iter=3000,
                    ),
                ),
            ]
        )
        self.model = CalibratedClassifierCV(base, method="sigmoid", cv=3)

    def fit(self, bars: pd.DataFrame, labels: pd.Series) -> LogisticMetaModel:
        features = build_feature_frame(bars)[FEATURE_COLUMNS]
        valid = features.notna().all(axis=1) & labels.notna()
        if valid.sum() < 100:
            raise ValueError("at least 100 labeled, complete rows are required")
        self.model.fit(features.loc[valid], labels.loc[valid].astype(int))
        return self

    def predict_probability(self, bars: pd.DataFrame) -> float:
        features = build_feature_frame(bars)[FEATURE_COLUMNS].iloc[[-1]]
        if features.isna().any(axis=None):
            raise ValueError("latest feature vector is incomplete")
        return float(self.model.predict_proba(features)[0, 1])

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)


def build_rule_signal_meta_events(
    bars: pd.DataFrame, strategy: BaseStrategy, horizon: int
) -> pd.DataFrame:
    """Build side-aware labels only at deterministic rule-signal timestamps."""
    if horizon < 1:
        raise ValueError("meta-label horizon must be positive")
    candidates = strategy.precompute(bars)
    features = build_feature_frame(bars)[FEATURE_COLUMNS]
    records: list[dict[str, Any]] = []
    signal_indices = candidates.index[candidates["side"].notna()]
    for index_value in signal_indices:
        index = int(index_value)
        if index + horizon >= len(bars) or features.iloc[index].isna().any():
            continue
        candidate = candidates.iloc[index]
        side = Side(str(candidate["side"]))
        signal_price = float(bars.iloc[index]["close"])
        entry_price = float(bars.iloc[index + 1]["open"])
        atr_value = float(candidate["atr"])
        distance = max(
            atr_value * strategy.parameters.stop_atr,
            signal_price * 0.0005,
        )
        if side is Side.BUY:
            stop = entry_price - distance
            target = entry_price + distance * strategy.parameters.reward_risk
        else:
            stop = entry_price + distance
            target = entry_price - distance * strategy.parameters.reward_risk
        label = 0
        for cursor in range(index + 1, index + horizon + 1):
            future = bars.iloc[cursor]
            if side is Side.BUY:
                if float(future["low"]) <= stop:
                    label = 0
                    break
                if float(future["high"]) >= target:
                    label = 1
                    break
            else:
                if float(future["high"]) >= stop:
                    label = 0
                    break
                if float(future["low"]) <= target:
                    label = 1
                    break
        record: dict[str, Any] = {
            column: float(features.iloc[index][column]) for column in FEATURE_COLUMNS
        }
        record.update(
            {
                "bar_index": index,
                "timestamp": pd.Timestamp(bars.iloc[index]["timestamp"]),
                "side": side.value,
                "label": label,
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(
        records,
        columns=[*FEATURE_COLUMNS, "bar_index", "timestamp", "side", "label"],
    )


class ChronologicalSignalMetaModel:
    """Elastic-net scorer with chronological fit, calibration, and evaluation slices."""

    model_id = "signal-elastic-net-logistic-v2"

    def __init__(self) -> None:
        self.base = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.25,
                        class_weight="balanced",
                        random_state=42,
                        max_iter=3000,
                    ),
                ),
            ]
        )
        self.calibrator = LogisticRegression(random_state=42)
        self.training_summary: dict[str, Any] = {}

    def fit(self, events: pd.DataFrame) -> ChronologicalSignalMetaModel:
        clean = events.dropna(subset=[*FEATURE_COLUMNS, "label"]).sort_values("timestamp")
        if len(clean) < 300:
            raise ValueError("at least 300 complete chronological signal events are required")
        fit_end = int(len(clean) * 0.70)
        calibration_end = int(len(clean) * 0.85)
        fit = clean.iloc[:fit_end]
        calibration = clean.iloc[fit_end:calibration_end]
        evaluation = clean.iloc[calibration_end:]
        for name, sample in (
            ("fit", fit),
            ("calibration", calibration),
            ("evaluation", evaluation),
        ):
            if sample["label"].nunique() < 2:
                raise ValueError(f"meta-label {name} slice requires both classes")
        self.base.fit(fit[FEATURE_COLUMNS], fit["label"].astype(int))
        calibration_raw = self.base.predict_proba(calibration[FEATURE_COLUMNS])[:, 1]
        self.calibrator.fit(
            self._logits(calibration_raw).reshape(-1, 1),
            calibration["label"].astype(int),
        )
        evaluation_probability = self.predict_events(evaluation)
        evaluation_label = evaluation["label"].astype(int).to_numpy()
        brier = float(brier_score_loss(evaluation_label, evaluation_probability))
        prevalence = float(fit["label"].mean())
        baseline_brier = float(np.mean((evaluation_label - prevalence) ** 2))
        self.training_summary = {
            "events": len(clean),
            "fit_events": len(fit),
            "calibration_events": len(calibration),
            "evaluation_events": len(evaluation),
            "fit_start": pd.Timestamp(fit["timestamp"].min()).isoformat(),
            "fit_end": pd.Timestamp(fit["timestamp"].max()).isoformat(),
            "evaluation_end": pd.Timestamp(evaluation["timestamp"].max()).isoformat(),
            "brier": brier,
            "baseline_brier": baseline_brier,
            "brier_skill": 1 - brier / baseline_brier if baseline_brier > 0 else 0.0,
            "roc_auc": float(roc_auc_score(evaluation_label, evaluation_probability)),
            "fit_prevalence": prevalence,
        }
        return self

    @staticmethod
    def _logits(probability: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, Any]:
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        return np.asarray(np.log(clipped / (1 - clipped)), dtype=np.float64)

    def predict_events(self, events: pd.DataFrame) -> np.ndarray[Any, Any]:
        raw = self.base.predict_proba(events[FEATURE_COLUMNS])[:, 1]
        probability = self.calibrator.predict_proba(self._logits(raw).reshape(-1, 1))[:, 1]
        return np.asarray(probability, dtype=np.float64)

    def predict_frame(self, bars: pd.DataFrame) -> pd.Series:
        features = build_feature_frame(bars)[FEATURE_COLUMNS]
        complete = features.notna().all(axis=1)
        probability = pd.Series(np.nan, index=bars.index, dtype=float)
        if complete.any():
            scored = features.loc[complete]
            raw = self.base.predict_proba(scored)[:, 1]
            probability.loc[complete] = self.calibrator.predict_proba(
                self._logits(raw).reshape(-1, 1)
            )[:, 1]
        return probability

    @property
    def descriptor(self) -> dict[str, Any]:
        scaler = self.base.named_steps["scale"]
        model = self.base.named_steps["model"]
        return {
            "model": self.model_id,
            "version": 2,
            "features": FEATURE_COLUMNS,
            "split": {"fit": 0.70, "calibration": 0.15, "evaluation": 0.15},
            "threshold": META_ELIGIBILITY_THRESHOLD,
            "random_seed": 42,
            "training": self.training_summary,
            "parameters": {
                "scaler_mean": np.asarray(scaler.mean_).round(12).tolist(),
                "scaler_scale": np.asarray(scaler.scale_).round(12).tolist(),
                "base_coef": np.asarray(model.coef_).round(12).tolist(),
                "base_intercept": np.asarray(model.intercept_).round(12).tolist(),
                "calibration_coef": np.asarray(self.calibrator.coef_).round(12).tolist(),
                "calibration_intercept": np.asarray(self.calibrator.intercept_).round(12).tolist(),
            },
        }

    @property
    def model_hash(self) -> str:
        payload = json.dumps(self.descriptor, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


class MetaFilteredStrategy(BaseStrategy):
    """A fitted model may suppress a rule signal but cannot create or resize one."""

    def __init__(
        self,
        base_strategy: BaseStrategy,
        model: ChronologicalSignalMetaModel,
        threshold: float = META_ELIGIBILITY_THRESHOLD,
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("meta-filter threshold must be between zero and one")
        self.base_strategy = base_strategy
        self.model = model
        self.threshold = threshold
        self.strategy_id = base_strategy.strategy_id
        self.timeframe = base_strategy.timeframe
        self.warmup = max(base_strategy.warmup, 60)
        super().__init__(
            base_strategy.parameters,
            f"{base_strategy.variant_id}-meta-elastic-net-t{int(threshold * 100):02d}",
        )

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        candidates = self.base_strategy.precompute(bars).copy()
        probability = self.model.predict_frame(bars)
        proposed = candidates["side"].notna()
        allowed = proposed & probability.ge(self.threshold)
        candidates.loc[proposed & ~allowed, ["side", "reason"]] = None
        candidates.loc[allowed, "reason"] = (
            candidates.loc[allowed, "reason"].astype(str)
            + "; meta score passed fixed eligibility threshold"
        )
        candidates["model_probability"] = probability
        return candidates


def optional_ml_backends() -> dict[str, bool]:
    return {
        name: importlib.util.find_spec(name) is not None
        for name in ("lightgbm", "torch", "transformers", "optuna", "shap", "mlflow")
    }


class LightGBMMetaModel:
    """Optional nonlinear tabular scorer with a deliberately bounded shape."""

    model_id = "lightgbm-meta-v1"

    def __init__(self, **overrides: Any) -> None:
        lightgbm = importlib.import_module("lightgbm")
        parameters = {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        parameters.update(overrides)
        self.model: Any = lightgbm.LGBMClassifier(**parameters)

    def fit(self, bars: pd.DataFrame, labels: pd.Series) -> LightGBMMetaModel:
        features = build_feature_frame(bars)[FEATURE_COLUMNS]
        valid = features.notna().all(axis=1) & labels.notna()
        if valid.sum() < 100:
            raise ValueError("at least 100 labeled, complete rows are required")
        self.model.fit(features.loc[valid], labels.loc[valid].astype(int))
        return self

    def predict_probability(self, bars: pd.DataFrame) -> float:
        features = build_feature_frame(bars)[FEATURE_COLUMNS].iloc[[-1]]
        if features.isna().any(axis=None):
            raise ValueError("latest feature vector is incomplete")
        return float(self.model.predict_proba(features)[0, 1])


class TemporalConvolutionalMetaModel:
    """Small local sequence model; deterministic risk remains outside the network."""

    model_id = "tcn-meta-v1"

    def __init__(
        self,
        sequence_length: int = 64,
        epochs: int = 15,
        device_preference: str = "auto",
    ) -> None:
        if device_preference not in {"auto", "cpu", "mps"}:
            raise ValueError("device_preference must be auto, cpu, or mps")
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.device_preference = device_preference
        self.scaler = StandardScaler()
        self.model: Any = None
        self.device = "cpu"

    def fit(self, bars: pd.DataFrame, labels: pd.Series) -> TemporalConvolutionalMetaModel:
        torch = importlib.import_module("torch")
        nn = torch.nn
        features = build_feature_frame(bars)[FEATURE_COLUMNS]
        valid = features.notna().all(axis=1) & labels.notna()
        clean_features = features.loc[valid]
        clean_labels = labels.loc[valid].astype(float)
        if len(clean_features) < self.sequence_length + 100:
            raise ValueError("not enough complete observations for the TCN")
        scaled = self.scaler.fit_transform(clean_features)
        sequences: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        targets: list[float] = []
        for index in range(self.sequence_length, len(scaled)):
            sequences.append(scaled[index - self.sequence_length : index])
            targets.append(float(clean_labels.iloc[index]))
        x = torch.tensor(np.asarray(sequences), dtype=torch.float32).transpose(1, 2)
        y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        mps_available = bool(torch.backends.mps.is_available())
        if self.device_preference == "mps" and not mps_available:
            raise ValueError("MPS was requested but is unavailable")
        self.device = (
            "mps" if mps_available and self.device_preference in {"auto", "mps"} else "cpu"
        )

        class CompactTcn(nn.Module):  # type: ignore[misc, name-defined]
            def __init__(self, input_channels: int) -> None:
                super().__init__()
                self.network = nn.Sequential(
                    nn.Conv1d(input_channels, 32, kernel_size=3, padding=2, dilation=1),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=2),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.head = nn.Linear(16, 1)

            def forward(self, values: Any) -> Any:
                return self.head(self.network(values).squeeze(-1))

        torch.manual_seed(42)
        model = CompactTcn(len(FEATURE_COLUMNS)).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        loss_function = nn.BCEWithLogitsLoss()
        x, y = x.to(self.device), y.to(self.device)
        model.train()
        for _epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_function(model(x), y)
            loss.backward()
            optimizer.step()
        self.model = model.eval()
        return self

    def predict_probability(self, bars: pd.DataFrame) -> float:
        if self.model is None:
            raise ValueError("TCN has not been fitted")
        torch = importlib.import_module("torch")
        features = build_feature_frame(bars)[FEATURE_COLUMNS].dropna()
        if len(features) < self.sequence_length:
            raise ValueError("latest sequence is incomplete")
        scaled = self.scaler.transform(features.iloc[-self.sequence_length :])
        tensor = (
            torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to(self.device)
        )
        with torch.no_grad():
            return float(torch.sigmoid(self.model(tensor)).item())


class FinBertHeadlineScorer:
    model_id = "prosusai-finbert-frozen"

    def __init__(self) -> None:
        transformers = importlib.import_module("transformers")
        self.pipeline: Any = transformers.pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            device=-1,
        )

    def score(self, headlines: list[str]) -> list[dict[str, float | str]]:
        if not headlines:
            return []
        output = self.pipeline(headlines, batch_size=16)
        return [
            {"label": str(item["label"]).lower(), "score": float(item["score"])} for item in output
        ]


class LocalExperimentTracker:
    def __init__(self, database_path: Path, artifact_root: Path) -> None:
        self.mlflow: Any = importlib.import_module("mlflow")
        database_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
        self.mlflow.set_tracking_uri(f"sqlite:///{database_path}")
        self.artifact_root = artifact_root

    def log(
        self,
        experiment: str,
        parameters: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str],
        artifacts: list[Path] | None = None,
    ) -> str:
        self.mlflow.set_experiment(experiment)
        with self.mlflow.start_run(tags=tags) as run:
            self.mlflow.log_params(parameters)
            self.mlflow.log_metrics(metrics)
            for artifact in artifacts or []:
                self.mlflow.log_artifact(str(artifact))
            return str(run.info.run_id)


def feature_ablation_sets() -> dict[str, list[str]]:
    return {
        "price_only": FEATURE_COLUMNS.copy(),
        "cross_market": [*FEATURE_COLUMNS, "dxy_return", "silver_return", "vix_change"],
        "macro_positioning": [
            *FEATURE_COLUMNS,
            "real_yield_change",
            "cot_managed_money_net",
            "macro_event_distance",
        ],
        "news": [*FEATURE_COLUMNS, "headline_positive", "headline_negative", "headline_count"],
    }


def write_model_card(path: Path, payload: dict[str, Any]) -> None:
    required = {
        "model_id",
        "candidate_hash",
        "dataset_hash",
        "feature_schema",
        "label_horizon",
        "calibration",
        "metrics",
        "limitations",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"model card is missing: {', '.join(sorted(missing))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
