from __future__ import annotations

import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from gold_trader import __version__
from gold_trader.config import Settings, SimulationSettings
from gold_trader.data import file_sha256, resample_bars
from gold_trader.database import Store
from gold_trader.generation import (
    GenerationStrategy,
    TrialSpec,
    _bootstrap_p_value,
    _deflated_sharpe_probability,
    _fold_pnl,
    _probability_of_backtest_overfit,
    _read_dataset,
    resource_health,
)
from gold_trader.research import stable_hash
from gold_trader.simulator import DeterministicSimulator, calculate_metrics
from gold_trader.sleep import SleepLease

GENERATION4_ID = "generation-4"
GENERATION4_SCHEMA = "4.0"
LEARNING_SCHEMA = "generation-3-learning-1"


def current_generation4_implementation_hash() -> str:
    source_root = Path(__file__).resolve().parent
    files = (
        "generation4.py",
        "generation.py",
        "strategies.py",
        "simulator.py",
        "risk.py",
        "features.py",
    )
    return stable_hash(
        {
            "package": __version__,
            "generation_schema": GENERATION4_SCHEMA,
            "source_files": {name: file_sha256(source_root / name) for name in files},
        }
    )


class Generation4Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = GENERATION4_SCHEMA
    generation_id: str = GENERATION4_ID
    status: Literal["draft", "sealed"] = "draft"
    created_at: datetime
    sealed_at: datetime | None = None
    parent_generation_id: str
    parent_manifest_hash: str
    learning_report_path: str
    learning_report_hash: str
    dataset_path: str
    dataset_hash: str
    dataset_rows: int = Field(gt=0)
    dataset_start: datetime
    dataset_end: datetime
    holdout_start: datetime
    seed: int = 20260724
    workers: int = Field(ge=1, le=12)
    trials_per_family: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    families: dict[str, dict[str, list[Any]]]
    benchmark_policy: dict[str, Any]
    settings_snapshot: dict[str, Any]
    implementation_hash: str
    manifest_hash: str = ""
    disclaimer: str = (
        "Research simulation only. Generation 4 is learned from Generation 3 discovery data; "
        "its holdout remains sealed and no result is approved automatically."
    )


def _manifest_digest(payload: dict[str, Any]) -> str:
    hashed = dict(payload)
    hashed["manifest_hash"] = ""
    return stable_hash(hashed)


def _annualized_return(total_return: float, years: float) -> float:
    if years <= 0:
        return 0.0
    if total_return <= -1:
        return -1.0
    return float((1 + total_return) ** (1 / years) - 1)


def buy_hold_benchmark(bars: pd.DataFrame) -> dict[str, float]:
    ordered = bars.sort_values("timestamp").dropna(subset=["close"])
    if len(ordered) < 2:
        return {
            "net_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "years": 0.0,
        }
    first = float(ordered.iloc[0]["close"])
    last = float(ordered.iloc[-1]["close"])
    start = pd.Timestamp(ordered.iloc[0]["timestamp"])
    end = pd.Timestamp(ordered.iloc[-1]["timestamp"])
    years = max((end - start).total_seconds() / (365.2425 * 86400), 1 / 365.2425)
    total_return = last / first - 1
    close = ordered["close"].astype(float)
    max_drawdown = float(-(close / close.cummax() - 1).min())
    annualized = _annualized_return(total_return, years)
    calmar = annualized / max(max_drawdown, 1e-12)
    return {
        "net_return": float(total_return),
        "annualized_return": annualized,
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "years": float(years),
    }


def _strategy_benchmark(metrics: dict[str, float], years: float) -> dict[str, float]:
    annualized = _annualized_return(float(metrics["net_return"]), years)
    drawdown = float(metrics["max_drawdown"])
    return {
        "annualized_return": annualized,
        "calmar": float(annualized / max(drawdown, 1e-12)),
    }


def _ranked_parameter_values(
    trials: list[dict[str, Any]],
    parent_values: list[Any],
    parameter: str,
    cap: int,
) -> list[Any]:
    counts = Counter(item["spec"]["parameters"][parameter] for item in trials)
    ranked = sorted(
        counts,
        key=lambda value: (-counts[value], parent_values.index(value)),
    )[:cap]
    return [value for value in parent_values if value in ranked]


def derive_generation4_spaces(
    parent_manifest: dict[str, Any], shortlisted: list[dict[str, Any]]
) -> dict[str, dict[str, list[Any]]]:
    spaces: dict[str, dict[str, list[Any]]] = {}
    caps = {
        "timeframe": 2,
        "lookback": 5,
        "atr_period": 3,
        "stop_atr": 4,
        "reward_risk": 4,
        "compression_window": 3,
        "compression_ratio": 3,
    }
    for family in ("compression-breakout", "time-series-momentum"):
        family_trials = sorted(
            [item for item in shortlisted if item["spec"]["family"] == family],
            key=lambda item: float(item["result"].get("score", -math.inf)),
            reverse=True,
        )
        if not family_trials:
            continue
        robust = family_trials[: max(10, len(family_trials) // 2)]
        parent_space = parent_manifest["families"][family]
        learned: dict[str, list[Any]] = {}
        for parameter, values in parent_space.items():
            learned[parameter] = _ranked_parameter_values(
                robust, values, parameter, caps.get(parameter, len(values))
            )
        learned["direction"] = ["both", "long-only"]
        learned["regime_window"] = [0, 100, 200]
        spaces[family] = learned
    if not spaces:
        raise ValueError("Generation 3 produced no discovery shortlists to learn from")
    return spaces


def build_generation3_learning_report(
    settings: Settings,
    store: Store,
    parent_generation_id: str,
    output_path: Path,
) -> dict[str, Any]:
    parent = store.get_generation(parent_generation_id)
    if parent is None or parent["status"] != "completed":
        raise ValueError("Generation 3 must be completed before creating its learning report")
    trials = store.list_trials(parent_generation_id, limit=100_000)
    shortlisted = [item for item in trials if item["result"].get("status") == "shortlisted"]
    spaces = derive_generation4_spaces(parent["manifest"], shortlisted)
    dataset = _read_dataset(Path(parent["manifest"]["dataset_path"]))
    timestamps = pd.to_datetime(dataset["timestamp"], utc=True)
    discovery = dataset[timestamps < pd.Timestamp(parent["manifest"]["holdout_start"])].reset_index(
        drop=True
    )
    status_counts = Counter(str(item["result"].get("status")) for item in trials)
    family_counts = Counter(item["spec"]["family"] for item in shortlisted)
    top = sorted(
        shortlisted,
        key=lambda item: float(item["result"].get("score", -math.inf)),
        reverse=True,
    )[:25]
    report: dict[str, Any] = {
        "schema_version": LEARNING_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "parent_generation_id": parent_generation_id,
        "parent_manifest_hash": parent["manifest_hash"],
        "dataset_hash": parent["manifest"]["dataset_hash"],
        "holdout_start": parent["manifest"]["holdout_start"],
        "holdout_opened": False,
        "discovery_only": True,
        "trial_count": len(trials),
        "status_counts": dict(status_counts),
        "shortlist_by_family": dict(family_counts),
        "buy_hold_discovery": buy_hold_benchmark(discovery),
        "learned_spaces": spaces,
        "top_discovery_trials": [
            {
                "trial_hash": item["spec"]["trial_hash"],
                "family": item["spec"]["family"],
                "parameters": item["spec"]["parameters"],
                "score": item["result"].get("score"),
                "metrics": item["result"].get("metrics"),
                "stressed_metrics": item["result"].get("stressed_metrics"),
                "fold_profitable_fraction": item["result"].get("fold_profitable_fraction"),
                "largest_profit_fraction": item["result"].get("largest_profit_fraction"),
            }
            for item in top
        ],
        "lessons": [
            "Only compression breakout and time-series momentum survived discovery gates.",
            "Generation 4 narrows parameters using the robust half of each surviving family.",
            "Long-only and trend-regime variants test whether gold's long-run bias adds value.",
            "A minimum 2% CAGR and buy-and-hold Calmar benchmark reject economically weak edges.",
            "The final 12-month holdout was not used to create this report.",
        ],
        "settings_hash": stable_hash(settings.model_dump(mode="json", exclude={"session_secret"})),
    }
    report["report_hash"] = stable_hash({**report, "report_hash": ""})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def create_generation4_manifest(
    settings: Settings,
    store: Store,
    parent_generation_id: str,
    learning_report_path: Path,
    output_path: Path,
    *,
    generation_id: str = GENERATION4_ID,
    workers: int | None = None,
    trials_per_family: int = 512,
    batch_size: int | None = None,
) -> Generation4Manifest:
    report = build_generation3_learning_report(
        settings, store, parent_generation_id, learning_report_path
    )
    parent = store.get_generation(parent_generation_id)
    assert parent is not None
    parent_manifest = parent["manifest"]
    dataset = Path(parent_manifest["dataset_path"]).resolve()
    bars = _read_dataset(dataset)
    manifest = Generation4Manifest(
        generation_id=generation_id,
        created_at=datetime.now(UTC),
        parent_generation_id=parent_generation_id,
        parent_manifest_hash=parent["manifest_hash"],
        learning_report_path=str(learning_report_path.resolve()),
        learning_report_hash=str(report["report_hash"]),
        dataset_path=str(dataset),
        dataset_hash=file_sha256(dataset),
        dataset_rows=len(bars),
        dataset_start=pd.Timestamp(bars["timestamp"].min()).to_pydatetime().astimezone(UTC),
        dataset_end=datetime.fromisoformat(str(parent_manifest["dataset_end"])).astimezone(UTC),
        holdout_start=datetime.fromisoformat(str(parent_manifest["holdout_start"])).astimezone(UTC),
        workers=workers or settings.generation.workers,
        trials_per_family=trials_per_family,
        batch_size=batch_size or settings.generation.batch_size,
        families=report["learned_spaces"],
        benchmark_policy={
            "benchmark": "unlevered-xauusd-buy-and-hold-price",
            "minimum_annualized_return": 0.02,
            "minimum_calmar_multiple": 1.0,
            "note": (
                "Price benchmark excludes broker financing and is an opportunity-cost diagnostic."
            ),
        },
        settings_snapshot=settings.model_dump(mode="json", exclude={"session_secret"}),
        implementation_hash=current_generation4_implementation_hash(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def seal_generation4_manifest(path: Path, store: Store | None = None) -> Generation4Manifest:
    current = Generation4Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    if current.status == "sealed":
        if current.manifest_hash != _manifest_digest(current.model_dump(mode="json")):
            raise ValueError("sealed Generation 4 manifest checksum does not match")
        return current
    dataset = Path(current.dataset_path)
    learning = Path(current.learning_report_path)
    if not dataset.is_file() or file_sha256(dataset) != current.dataset_hash:
        raise ValueError("Generation 4 dataset changed before sealing")
    if not learning.is_file():
        raise ValueError("Generation 4 learning report is missing")
    report = json.loads(learning.read_text(encoding="utf-8"))
    actual_report_hash = stable_hash({**report, "report_hash": ""})
    if (
        report.get("report_hash") != current.learning_report_hash
        or actual_report_hash != current.learning_report_hash
    ):
        raise ValueError("Generation 4 learning report changed before sealing")
    sealed_payload = current.model_dump(mode="json")
    sealed_payload.update({"status": "sealed", "sealed_at": datetime.now(UTC)})
    normalized = Generation4Manifest.model_validate(sealed_payload).model_dump(mode="json")
    normalized["manifest_hash"] = _manifest_digest(normalized)
    sealed = Generation4Manifest.model_validate(normalized)
    path.write_text(sealed.model_dump_json(indent=2), encoding="utf-8")
    if store is not None:
        store.save_generation(
            sealed.generation_id,
            sealed.manifest_hash,
            "sealed",
            sealed.model_dump(mode="json"),
            {"completed_trials": 0, "total_trials": total_generation4_trials(sealed)},
        )
    return sealed


def load_generation4_manifest(path: Path) -> Generation4Manifest:
    manifest = Generation4Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.status != "sealed" or not manifest.manifest_hash:
        raise ValueError("Generation 4 manifest must be sealed")
    if manifest.manifest_hash != _manifest_digest(manifest.model_dump(mode="json")):
        raise ValueError("sealed Generation 4 manifest was modified")
    if manifest.implementation_hash != current_generation4_implementation_hash():
        raise ValueError("Generation 4 implementation changed; create a new lineage")
    if file_sha256(Path(manifest.dataset_path)) != manifest.dataset_hash:
        raise ValueError("sealed Generation 4 dataset no longer matches")
    report = json.loads(Path(manifest.learning_report_path).read_text(encoding="utf-8"))
    actual_report_hash = stable_hash({**report, "report_hash": ""})
    if (
        report.get("report_hash") != manifest.learning_report_hash
        or actual_report_hash != manifest.learning_report_hash
    ):
        raise ValueError("sealed Generation 4 learning report no longer matches")
    return manifest


def total_generation4_trials(manifest: Generation4Manifest) -> int:
    return sum(
        min(
            manifest.trials_per_family,
            reduce(mul, (len(values) for values in space.values()), 1),
        )
        for space in manifest.families.values()
    )


def _decode_space(space: dict[str, list[Any]], index: int) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, choices in reversed(list(space.items())):
        values[key] = choices[index % len(choices)]
        index //= len(choices)
    return {key: values[key] for key in space}


def generation4_trial_specs(manifest: Generation4Manifest) -> list[TrialSpec]:
    prepared: list[dict[str, Any]] = []
    for family, space in manifest.families.items():
        size = reduce(mul, (len(values) for values in space.values()), 1)
        multiplier = 104_729
        while math.gcd(multiplier, size) != 1:
            multiplier += 2
        prepared.append(
            {
                "family": family,
                "space": space,
                "size": size,
                "limit": min(manifest.trials_per_family, size),
                "multiplier": multiplier,
                "offset": int(stable_hash({"seed": manifest.seed, "family": family})[:12], 16),
            }
        )
    specs: list[TrialSpec] = []
    ordinal = 0
    total = total_generation4_trials(manifest)
    for family_ordinal in range(max(int(item["limit"]) for item in prepared)):
        for item in prepared:
            if family_ordinal >= int(item["limit"]):
                continue
            index = (int(item["offset"]) + family_ordinal * int(item["multiplier"])) % int(
                item["size"]
            )
            parameters = _decode_space(item["space"], index)
            base = {
                "generation_id": manifest.generation_id,
                "manifest_hash": manifest.manifest_hash,
                "ordinal": ordinal,
                "search_trial_count": total,
                "family": item["family"],
                "parameters": parameters,
            }
            specs.append(TrialSpec(**base, trial_hash=stable_hash(base)))
            ordinal += 1
    return specs


class Generation4Strategy(GenerationStrategy):
    def __init__(self, family: str, parameters: dict[str, Any], trial_hash: str) -> None:
        self.generation4_parameters = dict(parameters)
        super().__init__(family, parameters, trial_hash)
        self.variant_id = f"g4-{trial_hash[:16]}"
        self.warmup = max(self.warmup, int(parameters.get("regime_window", 0)) + 5)

    @property
    def strategy_hash(self) -> str:
        return stable_hash(
            {
                "generation": GENERATION4_SCHEMA,
                "family": self.strategy_id,
                "timeframe": self.timeframe,
                "parameters": self.generation4_parameters,
                "implementation": current_generation4_implementation_hash(),
            }
        )

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        result = super().precompute(bars)
        if self.generation4_parameters.get("direction") == "long-only":
            result.loc[result["side"].eq("sell"), ["side", "reason"]] = [None, None]
        regime_window = int(self.generation4_parameters.get("regime_window", 0))
        if regime_window:
            close = bars["close"].astype(float)
            regime = close.ewm(span=regime_window, adjust=False).mean()
            invalid_buy = result["side"].eq("buy") & (close <= regime)
            invalid_sell = result["side"].eq("sell") & (close >= regime)
            result.loc[invalid_buy | invalid_sell, ["side", "reason"]] = [None, None]
        result["reason"] = result["reason"].str.replace("Generation 3", "Generation 4", regex=False)
        return result


def generation4_strategy_from_spec(spec: dict[str, Any]) -> Generation4Strategy:
    return Generation4Strategy(
        str(spec["family"]), dict(spec["parameters"]), str(spec["trial_hash"])
    )


_WORKER_DATA: pd.DataFrame | None = None
_WORKER_SETTINGS: Settings | None = None
_WORKER_HOLDOUT_START: datetime | None = None
_WORKER_RESAMPLED: dict[str, pd.DataFrame] = {}


def _initialize_worker(dataset_path: str, settings: dict[str, Any], holdout_start: str) -> None:
    global _WORKER_DATA, _WORKER_SETTINGS, _WORKER_HOLDOUT_START, _WORKER_RESAMPLED
    _WORKER_DATA = _read_dataset(Path(dataset_path))
    _WORKER_SETTINGS = Settings.model_validate(settings)
    _WORKER_HOLDOUT_START = datetime.fromisoformat(holdout_start).astimezone(UTC)
    _WORKER_RESAMPLED = {}


def _gate_detail(
    key: str, label: str, passed: bool, actual: str, requirement: str
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "passed": bool(passed),
        "actual": actual,
        "requirement": requirement,
        "reason": f"{label} was {actual}; required {requirement}.",
    }


def explain_generation4_result(result: dict[str, Any]) -> dict[str, Any]:
    details = list(result.get("gate_details") or [])
    failed = [item["reason"] for item in details if not item["passed"]]
    if failed:
        prefix = "Pruned" if result.get("status") == "pruned" else "Rejected"
        summary = f"{prefix} because " + " ".join(failed)
    else:
        summary = (
            "Passed every Generation 4 discovery and benchmark gate. Final validation and "
            "operator approval still apply."
        )
    return {"summary": summary, "failed_reasons": failed, "gates": details}


def _evaluate_trial(spec_payload: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_DATA is None or _WORKER_SETTINGS is None or _WORKER_HOLDOUT_START is None:
        raise RuntimeError("Generation 4 worker was not initialized")
    spec = TrialSpec.model_validate(spec_payload)
    strategy = generation4_strategy_from_spec(spec.model_dump(mode="json"))
    if strategy.timeframe not in _WORKER_RESAMPLED:
        _WORKER_RESAMPLED[strategy.timeframe] = resample_bars(_WORKER_DATA, strategy.timeframe)
    all_bars = _WORKER_RESAMPLED[strategy.timeframe]
    timestamps = pd.to_datetime(all_bars["timestamp"], utc=True)
    discovery = all_bars[timestamps < _WORKER_HOLDOUT_START].reset_index(drop=True)
    intraday = strategy.timeframe in {"5min", "15min"}
    recent_start = pd.Timestamp(_WORKER_HOLDOUT_START) - pd.DateOffset(
        months=24 if intraday else 60
    )
    recent_indices = discovery.index[
        pd.to_datetime(discovery["timestamp"], utc=True) >= recent_start
    ]
    context_start = (
        max(int(recent_indices.min()) - strategy.warmup, 0) if len(recent_indices) else 0
    )
    recent = discovery.iloc[context_start:].reset_index(drop=True)
    simulator = DeterministicSimulator(
        _WORKER_SETTINGS.risk,
        _WORKER_SETTINGS.simulation,
        _WORKER_SETTINGS.starting_equity,
    )
    _recent_trades, recent_metrics = simulator.run(recent, strategy)
    minimum_recent_trades = 20 if intraday else 8
    stage_one_details = [
        _gate_detail(
            "recent_trade_count",
            "Recent trade count",
            recent_metrics["trade_count"] >= minimum_recent_trades,
            str(int(recent_metrics["trade_count"])),
            f">= {minimum_recent_trades}",
        ),
        _gate_detail(
            "recent_return",
            "Recent net return",
            recent_metrics["net_return"] > -0.02,
            f"{recent_metrics['net_return'] * 100:.2f}%",
            "> -2.00%",
        ),
        _gate_detail(
            "recent_drawdown",
            "Recent max drawdown",
            recent_metrics["max_drawdown"] <= 0.10,
            f"{recent_metrics['max_drawdown'] * 100:.2f}%",
            "<= 10.00%",
        ),
    ]
    if not all(item["passed"] for item in stage_one_details):
        return {
            "status": "pruned",
            "stage": 1,
            "score": float(recent_metrics["net_return"] - recent_metrics["max_drawdown"]),
            "metrics": recent_metrics,
            "gates": {item["key"]: item["passed"] for item in stage_one_details},
            "gate_details": stage_one_details,
            "fold_pnl": [],
        }

    trades, metrics = simulator.run(discovery, strategy)
    stressed_settings = _WORKER_SETTINGS.simulation.model_copy(
        update={
            "spread_points": _WORKER_SETTINGS.simulation.spread_points * 2,
            "slippage_points": _WORKER_SETTINGS.simulation.slippage_points * 2,
            "commission_per_lot_round_trip": (
                _WORKER_SETTINGS.simulation.commission_per_lot_round_trip * 2
            ),
        }
    )
    stressed_trades, stressed_metrics = DeterministicSimulator(
        _WORKER_SETTINGS.risk, stressed_settings, _WORKER_SETTINGS.starting_equity
    ).run(discovery, strategy)
    folds = _fold_pnl(discovery, strategy, _WORKER_SETTINGS)
    profitable_fraction = float(np.mean(np.asarray(folds) > 0)) if folds else 0.0
    positive_folds = [value for value in folds if value > 0]
    concentration = max(positive_folds) / sum(positive_folds) if positive_folds else 1.0
    benchmark = buy_hold_benchmark(discovery)
    strategy_benchmark = _strategy_benchmark(metrics, benchmark["years"])
    minimum_trades = 80 if intraday else 30
    gate_details = [
        _gate_detail(
            "positive_return",
            "Base net return",
            metrics["net_return"] > 0,
            f"{metrics['net_return'] * 100:.2f}%",
            "> 0.00%",
        ),
        _gate_detail(
            "positive_stressed_return",
            "Doubled-cost net return",
            stressed_metrics["net_return"] > 0,
            f"{stressed_metrics['net_return'] * 100:.2f}%",
            "> 0.00%",
        ),
        _gate_detail(
            "profit_factor",
            "Profit factor",
            metrics["profit_factor"] >= 1.10,
            f"{metrics['profit_factor']:.2f}",
            ">= 1.10",
        ),
        _gate_detail(
            "sharpe",
            "Sharpe ratio",
            metrics["sharpe"] >= 0.50,
            f"{metrics['sharpe']:.2f}",
            ">= 0.50",
        ),
        _gate_detail(
            "drawdown",
            "Max drawdown",
            metrics["max_drawdown"] <= 0.10,
            f"{metrics['max_drawdown'] * 100:.2f}%",
            "<= 10.00%",
        ),
        _gate_detail(
            "trade_count",
            "Trade count",
            metrics["trade_count"] >= minimum_trades,
            str(int(metrics["trade_count"])),
            f">= {minimum_trades}",
        ),
        _gate_detail(
            "fold_stability",
            "Profitable folds",
            profitable_fraction >= 0.60,
            f"{profitable_fraction * 100:.2f}%",
            ">= 60.00%",
        ),
        _gate_detail(
            "profit_concentration",
            "Largest fold profit share",
            concentration <= 0.50,
            f"{concentration * 100:.2f}%",
            "<= 50.00%",
        ),
        _gate_detail(
            "economic_return",
            "Annualized return",
            strategy_benchmark["annualized_return"] >= 0.02,
            f"{strategy_benchmark['annualized_return'] * 100:.2f}%",
            ">= 2.00%",
        ),
        _gate_detail(
            "risk_adjusted_benchmark",
            "Calmar versus buy-and-hold",
            strategy_benchmark["calmar"] >= benchmark["calmar"],
            f"{strategy_benchmark['calmar']:.2f} vs {benchmark['calmar']:.2f}",
            ">= buy-and-hold",
        ),
    ]
    gates = {item["key"]: item["passed"] for item in gate_details}
    trade_pnl = [float(trade.pnl) for trade in trades]
    selection_statistics = {
        "deflated_sharpe_probability": _deflated_sharpe_probability(
            float(metrics["sharpe"]), int(metrics["trade_count"]), spec.search_trial_count
        ),
        "stationary_bootstrap_p_value": _bootstrap_p_value(trade_pnl, int(spec.trial_hash[:8], 16)),
    }
    score = (
        float(metrics["net_return"])
        + float(stressed_metrics["net_return"]) * 0.75
        + min(float(metrics["sharpe"]), 3.0) * 0.02
        - float(metrics["max_drawdown"]) * 1.5
        + min(float(strategy_benchmark["calmar"]), 3.0) * 0.02
    )
    return {
        "status": "shortlisted" if all(gates.values()) else "rejected",
        "stage": 2,
        "score": score,
        "metrics": metrics,
        "stressed_metrics": stressed_metrics,
        "gates": gates,
        "gate_details": gate_details,
        "benchmark": benchmark,
        "strategy_benchmark": strategy_benchmark,
        "fold_pnl": folds,
        "fold_profitable_fraction": profitable_fraction,
        "largest_profit_fraction": concentration,
        "stressed_trade_count": len(stressed_trades),
        "selection_statistics": selection_statistics,
    }


class Generation4Coordinator:
    """Single-writer Generation 4 coordinator with durable batch checkpoints."""

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.sleep_lease = SleepLease()

    def request_control(self, action: Literal["run", "pause", "stop"]) -> None:
        self.store.set_state(
            "generation4_control",
            {"action": action, "requested_at": datetime.now(UTC).isoformat()},
        )

    def run(self, manifest_path: Path) -> dict[str, Any]:
        manifest = load_generation4_manifest(manifest_path)
        health = resource_health(self.settings, manifest.workers)
        if not health["healthy"]:
            paused_summary = {
                "manifest_path": str(manifest_path.resolve()),
                "completed_trials": len(self.store.list_trials(manifest.generation_id)),
                "total_trials": total_generation4_trials(manifest),
                "shortlisted": 0,
                "resource_health": health,
                "pause_reason": "connect AC power or resolve the failed local resource check",
                "updated_at": datetime.now(UTC).isoformat(),
                "sleep_lease": False,
                "parent_manifest_hash": manifest.parent_manifest_hash,
                "learning_report_hash": manifest.learning_report_hash,
            }
            self.store.save_generation(
                manifest.generation_id,
                manifest.manifest_hash,
                "resource-paused",
                manifest.model_dump(mode="json"),
                paused_summary,
            )
            return {"status": "resource-paused", **paused_summary}

        self.request_control("run")
        self.sleep_lease.acquire()
        completed = {
            item["spec"]["trial_hash"] for item in self.store.list_trials(manifest.generation_id)
        }
        pending = [
            spec for spec in generation4_trial_specs(manifest) if spec.trial_hash not in completed
        ]
        total = len(completed) + len(pending)
        existing = self.store.list_trials(manifest.generation_id)
        summary: dict[str, Any] = {
            "manifest_path": str(manifest_path.resolve()),
            "completed_trials": len(completed),
            "total_trials": total,
            "shortlisted": sum(item["result"].get("status") == "shortlisted" for item in existing),
            "best_score": max(
                (float(item["result"].get("score", -math.inf)) for item in existing),
                default=None,
            ),
            "resource_health": health,
            "started_at": datetime.now(UTC).isoformat(),
            "sleep_lease": self.sleep_lease.active,
            "parent_manifest_hash": manifest.parent_manifest_hash,
            "learning_report_hash": manifest.learning_report_hash,
        }
        self.store.save_generation(
            manifest.generation_id,
            manifest.manifest_hash,
            "running",
            manifest.model_dump(mode="json"),
            summary,
        )
        no_improvement = 0
        batches = 0
        status = "completed"
        initializer_args = (
            manifest.dataset_path,
            manifest.settings_snapshot,
            manifest.holdout_start.isoformat(),
        )
        executor: ProcessPoolExecutor | None = None
        try:
            if manifest.workers > 1:
                executor = ProcessPoolExecutor(
                    max_workers=manifest.workers,
                    initializer=_initialize_worker,
                    initargs=initializer_args,
                )
            else:
                _initialize_worker(*initializer_args)
            for cursor in range(0, len(pending), manifest.batch_size):
                control = self.store.get_state("generation4_control") or {"action": "run"}
                if control.get("action") in {"pause", "stop"}:
                    status = "paused" if control["action"] == "pause" else "stopped"
                    break
                health = resource_health(self.settings, manifest.workers)
                if not health["healthy"]:
                    status = "resource-paused"
                    summary["resource_health"] = health
                    break
                batch = pending[cursor : cursor + manifest.batch_size]
                payloads = [spec.model_dump(mode="json") for spec in batch]
                results = (
                    [_evaluate_trial(payload) for payload in payloads]
                    if executor is None
                    else list(executor.map(_evaluate_trial, payloads))
                )
                improved = False
                for spec, result in zip(batch, results, strict=True):
                    result["completed_at"] = datetime.now(UTC).isoformat()
                    self.store.save_trial(
                        manifest.generation_id, spec.model_dump(mode="json"), result
                    )
                    score = float(result.get("score", -math.inf))
                    if summary["best_score"] is None or score > float(str(summary["best_score"])):
                        summary["best_score"] = score
                        summary["best_trial_hash"] = spec.trial_hash
                        improved = True
                batches += 1
                no_improvement = 0 if improved else no_improvement + 1
                trials = self.store.list_trials(manifest.generation_id)
                summary.update(
                    {
                        "completed_trials": len(trials),
                        "shortlisted": sum(
                            item["result"].get("status") == "shortlisted" for item in trials
                        ),
                        "batches": batches,
                        "no_improvement_batches": no_improvement,
                        "probability_of_backtest_overfit": _probability_of_backtest_overfit(trials),
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                )
                self.store.save_generation(
                    manifest.generation_id,
                    manifest.manifest_hash,
                    "running",
                    manifest.model_dump(mode="json"),
                    summary,
                )
                if (
                    batches >= self.settings.generation.minimum_batches_before_futility
                    and no_improvement >= self.settings.generation.no_improvement_batches
                    and summary["shortlisted"] == 0
                ):
                    status = "futile"
                    break
        except Exception:
            status = "failed"
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            self.sleep_lease.release()
            summary["finished_at"] = datetime.now(UTC).isoformat()
            summary["sleep_lease"] = False
            self.store.save_generation(
                manifest.generation_id,
                manifest.manifest_hash,
                status,
                manifest.model_dump(mode="json"),
                summary,
            )
            report_path = self.settings.project_root / "var/reports/generation-4-summary.json"
            report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return {"status": status, **summary}

    def finalize(
        self, manifest_path: Path, trial_hash: str, external_dataset: Path
    ) -> dict[str, Any]:
        manifest = load_generation4_manifest(manifest_path)
        selected = next(
            (
                item
                for item in self.store.list_trials(manifest.generation_id)
                if item["spec"]["trial_hash"] == trial_hash
            ),
            None,
        )
        if selected is None or selected["result"].get("status") != "shortlisted":
            raise ValueError("only a completed Generation 4 shortlist can be finalized")
        holdout_key = f"generation_holdout:{trial_hash}"
        prior_open = self.store.get_state(holdout_key)
        if prior_open is not None:
            candidate_hash = prior_open.get("candidate_hash")
            existing = self.store.get_research_run(str(candidate_hash)) if candidate_hash else None
            if existing is not None:
                return existing
            raise ValueError("this Generation 4 lineage already opened the final holdout")

        spec = selected["spec"]
        strategy = generation4_strategy_from_spec(spec)
        primary = _read_dataset(Path(manifest.dataset_path))
        primary_bars = resample_bars(primary, strategy.timeframe)
        timestamps = pd.to_datetime(primary_bars["timestamp"], utc=True)
        holdout = primary_bars[timestamps >= manifest.holdout_start].reset_index(drop=True)
        external = external_dataset.resolve()
        if not external.is_file() or external == Path(manifest.dataset_path).resolve():
            raise ValueError("an independently sourced external dataset is required")
        external_hash = file_sha256(external)
        if external_hash == manifest.dataset_hash:
            raise ValueError("external validation data must not duplicate the primary dataset")
        external_bars = resample_bars(_read_dataset(external), strategy.timeframe)
        external_timestamps = pd.to_datetime(external_bars["timestamp"], utc=True)
        external_holdout = external_bars[
            (external_timestamps >= manifest.holdout_start)
            & (external_timestamps <= manifest.dataset_end)
        ].reset_index(drop=True)
        if len(external_holdout) < strategy.warmup + 10:
            raise ValueError("external dataset does not cover enough of the sealed holdout")

        self.store.set_state(
            holdout_key,
            {
                "opened_at": datetime.now(UTC).isoformat(),
                "manifest_hash": manifest.manifest_hash,
                "external_dataset_hash": external_hash,
            },
        )
        simulator = DeterministicSimulator(
            self.settings.risk, self.settings.simulation, self.settings.starting_equity
        )
        trades, metrics = simulator.run(holdout, strategy)
        stressed_settings: SimulationSettings = self.settings.simulation.model_copy(
            update={
                "spread_points": self.settings.simulation.spread_points * 2,
                "slippage_points": self.settings.simulation.slippage_points * 2,
                "commission_per_lot_round_trip": (
                    self.settings.simulation.commission_per_lot_round_trip * 2
                ),
            }
        )
        stressed_trades, stressed = DeterministicSimulator(
            self.settings.risk, stressed_settings, self.settings.starting_equity
        ).run(holdout, strategy)
        external_trades, external_metrics = simulator.run(external_holdout, strategy)
        if not external_trades:
            external_metrics = calculate_metrics([], self.settings.starting_equity)
        intraday = strategy.timeframe in {"5min", "15min"}
        minimum_trades = (
            self.settings.research.intraday_min_trades
            if intraday
            else self.settings.research.swing_min_trades
        )
        selection = selected["result"].get("selection_statistics", {})
        benchmark = buy_hold_benchmark(holdout)
        strategy_benchmark = _strategy_benchmark(metrics, benchmark["years"])
        gates = {
            "positive_holdout_return": metrics["net_return"] > 0,
            "positive_stressed_return": stressed["net_return"] > 0,
            "profit_factor": metrics["profit_factor"] >= self.settings.research.min_profit_factor,
            "sharpe": metrics["sharpe"] >= self.settings.research.min_sharpe,
            "drawdown": metrics["max_drawdown"] <= self.settings.research.max_drawdown,
            "trade_count": metrics["trade_count"] >= minimum_trades,
            "fold_stability": selected["result"].get("fold_profitable_fraction", 0)
            >= self.settings.research.min_profitable_fold_fraction,
            "profit_concentration": selected["result"].get("largest_profit_fraction", 1) <= 0.5,
            "selection_bias": selection.get("deflated_sharpe_probability", 0) >= 0.50,
            "external_validation": bool(
                external_metrics["net_return"] > 0 and external_metrics["profit_factor"] >= 1.05
            ),
            "risk_adjusted_benchmark": strategy_benchmark["calmar"] >= benchmark["calmar"],
        }
        eligible = all(gates.values())
        configuration_hash = stable_hash(
            self.settings.model_dump(mode="json", exclude={"session_secret"})
        )
        model_descriptor = {"model": "rule-only", "version": 1}
        model_hash = stable_hash(model_descriptor)
        candidate_hash = stable_hash(
            {
                "manifest": manifest.manifest_hash,
                "trial": trial_hash,
                "dataset": manifest.dataset_hash,
                "external_dataset": external_hash,
                "configuration": configuration_hash,
                "strategy": strategy.strategy_hash,
                "model": model_hash,
                "implementation": manifest.implementation_hash,
            }
        )
        report = {
            "run_id": str(uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
            "strategy_id": strategy.strategy_id,
            "variant_id": strategy.variant_id,
            "timeframe": strategy.timeframe,
            "candidate_hash": candidate_hash,
            "dataset_hash": manifest.dataset_hash,
            "configuration_hash": configuration_hash,
            "strategy_hash": strategy.strategy_hash,
            "model_hash": model_hash,
            "model": model_descriptor,
            "implementation_hash": manifest.implementation_hash,
            "status": "eligible" if eligible else "ineligible",
            "eligible": eligible,
            "metrics": metrics,
            "stressed_metrics": stressed,
            "external_metrics": external_metrics,
            "benchmark": benchmark,
            "strategy_benchmark": strategy_benchmark,
            "gates": gates,
            "data": {
                "dataset_path": manifest.dataset_path,
                "holdout_start": manifest.holdout_start.isoformat(),
                "external_dataset_path": str(external),
            },
            "generation_four": {
                "manifest_hash": manifest.manifest_hash,
                "parent_manifest_hash": manifest.parent_manifest_hash,
                "learning_report_hash": manifest.learning_report_hash,
                "trial_spec": spec,
                "discovery_result": selected["result"],
                "holdout_opened_once": True,
            },
            "trades": [trade.as_dict() for trade in trades],
            "stressed_trade_count": len(stressed_trades),
            "disclaimer": "Simulation only. Finalization does not approve or start paper trading.",
        }
        existing = self.store.get_research_run(candidate_hash)
        if existing is not None:
            return existing
        self.store.save_research_run(report)
        self.store.set_state(
            holdout_key,
            {
                "opened_at": datetime.now(UTC).isoformat(),
                "manifest_hash": manifest.manifest_hash,
                "external_dataset_hash": external_hash,
                "candidate_hash": candidate_hash,
                "status": "complete",
            },
        )
        output = self.settings.project_root / "var/reports" / f"{candidate_hash}.json"
        output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return report


class Generation4Worker:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.running = True

    def run_forever(self, poll_seconds: float = 0.5) -> None:
        import signal
        import time

        def stop(*_args: Any) -> None:
            self.running = False
            Generation4Coordinator(self.settings, self.store).request_control("stop")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        self.store.recover_processing_commands("generation4.")
        while self.running:
            command = self.store.claim_next_command("generation4.")
            if command is None:
                time.sleep(poll_seconds)
                continue
            try:
                if command["type"] != "generation4.run":
                    raise ValueError(f"unsupported Generation 4 command: {command['type']}")
                payload = command["payload"]
                manifest = load_generation4_manifest(Path(payload["manifest_path"]))
                if payload.get("generation_id") != manifest.generation_id:
                    raise ValueError("queued Generation 4 ID does not match")
                if payload.get("manifest_hash") != manifest.manifest_hash:
                    raise ValueError("queued Generation 4 manifest hash does not match")
                result = Generation4Coordinator(self.settings, self.store).run(
                    Path(payload["manifest_path"])
                )
                self.store.complete_command(command["id"], result)
            except Exception as exc:
                self.store.complete_command(
                    command["id"],
                    {"error": str(exc), "type": type(exc).__name__},
                    failed=True,
                )


def generation4_worker_main() -> None:
    from gold_trader.config import get_settings

    settings = get_settings()
    store = Store(settings.database_url)
    store.initialize()
    Generation4Worker(settings, store).run_forever()
