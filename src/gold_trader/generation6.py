from __future__ import annotations

import json
import math
import signal
import time
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
from gold_trader.domain import Side
from gold_trader.features import atr, rsi
from gold_trader.generation import (
    TrialSpec,
    _bootstrap_p_value,
    _deflated_sharpe_probability,
    _probability_of_backtest_overfit,
    _read_dataset,
    resource_health,
)
from gold_trader.generation4 import buy_hold_benchmark
from gold_trader.generation5 import (
    Generation5Simulator,
    _annual_trade_counts,
    _gate_detail,
    _holding_statistics,
    _monthly_trade_counts,
    _strategy_benchmark,
    normalize_xau_session,
)
from gold_trader.research import stable_hash
from gold_trader.simulator import calculate_metrics
from gold_trader.sleep import SleepLease
from gold_trader.strategies import BaseStrategy, StrategyParameters
from gold_trader.worker import PaperWorker

GENERATION6_ID = "generation-6"
GENERATION6_SCHEMA = "6.0"
LEARNING_SCHEMA = "generation-6-hypothesis-learning-1"


def current_generation6_implementation_hash() -> str:
    source_root = Path(__file__).resolve().parent
    files = (
        "generation6.py",
        "generation5.py",
        "generation.py",
        "strategies.py",
        "simulator.py",
        "risk.py",
        "features.py",
    )
    return stable_hash(
        {
            "package": __version__,
            "generation_schema": GENERATION6_SCHEMA,
            "source_files": {name: file_sha256(source_root / name) for name in files},
        }
    )


class Generation6Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = GENERATION6_SCHEMA
    generation_id: str = GENERATION6_ID
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
    seed: int = 20260726
    workers: int = Field(ge=1, le=12)
    trials_per_family: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    families: dict[str, dict[str, list[Any]]]
    activity_policy: dict[str, Any]
    session_policy: dict[str, Any]
    settings_snapshot: dict[str, Any]
    implementation_hash: str
    manifest_hash: str = ""
    disclaimer: str = (
        "Simulation research only. Generation 6 tests new high-activity hypotheses after "
        "observing prior lineage results. It is not independent confirmation, and profitability "
        "is neither assumed nor promised."
    )


def generation6_spaces() -> dict[str, dict[str, list[Any]]]:
    common: dict[str, list[Any]] = {
        "timeframe": ["15min", "1h"],
        "atr_period": [10, 14],
        "stop_atr": [1.0, 1.25, 1.5, 2.0],
        "reward_risk": [1.0, 1.25, 1.5, 2.0],
        "max_holding_bars": [8, 16, 24],
        "direction": ["both", "long-only"],
    }
    return {
        "trend-state": {
            **common,
            "fast_window": [8, 12, 20],
            "slow_window": [35, 55, 80],
            "slope_window": [3, 6],
            "minimum_slope_atr": [0.0, 0.25, 0.5],
        },
        "momentum-consensus": {
            **common,
            "fast_lookback": [3, 6, 12],
            "slow_lookback": [20, 35, 55],
            "minimum_move_atr": [0.5, 1.0, 1.5],
        },
        "rsi-recovery": {
            **common,
            "rsi_period": [7, 10, 14],
            "rsi_level": [30, 35, 40],
            "regime_window": [0, 50, 100],
        },
        "session-trend": {
            "timeframe": ["15min"],
            "atr_period": [10, 14],
            "stop_atr": [1.0, 1.25, 1.5, 2.0],
            "reward_risk": [1.0, 1.25, 1.5, 2.0],
            "max_holding_bars": [4, 8, 12, 16],
            "direction": ["both", "long-only"],
            "fast_window": [8, 12, 20],
            "slow_window": [35, 55, 80],
            "session_window": ["6-12", "7-16", "8-18", "12-20"],
        },
    }


def build_generation6_learning_report(
    settings: Settings,
    store: Store,
    parent_generation_id: str,
    output_path: Path,
) -> dict[str, Any]:
    parent = store.get_generation(parent_generation_id)
    if parent is None or parent["status"] not in {"completed", "futile", "stopped"}:
        raise ValueError("Generation 5 must reach a terminal result before creating Generation 6")
    trials = store.list_trials(parent_generation_id)
    family_counts: dict[str, Counter[str]] = {}
    for item in trials:
        family = str(item["spec"]["family"])
        counter = family_counts.setdefault(family, Counter())
        counter[str(item["result"].get("status", "unknown"))] += 1
        for gate in item["result"].get("gate_details", []):
            if not gate.get("passed"):
                counter[f"failed:{gate['key']}"] += 1
    ranked = sorted(
        trials,
        key=lambda item: float(item["result"].get("score", -math.inf)),
        reverse=True,
    )[:20]
    report: dict[str, Any] = {
        "schema_version": LEARNING_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "parent_generation_id": parent_generation_id,
        "parent_manifest_hash": parent["manifest_hash"],
        "parent_status": parent["status"],
        "dataset_hash": parent["manifest"]["dataset_hash"],
        "holdout_start": parent["manifest"]["holdout_start"],
        "adaptive_period_reuse": True,
        "parent_trial_count": len(trials),
        "parent_family_outcomes": {
            family: dict(counts) for family, counts in family_counts.items()
        },
        "top_parent_trials": [
            {
                "trial_hash": item["spec"]["trial_hash"],
                "family": item["spec"]["family"],
                "parameters": item["spec"]["parameters"],
                "status": item["result"].get("status"),
                "score": item["result"].get("score"),
                "metrics": item["result"].get("metrics"),
                "failed_gates": [
                    gate["key"]
                    for gate in item["result"].get("gate_details", [])
                    if not gate.get("passed")
                ],
            }
            for item in ranked
        ],
        "fixed_spaces": generation6_spaces(),
        "lessons": [
            "Generation 5 found profitable compression variants but not enough recent trades.",
            "Generation 6 replaces rare event-only entries with state and recovery hypotheses.",
            (
                "Maximum holding periods remain explicit so one position cannot suppress "
                "later signals."
            ),
            "Persistent conditions may re-enter only after the previous simulated position closes.",
            "Higher activity must remain profitable under doubled modeled transaction costs.",
        ],
        "settings_hash": stable_hash(settings.model_dump(mode="json", exclude={"session_secret"})),
        "disclaimer": (
            "Generation 6 is adaptively designed from earlier results and is not a fresh blind "
            "test. A shortlist must still pass one-time final and independent-source validation."
        ),
    }
    report["report_hash"] = stable_hash({**report, "report_hash": ""})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def create_generation6_manifest(
    settings: Settings,
    store: Store,
    parent_generation_id: str,
    learning_report_path: Path,
    output_path: Path,
    *,
    generation_id: str = GENERATION6_ID,
    workers: int | None = None,
    trials_per_family: int = 384,
    batch_size: int | None = None,
) -> Generation6Manifest:
    report = build_generation6_learning_report(
        settings, store, parent_generation_id, learning_report_path
    )
    parent = store.get_generation(parent_generation_id)
    assert parent is not None
    source = parent["manifest"]
    dataset = Path(str(source["dataset_path"])).resolve()
    bars = _read_dataset(dataset)
    manifest = Generation6Manifest(
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
        dataset_end=datetime.fromisoformat(str(source["dataset_end"])).astimezone(UTC),
        holdout_start=datetime.fromisoformat(str(source["holdout_start"])).astimezone(UTC),
        workers=workers or settings.generation.workers,
        trials_per_family=trials_per_family,
        batch_size=batch_size or settings.generation.batch_size,
        families=generation6_spaces(),
        activity_policy={
            "minimum_recent_24_month_trades": 80,
            "minimum_discovery_trades": 400,
            "minimum_each_recent_full_year": 45,
            "minimum_median_recent_full_year": 55,
            "final_holdout_trades": 50,
            "final_active_months": 10,
            "minimum_trades_per_active_month": 3,
        },
        session_policy={
            "name": "utc-sunday-22-through-friday-22",
            "saturday": "excluded",
            "sunday": "22:00 UTC onward",
            "friday": "before 22:00 UTC",
            "applies_to": "primary, independent validation, replay, and live paper",
        },
        settings_snapshot=settings.model_dump(mode="json", exclude={"session_secret"}),
        implementation_hash=current_generation6_implementation_hash(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def _manifest_digest(payload: dict[str, Any]) -> str:
    selected = dict(payload)
    selected["manifest_hash"] = ""
    return stable_hash(selected)


def seal_generation6_manifest(path: Path, store: Store | None = None) -> Generation6Manifest:
    current = Generation6Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    if current.status == "sealed":
        if current.manifest_hash != _manifest_digest(current.model_dump(mode="json")):
            raise ValueError("sealed Generation 6 manifest checksum does not match")
        return current
    if file_sha256(Path(current.dataset_path)) != current.dataset_hash:
        raise ValueError("Generation 6 dataset changed before sealing")
    learning_path = Path(current.learning_report_path)
    if not learning_path.is_file():
        raise ValueError("Generation 6 learning report is missing")
    report = json.loads(learning_path.read_text(encoding="utf-8"))
    actual_hash = stable_hash({**report, "report_hash": ""})
    if (
        report.get("report_hash") != current.learning_report_hash
        or actual_hash != current.learning_report_hash
    ):
        raise ValueError("Generation 6 learning report changed before sealing")
    payload = current.model_dump(mode="json")
    payload.update({"status": "sealed", "sealed_at": datetime.now(UTC)})
    normalized = Generation6Manifest.model_validate(payload).model_dump(mode="json")
    normalized["manifest_hash"] = _manifest_digest(normalized)
    sealed = Generation6Manifest.model_validate(normalized)
    path.write_text(sealed.model_dump_json(indent=2), encoding="utf-8")
    if store is not None:
        store.save_generation(
            sealed.generation_id,
            sealed.manifest_hash,
            "sealed",
            sealed.model_dump(mode="json"),
            {
                "completed_trials": 0,
                "total_trials": total_generation6_trials(sealed),
                "manifest_path": str(path.resolve()),
            },
        )
    return sealed


def load_generation6_manifest(path: Path) -> Generation6Manifest:
    manifest = Generation6Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.status != "sealed" or not manifest.manifest_hash:
        raise ValueError("Generation 6 manifest must be sealed")
    if manifest.manifest_hash != _manifest_digest(manifest.model_dump(mode="json")):
        raise ValueError("sealed Generation 6 manifest was modified")
    if manifest.implementation_hash != current_generation6_implementation_hash():
        raise ValueError("Generation 6 implementation changed; create a new lineage")
    if file_sha256(Path(manifest.dataset_path)) != manifest.dataset_hash:
        raise ValueError("sealed Generation 6 dataset no longer matches")
    report = json.loads(Path(manifest.learning_report_path).read_text(encoding="utf-8"))
    actual_hash = stable_hash({**report, "report_hash": ""})
    if (
        report.get("report_hash") != manifest.learning_report_hash
        or actual_hash != manifest.learning_report_hash
    ):
        raise ValueError("sealed Generation 6 learning report no longer matches")
    return manifest


def total_generation6_trials(manifest: Generation6Manifest) -> int:
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


def generation6_trial_specs(manifest: Generation6Manifest) -> list[TrialSpec]:
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
    total = total_generation6_trials(manifest)
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


class Generation6Strategy(BaseStrategy):
    def __init__(self, family: str, parameters: dict[str, Any], trial_hash: str) -> None:
        if family not in generation6_spaces():
            raise ValueError(f"unknown Generation 6 family: {family}")
        self.strategy_id = family
        self.timeframe = str(parameters["timeframe"])
        self.spec_parameters = dict(parameters)
        self.max_holding_bars = int(parameters["max_holding_bars"])
        slowest = max(
            int(parameters.get("slow_window", 0)),
            int(parameters.get("slow_lookback", 0)),
            int(parameters.get("regime_window", 0)),
            int(parameters.get("rsi_period", 0)),
            int(parameters.get("atr_period", 14)),
        )
        self.warmup = max(slowest + 5, 60)
        super().__init__(
            StrategyParameters(
                lookback=max(int(parameters.get("slow_lookback", 20)), 2),
                atr_period=int(parameters.get("atr_period", 14)),
                stop_atr=float(parameters.get("stop_atr", 1.5)),
                reward_risk=float(parameters.get("reward_risk", 1.5)),
                fast_window=int(parameters.get("fast_window", 12)),
                slow_window=int(parameters.get("slow_window", 55)),
            ),
            variant_id=f"g6-{trial_hash[:16]}",
        )
        self._sealed_strategy_hash = stable_hash(
            {
                "generation": GENERATION6_SCHEMA,
                "family": self.strategy_id,
                "timeframe": self.timeframe,
                "parameters": self.spec_parameters,
                "implementation": current_generation6_implementation_hash(),
            }
        )

    @property
    def strategy_hash(self) -> str:
        return self._sealed_strategy_hash

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=bars.index, data={"side": None, "reason": None})
        close = bars["close"].astype(float)
        atr_value = atr(bars, int(self.spec_parameters["atr_period"])).replace(0, np.nan)
        family = self.strategy_id
        buy = pd.Series(False, index=bars.index)
        sell = pd.Series(False, index=bars.index)

        if family == "trend-state":
            fast = close.ewm(span=int(self.spec_parameters["fast_window"]), adjust=False).mean()
            slow = close.ewm(span=int(self.spec_parameters["slow_window"]), adjust=False).mean()
            slope_window = int(self.spec_parameters["slope_window"])
            slope = (fast - fast.shift(slope_window)) / atr_value
            threshold = float(self.spec_parameters["minimum_slope_atr"])
            buy = (fast > slow) & (close > fast) & (slope >= threshold)
            sell = (fast < slow) & (close < fast) & (slope <= -threshold)
        elif family == "momentum-consensus":
            fast_move = close - close.shift(int(self.spec_parameters["fast_lookback"]))
            slow_move = close - close.shift(int(self.spec_parameters["slow_lookback"]))
            normalized_move = fast_move / atr_value
            threshold = float(self.spec_parameters["minimum_move_atr"])
            buy = (fast_move > 0) & (slow_move > 0) & (normalized_move >= threshold)
            sell = (fast_move < 0) & (slow_move < 0) & (normalized_move <= -threshold)
        elif family == "rsi-recovery":
            oscillator = rsi(close, int(self.spec_parameters["rsi_period"]))
            level = float(self.spec_parameters["rsi_level"])
            buy = (oscillator.shift(1) <= level) & (oscillator > level)
            sell = (oscillator.shift(1) >= 100 - level) & (oscillator < 100 - level)
            regime_window = int(self.spec_parameters["regime_window"])
            if regime_window:
                regime = close.ewm(span=regime_window, adjust=False).mean()
                buy &= close >= regime
                sell &= close <= regime
        elif family == "session-trend":
            fast = close.ewm(span=int(self.spec_parameters["fast_window"]), adjust=False).mean()
            slow = close.ewm(span=int(self.spec_parameters["slow_window"]), adjust=False).mean()
            hours = pd.to_datetime(bars["timestamp"], utc=True).dt.hour
            start_text, end_text = str(self.spec_parameters["session_window"]).split("-")
            session_open = hours.ge(int(start_text)) & hours.lt(int(end_text))
            buy = session_open & (fast > slow) & (close > fast)
            sell = session_open & (fast < slow) & (close < fast)

        if self.spec_parameters.get("direction") == "long-only":
            sell[:] = False
        result.loc[buy, ["side", "reason"]] = [
            Side.BUY.value,
            f"Generation 6 {family} long condition",
        ]
        result.loc[sell, ["side", "reason"]] = [
            Side.SELL.value,
            f"Generation 6 {family} short condition",
        ]
        result["atr"] = atr_value
        return result


def generation6_strategy_from_spec(spec: dict[str, Any]) -> Generation6Strategy:
    return Generation6Strategy(
        str(spec["family"]), dict(spec["parameters"]), str(spec["trial_hash"])
    )


def explain_generation6_result(result: dict[str, Any]) -> dict[str, Any]:
    details = list(result.get("gate_details") or [])
    failed = [item["reason"] for item in details if not item["passed"]]
    if failed:
        prefix = "Pruned" if result.get("status") == "pruned" else "Rejected"
        summary = f"{prefix} because " + " ".join(failed)
    else:
        summary = (
            "Passed every Generation 6 activity, profitability, cost, stability, and benchmark "
            "discovery gate. Final validation and explicit approval are still required."
        )
    return {"summary": summary, "failed_reasons": failed, "gates": details}


def _fold_pnl(
    bars: pd.DataFrame, strategy: Generation6Strategy, settings: Settings, fold_count: int = 6
) -> list[float]:
    if len(bars) < strategy.warmup + fold_count * 10:
        return []
    splits = np.array_split(np.arange(len(bars)), fold_count)
    output: list[float] = []
    simulator = Generation5Simulator(settings.risk, settings.simulation, settings.starting_equity)
    for split in splits:
        if len(split) == 0:
            continue
        start = max(int(split[0]) - strategy.warmup, 0)
        subset = bars.iloc[start : int(split[-1]) + 1].reset_index(drop=True)
        trades, _metrics = simulator.run(subset, strategy)
        first_timestamp = pd.Timestamp(bars.iloc[int(split[0])]["timestamp"]).to_pydatetime()
        output.append(
            float(sum(trade.pnl for trade in trades if trade.entry_at >= first_timestamp))
        )
    return output


_WORKER_DATA: pd.DataFrame | None = None
_WORKER_SETTINGS: Settings | None = None
_WORKER_HOLDOUT_START: datetime | None = None
_WORKER_POLICY: dict[str, Any] = {}
_WORKER_RESAMPLED: dict[str, pd.DataFrame] = {}


def _initialize_worker(
    dataset_path: str,
    settings: dict[str, Any],
    holdout_start: str,
    activity_policy: dict[str, Any],
) -> None:
    global _WORKER_DATA, _WORKER_SETTINGS, _WORKER_HOLDOUT_START
    global _WORKER_POLICY, _WORKER_RESAMPLED
    raw = _read_dataset(Path(dataset_path))
    _WORKER_DATA, _evidence = normalize_xau_session(raw)
    _WORKER_SETTINGS = Settings.model_validate(settings)
    _WORKER_HOLDOUT_START = datetime.fromisoformat(holdout_start).astimezone(UTC)
    _WORKER_POLICY = dict(activity_policy)
    _WORKER_RESAMPLED = {}


def _evaluate_trial(spec_payload: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_DATA is None or _WORKER_SETTINGS is None or _WORKER_HOLDOUT_START is None:
        raise RuntimeError("Generation 6 worker was not initialized")
    spec = TrialSpec.model_validate(spec_payload)
    strategy = generation6_strategy_from_spec(spec.model_dump(mode="json"))
    if strategy.timeframe not in _WORKER_RESAMPLED:
        _WORKER_RESAMPLED[strategy.timeframe] = resample_bars(_WORKER_DATA, strategy.timeframe)
    all_bars = _WORKER_RESAMPLED[strategy.timeframe]
    timestamps = pd.to_datetime(all_bars["timestamp"], utc=True)
    discovery = all_bars[timestamps < _WORKER_HOLDOUT_START].reset_index(drop=True)
    recent_start = pd.Timestamp(_WORKER_HOLDOUT_START) - pd.DateOffset(months=24)
    recent_indices = discovery.index[
        pd.to_datetime(discovery["timestamp"], utc=True) >= recent_start
    ]
    context_start = (
        max(int(recent_indices.min()) - strategy.warmup, 0) if len(recent_indices) else 0
    )
    recent = discovery.iloc[context_start:].reset_index(drop=True)
    simulator = Generation5Simulator(
        _WORKER_SETTINGS.risk, _WORKER_SETTINGS.simulation, _WORKER_SETTINGS.starting_equity
    )
    _recent_trades, recent_metrics = simulator.run(recent, strategy)
    minimum_recent = int(_WORKER_POLICY["minimum_recent_24_month_trades"])
    stage_one_details = [
        _gate_detail(
            "recent_trade_count",
            "Recent 24-month trades",
            recent_metrics["trade_count"] >= minimum_recent,
            str(int(recent_metrics["trade_count"])),
            f">= {minimum_recent}",
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
        activity_credit = (
            min(float(recent_metrics["trade_count"]) / max(minimum_recent, 1), 2.0) * 0.005
        )
        return {
            "status": "pruned",
            "stage": 1,
            "score": float(
                recent_metrics["net_return"] - recent_metrics["max_drawdown"] + activity_credit
            ),
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
    stressed_trades, stressed_metrics = Generation5Simulator(
        _WORKER_SETTINGS.risk, stressed_settings, _WORKER_SETTINGS.starting_equity
    ).run(discovery, strategy)
    folds = _fold_pnl(discovery, strategy, _WORKER_SETTINGS)
    profitable_fraction = float(np.mean(np.asarray(folds) > 0)) if folds else 0.0
    positive_folds = [value for value in folds if value > 0]
    concentration = max(positive_folds) / sum(positive_folds) if positive_folds else 1.0
    annual_counts = _annual_trade_counts(trades, _WORKER_HOLDOUT_START)
    minimum_annual = min(annual_counts, default=0)
    median_annual = float(np.median(annual_counts)) if annual_counts else 0.0
    benchmark = buy_hold_benchmark(discovery)
    strategy_benchmark = _strategy_benchmark(metrics, benchmark["years"])
    minimum_total = int(_WORKER_POLICY["minimum_discovery_trades"])
    minimum_each_year = int(_WORKER_POLICY["minimum_each_recent_full_year"])
    minimum_median_year = int(_WORKER_POLICY["minimum_median_recent_full_year"])
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
            "Discovery trade count",
            metrics["trade_count"] >= minimum_total,
            str(int(metrics["trade_count"])),
            f">= {minimum_total}",
        ),
        _gate_detail(
            "annual_trade_floor",
            "Lowest recent full-year trades",
            minimum_annual >= minimum_each_year,
            str(minimum_annual),
            f">= {minimum_each_year}",
        ),
        _gate_detail(
            "annual_trade_median",
            "Median recent full-year trades",
            median_annual >= minimum_median_year,
            f"{median_annual:.1f}",
            f">= {minimum_median_year}",
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
    selection_statistics = {
        "deflated_sharpe_probability": _deflated_sharpe_probability(
            float(metrics["sharpe"]), int(metrics["trade_count"]), spec.search_trial_count
        ),
        "stationary_bootstrap_p_value": _bootstrap_p_value(
            [float(trade.pnl) for trade in trades], int(spec.trial_hash[:8], 16)
        ),
    }
    activity_score = min(median_annual / max(minimum_median_year, 1), 2.0) * 0.02
    score = (
        float(metrics["net_return"])
        + float(stressed_metrics["net_return"]) * 0.75
        + min(float(metrics["sharpe"]), 3.0) * 0.02
        - float(metrics["max_drawdown"]) * 1.5
        + min(float(strategy_benchmark["calmar"]), 3.0) * 0.02
        + activity_score
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
        "annual_trade_counts": annual_counts,
        "holding_statistics": _holding_statistics(trades),
        "fold_pnl": folds,
        "fold_profitable_fraction": profitable_fraction,
        "largest_profit_fraction": concentration,
        "stressed_trade_count": len(stressed_trades),
        "selection_statistics": selection_statistics,
    }


class Generation6Coordinator:
    """Single-writer, resumable coordinator for the sealed Generation 6 lineage."""

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.sleep_lease = SleepLease()

    def request_control(self, action: Literal["run", "pause", "stop"]) -> None:
        self.store.set_state(
            "generation6_control",
            {"action": action, "requested_at": datetime.now(UTC).isoformat()},
        )

    def run(self, manifest_path: Path) -> dict[str, Any]:
        manifest = load_generation6_manifest(manifest_path)
        health = resource_health(self.settings, manifest.workers)
        existing = self.store.list_trials(manifest.generation_id)
        if not health["healthy"]:
            paused_summary = {
                "manifest_path": str(manifest_path.resolve()),
                "completed_trials": len(existing),
                "total_trials": total_generation6_trials(manifest),
                "shortlisted": sum(
                    item["result"].get("status") == "shortlisted" for item in existing
                ),
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
        completed = {item["spec"]["trial_hash"] for item in existing}
        pending = [
            spec for spec in generation6_trial_specs(manifest) if spec.trial_hash not in completed
        ]
        summary: dict[str, Any] = {
            "manifest_path": str(manifest_path.resolve()),
            "completed_trials": len(completed),
            "total_trials": len(completed) + len(pending),
            "shortlisted": sum(item["result"].get("status") == "shortlisted" for item in existing),
            "best_score": max(
                (float(item["result"].get("score", -math.inf)) for item in existing), default=None
            ),
            "resource_health": health,
            "started_at": datetime.now(UTC).isoformat(),
            "sleep_lease": self.sleep_lease.active,
            "parent_manifest_hash": manifest.parent_manifest_hash,
            "learning_report_hash": manifest.learning_report_hash,
            "activity_policy": manifest.activity_policy,
            "session_policy": manifest.session_policy,
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
            manifest.activity_policy,
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
                control = self.store.get_state("generation6_control") or {"action": "run"}
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
                minimum_batches = max(self.settings.generation.minimum_batches_before_futility, 24)
                patience = max(self.settings.generation.no_improvement_batches, 12)
                if (
                    batches >= minimum_batches
                    and no_improvement >= patience
                    and summary["shortlisted"] == 0
                ):
                    status = "futile"
                    summary["stop_reason"] = (
                        "no shortlist or score improvement after broad four-family coverage"
                    )
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
            report_path = self.settings.project_root / "var/reports/generation-6-summary.json"
            report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return {"status": status, **summary}

    def finalize(
        self, manifest_path: Path, trial_hash: str, external_dataset: Path
    ) -> dict[str, Any]:
        manifest = load_generation6_manifest(manifest_path)
        selected = next(
            (
                item
                for item in self.store.list_trials(manifest.generation_id)
                if item["spec"]["trial_hash"] == trial_hash
            ),
            None,
        )
        if selected is None or selected["result"].get("status") != "shortlisted":
            raise ValueError("only a completed Generation 6 shortlist can be finalized")
        holdout_key = f"generation6_holdout:{trial_hash}"
        prior = self.store.get_state(holdout_key)
        if prior is not None:
            candidate_hash = prior.get("candidate_hash")
            existing = self.store.get_research_run(str(candidate_hash)) if candidate_hash else None
            if existing is not None:
                return existing
            raise ValueError("this Generation 6 lineage already opened final validation")

        spec = selected["spec"]
        strategy = generation6_strategy_from_spec(spec)
        primary_raw = _read_dataset(Path(manifest.dataset_path))
        primary, primary_session = normalize_xau_session(primary_raw)
        primary_bars = resample_bars(primary, strategy.timeframe)
        timestamps = pd.to_datetime(primary_bars["timestamp"], utc=True)
        holdout = primary_bars[
            (timestamps >= manifest.holdout_start) & (timestamps <= manifest.dataset_end)
        ].reset_index(drop=True)

        external = external_dataset.resolve()
        if not external.is_file() or external == Path(manifest.dataset_path).resolve():
            raise ValueError("an independently sourced external dataset is required")
        external_hash = file_sha256(external)
        if external_hash == manifest.dataset_hash:
            raise ValueError("external validation data must not duplicate the primary dataset")
        external_raw = _read_dataset(external)
        external_normalized, external_session = normalize_xau_session(external_raw)
        external_bars = resample_bars(external_normalized, strategy.timeframe)
        external_timestamps = pd.to_datetime(external_bars["timestamp"], utc=True)
        external_holdout = external_bars[
            (external_timestamps >= manifest.holdout_start)
            & (external_timestamps <= manifest.dataset_end)
        ].reset_index(drop=True)
        if len(external_holdout) < strategy.warmup + 10:
            raise ValueError("external dataset does not cover enough of final validation")

        self.store.set_state(
            holdout_key,
            {
                "opened_at": datetime.now(UTC).isoformat(),
                "manifest_hash": manifest.manifest_hash,
                "external_dataset_hash": external_hash,
            },
        )
        simulator = Generation5Simulator(
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
        stressed_trades, stressed = Generation5Simulator(
            self.settings.risk, stressed_settings, self.settings.starting_equity
        ).run(holdout, strategy)
        external_trades, external_metrics = simulator.run(external_holdout, strategy)
        if not external_trades:
            external_metrics = calculate_metrics([], self.settings.starting_equity)

        monthly_counts = _monthly_trade_counts(trades)
        minimum_monthly = int(manifest.activity_policy["minimum_trades_per_active_month"])
        active_months = sum(value >= minimum_monthly for value in monthly_counts.values())
        minimum_trades = int(manifest.activity_policy["final_holdout_trades"])
        minimum_active_months = int(manifest.activity_policy["final_active_months"])
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
            "monthly_activity": active_months >= minimum_active_months,
            "fold_stability": selected["result"].get("fold_profitable_fraction", 0)
            >= self.settings.research.min_profitable_fold_fraction,
            "profit_concentration": selected["result"].get("largest_profit_fraction", 1) <= 0.5,
            "selection_bias": selection.get("deflated_sharpe_probability", 0) >= 0.50,
            "bootstrap_significance": selection.get("stationary_bootstrap_p_value", 1) <= 0.10,
            "external_validation": bool(
                external_metrics["net_return"] > 0
                and external_metrics["profit_factor"] >= 1.05
                and external_metrics["trade_count"] >= minimum_trades * 0.75
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
            "activity": {
                "monthly_trade_counts": monthly_counts,
                "active_months": active_months,
                "holding_statistics": _holding_statistics(trades),
            },
            "data": {
                "dataset_path": manifest.dataset_path,
                "holdout_start": manifest.holdout_start.isoformat(),
                "holdout_end": manifest.dataset_end.isoformat(),
                "external_dataset_path": str(external),
                "primary_session_normalization": primary_session,
                "external_session_normalization": external_session,
            },
            "generation_six": {
                "manifest_hash": manifest.manifest_hash,
                "parent_manifest_hash": manifest.parent_manifest_hash,
                "learning_report_hash": manifest.learning_report_hash,
                "trial_spec": spec,
                "discovery_result": selected["result"],
                "holdout_opened_once": True,
                "adaptive_period_reuse": True,
                "new_hypothesis_lineage": True,
            },
            "generation_five": {
                "runtime_compatibility": "shared-session-and-maximum-holding-semantics",
                "trial_spec": spec,
            },
            "trades": [trade.as_dict() for trade in trades],
            "stressed_trade_count": len(stressed_trades),
            "disclaimer": (
                "Simulation only. This adaptive lineage reuses a period observed by earlier "
                "lineages. Finalization does not approve or start paper trading."
            ),
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


class Generation6Worker:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.running = True

    def run_forever(self, poll_seconds: float = 0.5) -> None:
        def stop(*_args: Any) -> None:
            self.running = False
            Generation6Coordinator(self.settings, self.store).request_control("stop")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        self.store.recover_processing_commands("generation6.")
        while self.running:
            command = self.store.claim_next_command("generation6.")
            if command is None:
                time.sleep(poll_seconds)
                continue
            try:
                if command["type"] != "generation6.run":
                    raise ValueError(f"unsupported Generation 6 command: {command['type']}")
                payload = command["payload"]
                manifest = load_generation6_manifest(Path(payload["manifest_path"]))
                if (
                    payload.get("generation_id") != manifest.generation_id
                    or payload.get("manifest_hash") != manifest.manifest_hash
                ):
                    raise ValueError("queued Generation 6 lineage does not match")
                result = Generation6Coordinator(self.settings, self.store).run(
                    Path(payload["manifest_path"])
                )
                self.store.complete_command(command["id"], result)
            except Exception as exc:
                self.store.complete_command(
                    command["id"], {"error": str(exc), "type": type(exc).__name__}, failed=True
                )


class Generation6PaperWorker(PaperWorker):
    """Isolated paper worker so adding G6 cannot mutate the sealed G5 runtime hash."""

    def _resolve_approved_strategy(self, report: dict[str, Any]) -> Any:
        evidence = report.get("generation_six")
        if evidence:
            strategy = generation6_strategy_from_spec(evidence["trial_spec"])
            if strategy.strategy_hash != report["strategy_hash"]:
                raise ValueError("Generation 6 strategy no longer matches its approved hash")
            return strategy
        return super()._resolve_approved_strategy(report)

    def _restore_live_if_needed(self) -> None:
        paper = self.store.get_state("paper") or {}
        if paper.get("mode") != "live-g6":
            return
        candidate_hash = paper.get("candidate_hash")
        report = self.store.get_research_run(str(candidate_hash)) if candidate_hash else None
        if report is None or not report.get("generation_six"):
            return
        paper["mode"] = "live"
        self.store.set_state("paper", paper)
        super()._restore_live_if_needed()
        restored = self.store.get_state("paper") or paper
        restored["mode"] = "live-g6"
        restored["worker_prefix"] = "paper6."
        self.store.set_state("paper", restored)

    def run_forever(self, poll_seconds: float = 0.5) -> None:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        self.store.recover_processing_commands("paper6.")
        self._restore_live_if_needed()
        while self.running:
            command = self.store.claim_next_command("paper6.")
            if command is not None:
                try:
                    mapped = command["type"].replace("paper6.", "paper.", 1)
                    result = self._handle(mapped, command["payload"])
                    if result.get("mode") == "live":
                        result["mode"] = "live-g6"
                    result["worker_prefix"] = "paper6."
                    self.store.set_state("paper", result)
                    self.store.complete_command(command["id"], result)
                except Exception as exc:
                    self.sleep_lease.release()
                    self.store.complete_command(
                        command["id"], {"error": str(exc), "type": type(exc).__name__}, failed=True
                    )
            paper = self.store.get_state("paper") or {}
            live_running = (
                self.live_engine is not None
                and paper.get("mode") in {"live", "live-g6"}
                and paper.get("status") in {"running", "risk-halted"}
            )
            if live_running and time.monotonic() >= self.next_live_poll:
                engine = self.live_engine
                assert engine is not None
                updated = engine.poll_once()
                delay = float(
                    updated.get("next_retry_seconds", self.settings.live_paper.poll_seconds)
                )
                self.next_live_poll = time.monotonic() + delay
            time.sleep(poll_seconds)
        self.sleep_lease.release()


def generation6_worker_main() -> None:
    from gold_trader.config import get_settings

    settings = get_settings()
    store = Store(settings.database_url)
    store.initialize()
    Generation6Worker(settings, store).run_forever()


def generation6_paper_worker_main() -> None:
    from gold_trader.config import get_settings

    settings = get_settings()
    store = Store(settings.database_url)
    store.initialize()
    Generation6PaperWorker(settings, store).run_forever()
