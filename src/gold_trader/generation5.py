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
from gold_trader.domain import Side
from gold_trader.generation import (
    GenerationStrategy,
    TrialSpec,
    _bootstrap_p_value,
    _deflated_sharpe_probability,
    _probability_of_backtest_overfit,
    _read_dataset,
    resource_health,
)
from gold_trader.generation4 import buy_hold_benchmark
from gold_trader.research import stable_hash
from gold_trader.simulator import DeterministicSimulator, TradeResult, calculate_metrics
from gold_trader.sleep import SleepLease
from gold_trader.strategies import BaseStrategy

GENERATION5_ID = "generation-5"
GENERATION5_SCHEMA = "5.0"
LEARNING_SCHEMA = "generation-5-frequency-learning-1"


def current_generation5_implementation_hash() -> str:
    source_root = Path(__file__).resolve().parent
    files = (
        "generation5.py",
        "generation.py",
        "strategies.py",
        "simulator.py",
        "risk.py",
        "features.py",
        "live_paper.py",
        "worker.py",
    )
    return stable_hash(
        {
            "package": __version__,
            "generation_schema": GENERATION5_SCHEMA,
            "source_files": {name: file_sha256(source_root / name) for name in files},
        }
    )


class Generation5Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = GENERATION5_SCHEMA
    generation_id: str = GENERATION5_ID
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
    seed: int = 20260725
    workers: int = Field(ge=1, le=12)
    trials_per_family: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    families: dict[str, dict[str, list[Any]]]
    frequency_policy: dict[str, Any]
    session_policy: dict[str, Any]
    settings_snapshot: dict[str, Any]
    implementation_hash: str
    manifest_hash: str = ""
    disclaimer: str = (
        "Simulation research only. Generation 5 targets repeatable activity, not maximum trade "
        "count. Its period was used by earlier lineages, so all cross-generation comparisons "
        "must disclose adaptive reuse and profitability is not promised."
    )


def _manifest_digest(payload: dict[str, Any]) -> str:
    normalized = dict(payload)
    normalized["manifest_hash"] = ""
    return stable_hash(normalized)


def generation5_spaces() -> dict[str, dict[str, list[Any]]]:
    common: dict[str, list[Any]] = {
        "timeframe": ["15min", "1h"],
        "lookback": [8, 12, 20, 35, 55],
        "atr_period": [10, 14],
        "stop_atr": [1.0, 1.25, 1.5, 2.0],
        "reward_risk": [1.25, 1.5, 2.0],
        "direction": ["both", "long-only"],
        "regime_window": [0, 50, 100],
        "max_holding_bars": [16, 32, 48],
    }
    return {
        "compression-breakout": {
            **common,
            "compression_window": [12, 20, 36, 55],
            "compression_ratio": [0.65, 0.75, 0.85],
        },
        "time-series-momentum": dict(common),
        "trend-pullback": {
            **common,
            "fast_window": [8, 12, 20],
            "slow_window": [35, 55, 80],
        },
        "mean-reversion": {
            **common,
            "band_std": [1.5, 1.75, 2.0],
            "rsi_level": [25, 30, 35],
        },
    }


def normalize_xau_session(bars: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply one conservative UTC XAU session calendar to every source."""
    ordered = bars.copy()
    timestamps = pd.to_datetime(ordered["timestamp"], utc=True)
    weekday = timestamps.dt.dayofweek
    hour = timestamps.dt.hour
    session = (weekday <= 3) | ((weekday == 4) & (hour < 22)) | (
        (weekday == 6) & (hour >= 22)
    )
    filtered = ordered.loc[session].sort_values("timestamp").reset_index(drop=True)
    return filtered, {
        "policy": "utc-sunday-22-through-friday-22",
        "input_rows": len(ordered),
        "output_rows": len(filtered),
        "removed_rows": int((~session).sum()),
        "removed_fraction": float((~session).mean()) if len(ordered) else 0.0,
    }


def build_generation5_learning_report(
    settings: Settings,
    store: Store,
    parent_generation_id: str,
    output_path: Path,
) -> dict[str, Any]:
    parent = store.get_generation(parent_generation_id)
    if parent is None or parent["status"] != "completed":
        raise ValueError("Generation 4 must be completed before creating Generation 5")
    finalizations = []
    for run in store.list_research_runs(limit=500):
        evidence = run.get("generation_four")
        if not evidence or evidence.get("manifest_hash") != parent["manifest_hash"]:
            continue
        finalizations.append(
            {
                "candidate_hash": run["candidate_hash"],
                "trial_hash": evidence["trial_spec"]["trial_hash"],
                "timeframe": run["timeframe"],
                "holdout_trade_count": run["metrics"]["trade_count"],
                "external_trade_count": run.get("external_metrics", {}).get("trade_count", 0),
                "failed_gates": [
                    name for name, passed in run.get("gates", {}).items() if not passed
                ],
            }
        )
    report: dict[str, Any] = {
        "schema_version": LEARNING_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "parent_generation_id": parent_generation_id,
        "parent_manifest_hash": parent["manifest_hash"],
        "dataset_hash": parent["manifest"]["dataset_hash"],
        "holdout_start": parent["manifest"]["holdout_start"],
        "adaptive_period_reuse": True,
        "prior_finalizations": finalizations,
        "fixed_spaces": generation5_spaces(),
        "lessons": [
            "Generation 4 candidates passed quality gates but failed final annual trade count.",
            "Long holding periods blocked otherwise valid signals under the one-position rule.",
            "Independent data contained materially more weekend bars than the primary source.",
            "Generation 5 applies one session calendar and tests explicit maximum holding times.",
            "Activity gates are evaluated in each of the five most recent discovery years.",
        ],
        "settings_hash": stable_hash(
            settings.model_dump(mode="json", exclude={"session_secret"})
        ),
        "disclaimer": (
            "This lineage adapts to aggregate lessons from earlier finalizations. A passing result "
            "is still simulation evidence and must complete forward paper observation."
        ),
    }
    report["report_hash"] = stable_hash({**report, "report_hash": ""})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def create_generation5_manifest(
    settings: Settings,
    store: Store,
    parent_generation_id: str,
    learning_report_path: Path,
    output_path: Path,
    *,
    generation_id: str = GENERATION5_ID,
    workers: int | None = None,
    trials_per_family: int = 384,
    batch_size: int | None = None,
) -> Generation5Manifest:
    report = build_generation5_learning_report(
        settings, store, parent_generation_id, learning_report_path
    )
    parent = store.get_generation(parent_generation_id)
    assert parent is not None
    parent_manifest = parent["manifest"]
    dataset = Path(str(parent_manifest["dataset_path"])).resolve()
    bars = _read_dataset(dataset)
    manifest = Generation5Manifest(
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
        families=generation5_spaces(),
        frequency_policy={
            "minimum_recent_24_month_trades": 72,
            "minimum_discovery_trades": 360,
            "minimum_each_recent_full_year": 40,
            "minimum_median_recent_full_year": 48,
            "final_holdout_trades": 48,
            "final_active_months": 9,
            "minimum_trades_per_active_month": 2,
        },
        session_policy={
            "name": "utc-sunday-22-through-friday-22",
            "saturday": "excluded",
            "sunday": "22:00 UTC onward",
            "friday": "before 22:00 UTC",
            "applies_to": "primary and independent validation sources",
        },
        settings_snapshot=settings.model_dump(mode="json", exclude={"session_secret"}),
        implementation_hash=current_generation5_implementation_hash(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def seal_generation5_manifest(
    path: Path, store: Store | None = None
) -> Generation5Manifest:
    current = Generation5Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    if current.status == "sealed":
        if current.manifest_hash != _manifest_digest(current.model_dump(mode="json")):
            raise ValueError("sealed Generation 5 manifest checksum does not match")
        return current
    if file_sha256(Path(current.dataset_path)) != current.dataset_hash:
        raise ValueError("Generation 5 dataset changed before sealing")
    learning = Path(current.learning_report_path)
    if not learning.is_file():
        raise ValueError("Generation 5 learning report is missing")
    report = json.loads(learning.read_text(encoding="utf-8"))
    actual_report_hash = stable_hash({**report, "report_hash": ""})
    if (
        report.get("report_hash") != current.learning_report_hash
        or actual_report_hash != current.learning_report_hash
    ):
        raise ValueError("Generation 5 learning report changed before sealing")
    sealed_payload = current.model_dump(mode="json")
    sealed_payload.update({"status": "sealed", "sealed_at": datetime.now(UTC)})
    normalized = Generation5Manifest.model_validate(sealed_payload).model_dump(mode="json")
    normalized["manifest_hash"] = _manifest_digest(normalized)
    sealed = Generation5Manifest.model_validate(normalized)
    path.write_text(sealed.model_dump_json(indent=2), encoding="utf-8")
    if store is not None:
        store.save_generation(
            sealed.generation_id,
            sealed.manifest_hash,
            "sealed",
            sealed.model_dump(mode="json"),
            {
                "completed_trials": 0,
                "total_trials": total_generation5_trials(sealed),
                "manifest_path": str(path.resolve()),
            },
        )
    return sealed


def load_generation5_manifest(path: Path) -> Generation5Manifest:
    manifest = Generation5Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.status != "sealed" or not manifest.manifest_hash:
        raise ValueError("Generation 5 manifest must be sealed")
    if manifest.manifest_hash != _manifest_digest(manifest.model_dump(mode="json")):
        raise ValueError("sealed Generation 5 manifest was modified")
    if manifest.implementation_hash != current_generation5_implementation_hash():
        raise ValueError("Generation 5 implementation changed; create a new lineage")
    if file_sha256(Path(manifest.dataset_path)) != manifest.dataset_hash:
        raise ValueError("sealed Generation 5 dataset no longer matches")
    report = json.loads(Path(manifest.learning_report_path).read_text(encoding="utf-8"))
    actual_report_hash = stable_hash({**report, "report_hash": ""})
    if (
        report.get("report_hash") != manifest.learning_report_hash
        or actual_report_hash != manifest.learning_report_hash
    ):
        raise ValueError("sealed Generation 5 learning report no longer matches")
    return manifest


def total_generation5_trials(manifest: Generation5Manifest) -> int:
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


def generation5_trial_specs(manifest: Generation5Manifest) -> list[TrialSpec]:
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
                "offset": int(
                    stable_hash({"seed": manifest.seed, "family": family})[:12], 16
                ),
            }
        )
    specs: list[TrialSpec] = []
    ordinal = 0
    total = total_generation5_trials(manifest)
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


class Generation5Strategy(GenerationStrategy):
    def __init__(self, family: str, parameters: dict[str, Any], trial_hash: str) -> None:
        self.generation5_parameters = dict(parameters)
        self.max_holding_bars = int(parameters["max_holding_bars"])
        super().__init__(family, parameters, trial_hash)
        self.variant_id = f"g5-{trial_hash[:16]}"
        self.warmup = max(self.warmup, int(parameters.get("regime_window", 0)) + 5)

    @property
    def strategy_hash(self) -> str:
        return stable_hash(
            {
                "generation": GENERATION5_SCHEMA,
                "family": self.strategy_id,
                "timeframe": self.timeframe,
                "parameters": self.generation5_parameters,
                "implementation": current_generation5_implementation_hash(),
            }
        )

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        result = super().precompute(bars)
        if self.generation5_parameters.get("direction") == "long-only":
            result.loc[result["side"].eq("sell"), ["side", "reason"]] = [None, None]
        regime_window = int(self.generation5_parameters.get("regime_window", 0))
        if regime_window:
            close = bars["close"].astype(float)
            regime = close.ewm(span=regime_window, adjust=False).mean()
            invalid_buy = result["side"].eq("buy") & (close <= regime)
            invalid_sell = result["side"].eq("sell") & (close >= regime)
            result.loc[invalid_buy | invalid_sell, ["side", "reason"]] = [None, None]
        result["reason"] = result["reason"].str.replace(
            "Generation 3", "Generation 5", regex=False
        )
        return result


def generation5_strategy_from_spec(spec: dict[str, Any]) -> Generation5Strategy:
    return Generation5Strategy(
        str(spec["family"]), dict(spec["parameters"]), str(spec["trial_hash"])
    )


class Generation5Simulator(DeterministicSimulator):
    """Deterministic simulator with a pre-registered maximum holding period."""

    def _execute(
        self, data: pd.DataFrame, entry_index: int, signal: Any, lots: float
    ) -> tuple[TradeResult, int]:
        entry_bar = data.iloc[entry_index]
        half_spread = self.simulation.spread_points / 2
        adverse = half_spread + self.simulation.slippage_points
        sign = 1 if signal.side is Side.BUY else -1
        entry_price = float(entry_bar["open"]) + sign * adverse
        stop_distance = abs(signal.reference_price - signal.stop_loss)
        target_distance = abs(signal.take_profit - signal.reference_price)
        stop = entry_price - sign * stop_distance
        target = entry_price + sign * target_distance
        exit_price = entry_price
        reason = "end-of-data"
        exit_index = len(data) - 1
        maximum = int(getattr(signal, "max_holding_bars", 0) or 0)
        if not maximum:
            maximum = int(getattr(self, "max_holding_bars", 0) or 0)

        for cursor in range(entry_index, len(data)):
            bar = data.iloc[cursor]
            open_price = float(bar["open"])
            high = float(bar["high"])
            low = float(bar["low"])
            if signal.side is Side.BUY:
                if open_price <= stop:
                    exit_price, reason = open_price - adverse, "gap-through-stop"
                elif open_price >= target:
                    exit_price, reason = open_price - adverse, "gap-through-target"
                elif low <= stop:
                    exit_price, reason = stop - adverse, "stop-loss"
                elif high >= target:
                    exit_price, reason = target - adverse, "take-profit"
                elif maximum and cursor - entry_index + 1 >= maximum:
                    exit_price, reason = float(bar["close"]) - adverse, "maximum-holding-time"
                else:
                    continue
            else:
                if open_price >= stop:
                    exit_price, reason = open_price + adverse, "gap-through-stop"
                elif open_price <= target:
                    exit_price, reason = open_price + adverse, "gap-through-target"
                elif high >= stop:
                    exit_price, reason = stop + adverse, "stop-loss"
                elif low <= target:
                    exit_price, reason = target + adverse, "take-profit"
                elif maximum and cursor - entry_index + 1 >= maximum:
                    exit_price, reason = float(bar["close"]) + adverse, "maximum-holding-time"
                else:
                    continue
            exit_index = cursor
            break
        else:
            final = data.iloc[-1]
            exit_price = float(final["close"]) - sign * adverse

        entry_at = pd.Timestamp(entry_bar["timestamp"]).to_pydatetime().astimezone(UTC)
        exit_at = pd.Timestamp(data.iloc[exit_index]["timestamp"]).to_pydatetime().astimezone(UTC)
        held_days = max((exit_at - entry_at).total_seconds() / 86400, 0)
        swap = self.simulation.swap_per_lot_day * lots * held_days
        commission = self.simulation.commission_per_lot_round_trip * lots
        gross = (exit_price - entry_price) * sign * self.risk.contract_size_oz * lots
        pnl = gross - commission + swap
        return (
            TradeResult(
                trade_id=str(uuid4()),
                strategy_id=signal.strategy_id,
                side=signal.side.value,
                entry_at=entry_at,
                exit_at=exit_at,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_loss=stop,
                take_profit=target,
                lots=lots,
                pnl=pnl,
                commission=commission,
                swap=swap,
                reason=reason,
            ),
            exit_index,
        )

    def run(
        self, bars: pd.DataFrame, strategy: BaseStrategy
    ) -> tuple[list[TradeResult], dict[str, float]]:
        self.max_holding_bars = int(getattr(strategy, "max_holding_bars", 0))
        return super().run(bars, strategy)


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


def explain_generation5_result(result: dict[str, Any]) -> dict[str, Any]:
    details = list(result.get("gate_details") or [])
    failed = [item["reason"] for item in details if not item["passed"]]
    if failed:
        prefix = "Pruned" if result.get("status") == "pruned" else "Rejected"
        summary = f"{prefix} because " + " ".join(failed)
    else:
        summary = (
            "Passed every Generation 5 profitability, cost, frequency, stability, and "
            "benchmark discovery gate. Final validation and operator approval still apply."
        )
    return {"summary": summary, "failed_reasons": failed, "gates": details}


def _annual_trade_counts(trades: list[TradeResult], anchor: datetime) -> list[int]:
    output: list[int] = []
    anchor_timestamp = pd.Timestamp(anchor)
    for years_back in range(5, 0, -1):
        start = anchor_timestamp - pd.DateOffset(years=years_back)
        end = anchor_timestamp - pd.DateOffset(years=years_back - 1)
        output.append(
            sum(start <= pd.Timestamp(trade.entry_at) < end for trade in trades)
        )
    return output


def _monthly_trade_counts(trades: list[TradeResult]) -> dict[str, int]:
    counts = Counter(pd.Timestamp(trade.entry_at).strftime("%Y-%m") for trade in trades)
    return dict(sorted(counts.items()))


def _holding_statistics(trades: list[TradeResult]) -> dict[str, float]:
    if not trades:
        return {"median_hours": 0.0, "maximum_hours": 0.0}
    durations = np.asarray(
        [(trade.exit_at - trade.entry_at).total_seconds() / 3600 for trade in trades],
        dtype=float,
    )
    return {
        "median_hours": float(np.median(durations)),
        "maximum_hours": float(np.max(durations)),
    }


def _strategy_benchmark(metrics: dict[str, float], years: float) -> dict[str, float]:
    total_return = float(metrics["net_return"])
    annualized = (
        float((1 + total_return) ** (1 / years) - 1)
        if years > 0 and total_return > -1
        else 0.0
    )
    drawdown = float(metrics["max_drawdown"])
    return {
        "annualized_return": annualized,
        "calmar": float(annualized / max(drawdown, 1e-12)),
    }


def _fold_pnl(
    bars: pd.DataFrame, strategy: Generation5Strategy, settings: Settings, fold_count: int = 6
) -> list[float]:
    if len(bars) < strategy.warmup + fold_count * 10:
        return []
    splits = np.array_split(np.arange(len(bars)), fold_count)
    output: list[float] = []
    simulator = Generation5Simulator(
        settings.risk, settings.simulation, settings.starting_equity
    )
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
    frequency_policy: dict[str, Any],
) -> None:
    global _WORKER_DATA, _WORKER_SETTINGS, _WORKER_HOLDOUT_START
    global _WORKER_POLICY, _WORKER_RESAMPLED
    raw = _read_dataset(Path(dataset_path))
    _WORKER_DATA, _quality = normalize_xau_session(raw)
    _WORKER_SETTINGS = Settings.model_validate(settings)
    _WORKER_HOLDOUT_START = datetime.fromisoformat(holdout_start).astimezone(UTC)
    _WORKER_POLICY = dict(frequency_policy)
    _WORKER_RESAMPLED = {}


def _evaluate_trial(spec_payload: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_DATA is None or _WORKER_SETTINGS is None or _WORKER_HOLDOUT_START is None:
        raise RuntimeError("Generation 5 worker was not initialized")
    spec = TrialSpec.model_validate(spec_payload)
    strategy = generation5_strategy_from_spec(spec.model_dump(mode="json"))
    if strategy.timeframe not in _WORKER_RESAMPLED:
        _WORKER_RESAMPLED[strategy.timeframe] = resample_bars(
            _WORKER_DATA, strategy.timeframe
        )
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
        _WORKER_SETTINGS.risk,
        _WORKER_SETTINGS.simulation,
        _WORKER_SETTINGS.starting_equity,
    )
    _recent_trades, recent_metrics = simulator.run(recent, strategy)
    recent_minimum = int(_WORKER_POLICY["minimum_recent_24_month_trades"])
    stage_one_details = [
        _gate_detail(
            "recent_trade_count",
            "Recent 24-month trades",
            recent_metrics["trade_count"] >= recent_minimum,
            str(int(recent_metrics["trade_count"])),
            f">= {recent_minimum}",
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


class Generation5Coordinator:
    """Single-writer, resumable coordinator for frequency-first local research."""

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.sleep_lease = SleepLease()

    def request_control(self, action: Literal["run", "pause", "stop"]) -> None:
        self.store.set_state(
            "generation5_control",
            {"action": action, "requested_at": datetime.now(UTC).isoformat()},
        )

    def run(self, manifest_path: Path) -> dict[str, Any]:
        manifest = load_generation5_manifest(manifest_path)
        health = resource_health(self.settings, manifest.workers)
        if not health["healthy"]:
            paused_summary = {
                "manifest_path": str(manifest_path.resolve()),
                "completed_trials": len(self.store.list_trials(manifest.generation_id)),
                "total_trials": total_generation5_trials(manifest),
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
            item["spec"]["trial_hash"]
            for item in self.store.list_trials(manifest.generation_id)
        }
        pending = [
            spec for spec in generation5_trial_specs(manifest) if spec.trial_hash not in completed
        ]
        total = len(completed) + len(pending)
        existing = self.store.list_trials(manifest.generation_id)
        summary: dict[str, Any] = {
            "manifest_path": str(manifest_path.resolve()),
            "completed_trials": len(completed),
            "total_trials": total,
            "shortlisted": sum(
                item["result"].get("status") == "shortlisted" for item in existing
            ),
            "best_score": max(
                (float(item["result"].get("score", -math.inf)) for item in existing),
                default=None,
            ),
            "resource_health": health,
            "started_at": datetime.now(UTC).isoformat(),
            "sleep_lease": self.sleep_lease.active,
            "parent_manifest_hash": manifest.parent_manifest_hash,
            "learning_report_hash": manifest.learning_report_hash,
            "frequency_policy": manifest.frequency_policy,
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
            manifest.frequency_policy,
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
                control = self.store.get_state("generation5_control") or {"action": "run"}
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
                    if summary["best_score"] is None or score > float(
                        str(summary["best_score"])
                    ):
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
                            item["result"].get("status") == "shortlisted"
                            for item in trials
                        ),
                        "batches": batches,
                        "no_improvement_batches": no_improvement,
                        "probability_of_backtest_overfit": (
                            _probability_of_backtest_overfit(trials)
                        ),
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
            report_path = self.settings.project_root / "var/reports/generation-5-summary.json"
            report_path.write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
        return {"status": status, **summary}

    def finalize(
        self, manifest_path: Path, trial_hash: str, external_dataset: Path
    ) -> dict[str, Any]:
        manifest = load_generation5_manifest(manifest_path)
        selected = next(
            (
                item
                for item in self.store.list_trials(manifest.generation_id)
                if item["spec"]["trial_hash"] == trial_hash
            ),
            None,
        )
        if selected is None or selected["result"].get("status") != "shortlisted":
            raise ValueError("only a completed Generation 5 shortlist can be finalized")
        holdout_key = f"generation_holdout:{trial_hash}"
        prior_open = self.store.get_state(holdout_key)
        if prior_open is not None:
            candidate_hash = prior_open.get("candidate_hash")
            existing = (
                self.store.get_research_run(str(candidate_hash)) if candidate_hash else None
            )
            if existing is not None:
                return existing
            raise ValueError("this Generation 5 lineage already opened final validation")

        spec = selected["spec"]
        strategy = generation5_strategy_from_spec(spec)
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
        minimum_monthly = int(manifest.frequency_policy["minimum_trades_per_active_month"])
        active_months = sum(value >= minimum_monthly for value in monthly_counts.values())
        minimum_trades = int(manifest.frequency_policy["final_holdout_trades"])
        minimum_active_months = int(manifest.frequency_policy["final_active_months"])
        selection = selected["result"].get("selection_statistics", {})
        benchmark = buy_hold_benchmark(holdout)
        strategy_benchmark = _strategy_benchmark(metrics, benchmark["years"])
        gates = {
            "positive_holdout_return": metrics["net_return"] > 0,
            "positive_stressed_return": stressed["net_return"] > 0,
            "profit_factor": (
                metrics["profit_factor"] >= self.settings.research.min_profit_factor
            ),
            "sharpe": metrics["sharpe"] >= self.settings.research.min_sharpe,
            "drawdown": metrics["max_drawdown"] <= self.settings.research.max_drawdown,
            "trade_count": metrics["trade_count"] >= minimum_trades,
            "monthly_activity": active_months >= minimum_active_months,
            "fold_stability": selected["result"].get("fold_profitable_fraction", 0)
            >= self.settings.research.min_profitable_fold_fraction,
            "profit_concentration": (
                selected["result"].get("largest_profit_fraction", 1) <= 0.5
            ),
            "selection_bias": selection.get("deflated_sharpe_probability", 0) >= 0.50,
            "external_validation": bool(
                external_metrics["net_return"] > 0
                and external_metrics["profit_factor"] >= 1.05
                and external_metrics["trade_count"] >= minimum_trades * 0.75
            ),
            "risk_adjusted_benchmark": (
                strategy_benchmark["calmar"] >= benchmark["calmar"]
            ),
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
            "generation_five": {
                "manifest_hash": manifest.manifest_hash,
                "parent_manifest_hash": manifest.parent_manifest_hash,
                "learning_report_hash": manifest.learning_report_hash,
                "trial_spec": spec,
                "discovery_result": selected["result"],
                "holdout_opened_once": True,
                "adaptive_period_reuse": True,
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


class Generation5Worker:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.running = True

    def run_forever(self, poll_seconds: float = 0.5) -> None:
        import signal
        import time

        def stop(*_args: Any) -> None:
            self.running = False
            Generation5Coordinator(self.settings, self.store).request_control("stop")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        self.store.recover_processing_commands("generation5.")
        while self.running:
            command = self.store.claim_next_command("generation5.")
            if command is None:
                time.sleep(poll_seconds)
                continue
            try:
                if command["type"] != "generation5.run":
                    raise ValueError(
                        f"unsupported Generation 5 command: {command['type']}"
                    )
                payload = command["payload"]
                manifest = load_generation5_manifest(Path(payload["manifest_path"]))
                if payload.get("generation_id") != manifest.generation_id:
                    raise ValueError("queued Generation 5 ID does not match")
                if payload.get("manifest_hash") != manifest.manifest_hash:
                    raise ValueError("queued Generation 5 manifest hash does not match")
                result = Generation5Coordinator(self.settings, self.store).run(
                    Path(payload["manifest_path"])
                )
                self.store.complete_command(command["id"], result)
            except Exception as exc:
                self.store.complete_command(
                    command["id"],
                    {"error": str(exc), "type": type(exc).__name__},
                    failed=True,
                )


def generation5_worker_main() -> None:
    from gold_trader.config import get_settings

    settings = get_settings()
    store = Store(settings.database_url)
    store.initialize()
    Generation5Worker(settings, store).run_forever()
