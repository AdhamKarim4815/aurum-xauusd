from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
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
from gold_trader.data import file_sha256, normalize_bars, resample_bars
from gold_trader.database import Store
from gold_trader.features import atr, rsi
from gold_trader.research import stable_hash
from gold_trader.simulator import DeterministicSimulator, calculate_metrics
from gold_trader.sleep import SleepLease
from gold_trader.strategies import BaseStrategy, StrategyParameters

GENERATION_ID = "generation-3"
GENERATION_SCHEMA = "3.0"

FAMILY_SPACES: dict[str, dict[str, list[Any]]] = {
    "session-breakout": {
        "timeframe": ["15min"],
        "lookback": [8, 12, 20, 36, 55],
        "atr_period": [10, 14, 20],
        "stop_atr": [1.0, 1.25, 1.5, 2.0],
        "reward_risk": [1.0, 1.5, 2.0, 2.5, 3.0],
        "session_start": [6, 7, 8],
        "session_end": [15, 16, 17, 19],
    },
    "volatility-breakout": {
        "timeframe": ["5min", "15min"],
        "lookback": [8, 12, 20, 36, 55],
        "atr_period": [10, 14, 20],
        "stop_atr": [1.0, 1.5, 2.0, 2.5],
        "reward_risk": [1.0, 1.5, 2.0, 2.5, 3.0],
        "volatility_multiple": [0.8, 1.0, 1.2, 1.5],
    },
    "trend-pullback": {
        "timeframe": ["1h"],
        "lookback": [20, 36, 55, 80],
        "atr_period": [10, 14, 20],
        "stop_atr": [1.0, 1.5, 2.0, 2.5],
        "reward_risk": [1.0, 1.5, 2.0, 2.5, 3.0],
        "fast_window": [8, 12, 20, 30],
        "slow_window": [36, 50, 80, 120],
    },
    "time-series-momentum": {
        "timeframe": ["1h", "4h"],
        "lookback": [12, 20, 35, 55, 80, 120],
        "atr_period": [10, 14, 20],
        "stop_atr": [1.25, 1.5, 2.0, 2.5, 3.0],
        "reward_risk": [1.0, 1.5, 2.0, 2.5, 3.0],
    },
    "mean-reversion": {
        "timeframe": ["15min", "1h"],
        "lookback": [12, 20, 36, 55],
        "atr_period": [10, 14, 20],
        "stop_atr": [0.75, 1.0, 1.25, 1.5, 2.0],
        "reward_risk": [0.75, 1.0, 1.25, 1.5, 2.0],
        "band_std": [1.5, 1.75, 2.0, 2.25, 2.5],
        "rsi_level": [20, 25, 30, 35],
    },
    "compression-breakout": {
        "timeframe": ["15min", "1h"],
        "lookback": [8, 12, 20, 36],
        "atr_period": [10, 14, 20],
        "stop_atr": [1.0, 1.5, 2.0, 2.5],
        "reward_risk": [1.0, 1.5, 2.0, 2.5, 3.0],
        "compression_window": [20, 36, 55, 80],
        "compression_ratio": [0.55, 0.65, 0.75, 0.85],
    },
}


def current_implementation_hash() -> str:
    source_root = Path(__file__).resolve().parent
    files = (
        "generation.py",
        "strategies.py",
        "simulator.py",
        "risk.py",
        "features.py",
    )
    return stable_hash(
        {
            "package": __version__,
            "generation_schema": GENERATION_SCHEMA,
            "source_files": {name: file_sha256(source_root / name) for name in files},
        }
    )


class GenerationManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = GENERATION_SCHEMA
    generation_id: str = GENERATION_ID
    status: Literal["draft", "sealed"] = "draft"
    created_at: datetime
    sealed_at: datetime | None = None
    dataset_path: str
    dataset_hash: str
    dataset_rows: int = Field(gt=0)
    dataset_start: datetime
    dataset_end: datetime
    holdout_start: datetime
    seed: int = 20260723
    workers: int = Field(ge=1, le=12)
    trials_per_family: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    families: dict[str, dict[str, list[Any]]]
    settings_snapshot: dict[str, Any]
    implementation_hash: str
    manifest_hash: str = ""
    disclaimer: str = (
        "Research simulation only. Search results are not approvals and profitability is not "
        "assumed or promised."
    )


class TrialSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    generation_id: str
    manifest_hash: str
    ordinal: int
    search_trial_count: int
    family: str
    parameters: dict[str, Any]
    trial_hash: str


def explain_trial_result(spec: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Turn machine gates into operator-facing evidence with actual thresholds."""
    metrics = result.get("metrics") or {}
    stressed = result.get("stressed_metrics") or {}
    gates = result.get("gates") or {}
    timeframe = str((spec.get("parameters") or {}).get("timeframe", ""))
    intraday = timeframe in {"5min", "15min"}

    def number(source: dict[str, Any], key: str) -> float:
        value = source.get(key, 0.0)
        return float(value) if value is not None else 0.0

    def gate(
        key: str,
        label: str,
        value: float,
        requirement: str,
        calculated_pass: bool,
        *,
        display: str,
    ) -> dict[str, Any]:
        passed = bool(gates.get(key, calculated_pass))
        return {
            "key": key,
            "label": label,
            "passed": passed,
            "actual": display,
            "requirement": requirement,
            "reason": f"{label} was {display}; required {requirement}.",
        }

    if int(result.get("stage", 0)) == 1:
        minimum_trades = 20 if intraday else 8
        trade_count = number(metrics, "trade_count")
        net_return = number(metrics, "net_return")
        drawdown = number(metrics, "max_drawdown")
        details = [
            gate(
                "recent_trade_count",
                "Recent trade count",
                trade_count,
                f">= {minimum_trades}",
                trade_count >= minimum_trades,
                display=str(int(trade_count)),
            ),
            gate(
                "recent_return",
                "Recent net return",
                net_return,
                "> -2.00%",
                net_return > -0.02,
                display=f"{net_return * 100:.2f}%",
            ),
            gate(
                "recent_drawdown",
                "Recent max drawdown",
                drawdown,
                "<= 15.00%",
                drawdown <= 0.15,
                display=f"{drawdown * 100:.2f}%",
            ),
        ]
    else:
        minimum_trades = 60 if intraday else 25
        net_return = number(metrics, "net_return")
        stressed_return = number(stressed, "net_return")
        profit_factor = number(metrics, "profit_factor")
        sharpe = number(metrics, "sharpe")
        drawdown = number(metrics, "max_drawdown")
        trade_count = number(metrics, "trade_count")
        fold_fraction = number(result, "fold_profitable_fraction")
        concentration = number(result, "largest_profit_fraction")
        details = [
            gate(
                "positive_return",
                "Base net return",
                net_return,
                "> 0.00%",
                net_return > 0,
                display=f"{net_return * 100:.2f}%",
            ),
            gate(
                "positive_stressed_return",
                "Doubled-cost net return",
                stressed_return,
                "> 0.00%",
                stressed_return > 0,
                display=f"{stressed_return * 100:.2f}%",
            ),
            gate(
                "profit_factor",
                "Profit factor",
                profit_factor,
                ">= 1.05",
                profit_factor >= 1.05,
                display=f"{profit_factor:.2f}",
            ),
            gate(
                "sharpe",
                "Sharpe ratio",
                sharpe,
                ">= 0.30",
                sharpe >= 0.30,
                display=f"{sharpe:.2f}",
            ),
            gate(
                "drawdown",
                "Max drawdown",
                drawdown,
                "<= 15.00%",
                drawdown <= 0.15,
                display=f"{drawdown * 100:.2f}%",
            ),
            gate(
                "trade_count",
                "Trade count",
                trade_count,
                f">= {minimum_trades}",
                trade_count >= minimum_trades,
                display=str(int(trade_count)),
            ),
            gate(
                "fold_stability",
                "Profitable folds",
                fold_fraction,
                ">= 50.00%",
                fold_fraction >= 0.50,
                display=f"{fold_fraction * 100:.2f}%",
            ),
            gate(
                "profit_concentration",
                "Largest fold profit share",
                concentration,
                "<= 60.00%",
                concentration <= 0.60,
                display=f"{concentration * 100:.2f}%",
            ),
        ]

    failed_reasons = [item["reason"] for item in details if not item["passed"]]
    status = str(result.get("status", "unknown"))
    if failed_reasons:
        prefix = "Pruned" if status == "pruned" else "Rejected"
        summary = f"{prefix} because " + " ".join(failed_reasons)
    else:
        summary = (
            "Passed every discovery gate. Final holdout and external-dataset gates still apply."
        )
    return {
        "summary": summary,
        "failed_reasons": failed_reasons,
        "gates": details,
    }


def _read_dataset(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    normalized, quality = normalize_bars(raw)
    if not quality.valid:
        raise ValueError(f"generation dataset failed quality: {quality.as_dict()}")
    return normalized


def _manifest_digest(payload: dict[str, Any]) -> str:
    hashed = dict(payload)
    hashed["manifest_hash"] = ""
    return stable_hash(hashed)


def create_manifest(
    settings: Settings,
    dataset_path: Path,
    output_path: Path,
    *,
    generation_id: str = GENERATION_ID,
    workers: int | None = None,
    trials_per_family: int | None = None,
    batch_size: int | None = None,
) -> GenerationManifest:
    dataset = dataset_path.resolve()
    bars = _read_dataset(dataset)
    start = pd.Timestamp(bars["timestamp"].min()).to_pydatetime().astimezone(UTC)
    end = pd.Timestamp(bars["timestamp"].max()).to_pydatetime().astimezone(UTC)
    holdout_start = (pd.Timestamp(end) - pd.DateOffset(months=12)).to_pydatetime()
    manifest = GenerationManifest(
        generation_id=generation_id,
        created_at=datetime.now(UTC),
        dataset_path=str(dataset),
        dataset_hash=file_sha256(dataset),
        dataset_rows=len(bars),
        dataset_start=start,
        dataset_end=end,
        holdout_start=holdout_start,
        workers=workers or settings.generation.workers,
        trials_per_family=trials_per_family or settings.generation.trials_per_family,
        batch_size=batch_size or settings.generation.batch_size,
        families=FAMILY_SPACES,
        settings_snapshot=settings.model_dump(mode="json", exclude={"session_secret"}),
        implementation_hash=current_implementation_hash(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def seal_manifest(path: Path, store: Store | None = None) -> GenerationManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    current = GenerationManifest.model_validate(raw)
    if current.status == "sealed":
        if current.manifest_hash != _manifest_digest(current.model_dump(mode="json")):
            raise ValueError("sealed manifest checksum does not match")
        return current
    dataset = Path(current.dataset_path)
    if not dataset.is_file() or file_sha256(dataset) != current.dataset_hash:
        raise ValueError("dataset changed or disappeared before manifest sealing")
    sealed_payload = current.model_dump(mode="json")
    sealed_payload.update({"status": "sealed", "sealed_at": datetime.now(UTC)})
    normalized = GenerationManifest.model_validate(sealed_payload).model_dump(mode="json")
    normalized["manifest_hash"] = _manifest_digest(normalized)
    sealed = GenerationManifest.model_validate(normalized)
    path.write_text(sealed.model_dump_json(indent=2), encoding="utf-8")
    if store is not None:
        store.save_generation(
            sealed.generation_id,
            sealed.manifest_hash,
            "sealed",
            sealed.model_dump(mode="json"),
            {"completed_trials": 0, "total_trials": _total_trials(sealed)},
        )
    return sealed


def load_sealed_manifest(path: Path) -> GenerationManifest:
    manifest = GenerationManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.status != "sealed" or not manifest.manifest_hash:
        raise ValueError("generation manifest must be sealed before it can run")
    if manifest.manifest_hash != _manifest_digest(manifest.model_dump(mode="json")):
        raise ValueError("sealed generation manifest was modified")
    if manifest.implementation_hash != current_implementation_hash():
        raise ValueError("Generation 3 implementation changed; create a new manifest lineage")
    dataset = Path(manifest.dataset_path)
    if not dataset.is_file() or file_sha256(dataset) != manifest.dataset_hash:
        raise ValueError("sealed generation dataset no longer matches its hash")
    return manifest


def _total_trials(manifest: GenerationManifest) -> int:
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


def trial_specs(manifest: GenerationManifest) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    ordinal = 0
    prepared: list[dict[str, Any]] = []
    for family, space in manifest.families.items():
        space_size = reduce(mul, (len(values) for values in space.values()), 1)
        multiplier = 104_729
        while math.gcd(multiplier, space_size) != 1:
            multiplier += 2
        offset = int(stable_hash({"seed": manifest.seed, "family": family})[:12], 16)
        prepared.append(
            {
                "family": family,
                "space": space,
                "space_size": space_size,
                "limit": min(manifest.trials_per_family, space_size),
                "multiplier": multiplier,
                "offset": offset,
            }
        )
    for family_ordinal in range(max(int(item["limit"]) for item in prepared)):
        for item in prepared:
            if family_ordinal >= int(item["limit"]):
                continue
            family = str(item["family"])
            space = item["space"]
            space_size = int(item["space_size"])
            space_index = (
                int(item["offset"]) + family_ordinal * int(item["multiplier"])
            ) % space_size
            parameters = _decode_space(space, space_index)
            base = {
                "generation_id": manifest.generation_id,
                "manifest_hash": manifest.manifest_hash,
                "ordinal": ordinal,
                "search_trial_count": _total_trials(manifest),
                "family": family,
                "parameters": parameters,
            }
            specs.append(TrialSpec(**base, trial_hash=stable_hash(base)))
            ordinal += 1
    return specs


class GenerationStrategy(BaseStrategy):
    def __init__(self, family: str, parameters: dict[str, Any], trial_hash: str) -> None:
        if family not in FAMILY_SPACES:
            raise ValueError(f"unknown Generation 3 family: {family}")
        self.strategy_id = family
        self.timeframe = str(parameters["timeframe"])
        self.spec_parameters = dict(parameters)
        slowest = max(
            int(parameters.get("lookback", 20)),
            int(parameters.get("slow_window", 0)),
            int(parameters.get("compression_window", 0)),
            int(parameters.get("atr_period", 14)),
        )
        self.warmup = max(slowest + 5, 60)
        common = StrategyParameters(
            lookback=int(parameters.get("lookback", 20)),
            atr_period=int(parameters.get("atr_period", 14)),
            stop_atr=float(parameters.get("stop_atr", 1.5)),
            reward_risk=float(parameters.get("reward_risk", 2.0)),
            fast_window=int(parameters.get("fast_window", 20)),
            slow_window=int(parameters.get("slow_window", 50)),
            band_std=float(parameters.get("band_std", 2.0)),
        )
        super().__init__(common, variant_id=f"g3-{trial_hash[:16]}")
        self._trial_hash = trial_hash

    @property
    def strategy_hash(self) -> str:
        return stable_hash(
            {
                "generation": GENERATION_SCHEMA,
                "family": self.strategy_id,
                "timeframe": self.timeframe,
                "parameters": self.spec_parameters,
                "implementation": current_implementation_hash(),
            }
        )

    def precompute(self, bars: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=bars.index, data={"side": None, "reason": None})
        close = bars["close"]
        lookback = int(self.spec_parameters["lookback"])
        atr_value = atr(bars, int(self.spec_parameters.get("atr_period", 14)))
        previous_high = bars["high"].shift(1).rolling(lookback).max()
        previous_low = bars["low"].shift(1).rolling(lookback).min()
        family = self.strategy_id
        buy = pd.Series(False, index=bars.index)
        sell = pd.Series(False, index=bars.index)

        if family == "session-breakout":
            hours = pd.to_datetime(bars["timestamp"], utc=True).dt.hour
            session = hours.between(
                int(self.spec_parameters["session_start"]),
                int(self.spec_parameters["session_end"]),
            )
            buy = session & (close > previous_high)
            sell = session & (close < previous_low)
        elif family == "volatility-breakout":
            baseline = atr_value.rolling(max(lookback, 20)).mean()
            volatile = atr_value >= baseline * float(self.spec_parameters["volatility_multiple"])
            buy = volatile & (close > previous_high)
            sell = volatile & (close < previous_low)
        elif family == "trend-pullback":
            fast = close.ewm(span=int(self.spec_parameters["fast_window"]), adjust=False).mean()
            slow = close.ewm(span=int(self.spec_parameters["slow_window"]), adjust=False).mean()
            buy = (fast > slow) & (close.shift(1) <= fast.shift(1)) & (close > fast)
            sell = (fast < slow) & (close.shift(1) >= fast.shift(1)) & (close < fast)
        elif family == "time-series-momentum":
            buy = close > previous_high
            sell = close < previous_low
        elif family == "mean-reversion":
            mean = close.rolling(lookback).mean()
            deviation = close.rolling(lookback).std()
            level = float(self.spec_parameters["rsi_level"])
            rsi_value = rsi(close)
            band = float(self.spec_parameters["band_std"])
            buy = (close < mean - band * deviation) & (rsi_value < level)
            sell = (close > mean + band * deviation) & (rsi_value > 100 - level)
        elif family == "compression-breakout":
            window = int(self.spec_parameters["compression_window"])
            range_mean = (bars["high"] - bars["low"]).rolling(window).mean()
            compressed = range_mean < atr_value * float(self.spec_parameters["compression_ratio"])
            released = compressed.shift(1).fillna(False)
            buy = released & (close > previous_high)
            sell = released & (close < previous_low)

        result.loc[buy.fillna(False), ["side", "reason"]] = [
            "buy",
            f"Generation 3 {family} long rule",
        ]
        result.loc[sell.fillna(False), ["side", "reason"]] = [
            "sell",
            f"Generation 3 {family} short rule",
        ]
        result["atr"] = atr_value
        return result


def strategy_from_spec(spec: dict[str, Any]) -> GenerationStrategy:
    return GenerationStrategy(
        str(spec["family"]),
        dict(spec["parameters"]),
        str(spec["trial_hash"]),
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


def _fold_pnl(
    bars: pd.DataFrame, strategy: BaseStrategy, settings: Settings, fold_count: int = 6
) -> list[float]:
    if len(bars) < strategy.warmup + fold_count * 10:
        return []
    splits = np.array_split(np.arange(len(bars)), fold_count)
    output: list[float] = []
    simulator = DeterministicSimulator(settings.risk, settings.simulation, settings.starting_equity)
    for split in splits:
        if len(split) == 0:
            continue
        start = max(int(split[0]) - strategy.warmup, 0)
        subset = bars.iloc[start : int(split[-1]) + 1].reset_index(drop=True)
        trades, _metrics = simulator.run(subset, strategy)
        first_timestamp = pd.Timestamp(bars.iloc[int(split[0])]["timestamp"]).to_pydatetime()
        pnl = sum(trade.pnl for trade in trades if trade.entry_at >= first_timestamp)
        output.append(float(pnl))
    return output


def _bootstrap_p_value(values: list[float], seed: int) -> float:
    if len(values) < 10:
        return 1.0
    sample = np.asarray(values, dtype=float)
    observed = float(sample.mean())
    centered = sample - observed
    random = np.random.default_rng(seed)
    block = max(int(math.sqrt(len(sample))), 2)
    simulated: list[float] = []
    for _ in range(250):
        collected: list[float] = []
        while len(collected) < len(sample):
            start = int(random.integers(0, len(sample)))
            collected.extend(centered.take(np.arange(start, start + block), mode="wrap").tolist())
        simulated.append(float(np.mean(collected[: len(sample)])))
    return float((1 + sum(value >= observed for value in simulated)) / (len(simulated) + 1))


def _deflated_sharpe_probability(sharpe: float, trades: int, trials: int) -> float:
    if trades < 3 or not math.isfinite(sharpe):
        return 0.0
    expected_max = math.sqrt(max(2 * math.log(max(trials, 2)), 0))
    standard_error = math.sqrt((1 + 0.5 * sharpe**2) / max(trades - 1, 1))
    z_score = (sharpe - expected_max * standard_error) / max(standard_error, 1e-12)
    return float(0.5 * (1 + math.erf(z_score / math.sqrt(2))))


def _evaluate_trial(spec_payload: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_DATA is None or _WORKER_SETTINGS is None or _WORKER_HOLDOUT_START is None:
        raise RuntimeError("generation worker was not initialized")
    spec = TrialSpec.model_validate(spec_payload)
    strategy = strategy_from_spec(spec.model_dump(mode="json"))
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
    stage_one_pass = (
        recent_metrics["trade_count"] >= (20 if intraday else 8)
        and recent_metrics["net_return"] > -0.02
        and recent_metrics["max_drawdown"] <= 0.15
    )
    if not stage_one_pass:
        return {
            "status": "pruned",
            "stage": 1,
            "score": float(recent_metrics["net_return"] - recent_metrics["max_drawdown"]),
            "metrics": recent_metrics,
            "gates": {"stage_one": False},
            "fold_pnl": [],
        }

    trades, metrics = simulator.run(discovery, strategy)
    stressed = _WORKER_SETTINGS.simulation.model_copy(
        update={
            "spread_points": _WORKER_SETTINGS.simulation.spread_points * 2,
            "slippage_points": _WORKER_SETTINGS.simulation.slippage_points * 2,
            "commission_per_lot_round_trip": (
                _WORKER_SETTINGS.simulation.commission_per_lot_round_trip * 2
            ),
        }
    )
    stressed_trades, stressed_metrics = DeterministicSimulator(
        _WORKER_SETTINGS.risk, stressed, _WORKER_SETTINGS.starting_equity
    ).run(discovery, strategy)
    folds = _fold_pnl(discovery, strategy, _WORKER_SETTINGS)
    profitable_fraction = float(np.mean(np.asarray(folds) > 0)) if folds else 0.0
    positive_folds = [value for value in folds if value > 0]
    concentration = max(positive_folds) / sum(positive_folds) if positive_folds else 1.0
    minimum_trades = 60 if intraday else 25
    gates = {
        "stage_one": True,
        "positive_return": metrics["net_return"] > 0,
        "positive_stressed_return": stressed_metrics["net_return"] > 0,
        "profit_factor": metrics["profit_factor"] >= 1.05,
        "sharpe": metrics["sharpe"] >= 0.30,
        "drawdown": metrics["max_drawdown"] <= 0.15,
        "trade_count": metrics["trade_count"] >= minimum_trades,
        "fold_stability": profitable_fraction >= 0.50,
        "profit_concentration": concentration <= 0.60,
    }
    trade_pnl = [float(trade.pnl) for trade in trades]
    dsr = _deflated_sharpe_probability(
        float(metrics["sharpe"]),
        int(metrics["trade_count"]),
        spec.search_trial_count,
    )
    spa = _bootstrap_p_value(trade_pnl, int(spec.trial_hash[:8], 16))
    score = (
        float(metrics["net_return"])
        + min(float(metrics["sharpe"]), 3.0) * 0.02
        - float(metrics["max_drawdown"]) * 1.5
        + float(stressed_metrics["net_return"]) * 0.5
    )
    return {
        "status": "shortlisted" if all(gates.values()) else "rejected",
        "stage": 2,
        "score": score,
        "metrics": metrics,
        "stressed_metrics": stressed_metrics,
        "gates": gates,
        "fold_pnl": folds,
        "fold_profitable_fraction": profitable_fraction,
        "largest_profit_fraction": concentration,
        "stressed_trade_count": len(stressed_trades),
        "selection_statistics": {
            "deflated_sharpe_probability": dsr,
            "stationary_bootstrap_p_value": spa,
        },
    }


def resource_health(settings: Settings, workers: int) -> dict[str, Any]:
    disk = shutil.disk_usage(settings.project_root)
    free_gb = disk.free / 1024**3
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    on_ac = True
    if sys.platform == "darwin":
        try:
            output = subprocess.run(
                ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5, check=False
            ).stdout
            on_ac = "AC Power" in output
        except (OSError, subprocess.SubprocessError):
            on_ac = False
    checks = {
        "on_ac_power": on_ac,
        "free_disk": free_gb >= settings.generation.minimum_free_disk_gb,
        "system_load": load <= workers * settings.generation.maximum_load_per_worker,
    }
    return {
        "healthy": all(checks.values()),
        "checks": checks,
        "free_disk_gb": free_gb,
        "load": load,
    }


def _probability_of_backtest_overfit(trials: list[dict[str, Any]]) -> float:
    rows = [item["result"].get("fold_pnl", []) for item in trials]
    matrix = np.asarray([row for row in rows if len(row) >= 4], dtype=float)
    if len(matrix) < 2:
        return 1.0
    folds = matrix.shape[1]
    estimates: list[float] = []
    for split in range(1, folds):
        training = matrix[:, :split].mean(axis=1)
        testing = matrix[:, split:].mean(axis=1)
        winner = int(np.argmax(training))
        rank = float(np.argsort(np.argsort(testing))[winner] / max(len(testing) - 1, 1))
        estimates.append(float(rank < 0.5))
    return float(np.mean(estimates))


class GenerationCoordinator:
    """Single-writer, resumable coordinator for bounded local CPU research."""

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.sleep_lease = SleepLease()

    def request_control(self, action: Literal["run", "pause", "stop"]) -> None:
        self.store.set_state(
            "generation_control", {"action": action, "requested_at": datetime.now(UTC).isoformat()}
        )

    def run(self, manifest_path: Path) -> dict[str, Any]:
        manifest = load_sealed_manifest(manifest_path)
        health = resource_health(self.settings, manifest.workers)
        if not health["healthy"]:
            paused_summary = {
                "manifest_path": str(manifest_path.resolve()),
                "completed_trials": len(self.store.list_trials(manifest.generation_id)),
                "total_trials": _total_trials(manifest),
                "shortlisted": 0,
                "resource_health": health,
                "pause_reason": "connect AC power or resolve the failed local resource check",
                "updated_at": datetime.now(UTC).isoformat(),
                "sleep_lease": False,
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
        pending = [spec for spec in trial_specs(manifest) if spec.trial_hash not in completed]
        total = len(completed) + len(pending)
        summary: dict[str, Any] = {
            "manifest_path": str(manifest_path.resolve()),
            "completed_trials": len(completed),
            "total_trials": total,
            "shortlisted": 0,
            "best_score": None,
            "resource_health": health,
            "started_at": datetime.now(UTC).isoformat(),
            "sleep_lease": self.sleep_lease.active,
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
                control = self.store.get_state("generation_control") or {"action": "run"}
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
                if executor is None:
                    results = [_evaluate_trial(payload) for payload in payloads]
                else:
                    results = list(executor.map(_evaluate_trial, payloads))
                improved = False
                for spec, result in zip(batch, results, strict=True):
                    result["completed_at"] = datetime.now(UTC).isoformat()
                    self.store.save_trial(
                        manifest.generation_id, spec.model_dump(mode="json"), result
                    )
                    score = float(result.get("score", -math.inf))
                    if summary["best_score"] is None or score > float(summary["best_score"]):
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
            report_path = self.settings.project_root / "var/reports/generation-3-summary.json"
            report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return {"status": status, **summary}

    def finalize(
        self,
        manifest_path: Path,
        trial_hash: str,
        external_dataset: Path,
    ) -> dict[str, Any]:
        manifest = load_sealed_manifest(manifest_path)
        selected = next(
            (
                item
                for item in self.store.list_trials(manifest.generation_id)
                if item["spec"]["trial_hash"] == trial_hash
            ),
            None,
        )
        if selected is None or selected["result"].get("status") != "shortlisted":
            raise ValueError("only a completed Generation 3 shortlist entry can be finalized")
        holdout_key = f"generation_holdout:{trial_hash}"
        prior_open = self.store.get_state(holdout_key)
        if prior_open is not None:
            candidate_hash = prior_open.get("candidate_hash")
            existing = self.store.get_research_run(str(candidate_hash)) if candidate_hash else None
            if existing is not None:
                return existing
            raise ValueError(
                "this trial lineage already opened the final holdout and needs operator review"
            )
        spec = selected["spec"]
        strategy = strategy_from_spec(spec)
        primary = _read_dataset(Path(manifest.dataset_path))
        bars = resample_bars(primary, strategy.timeframe)
        timestamps = pd.to_datetime(bars["timestamp"], utc=True)
        holdout = bars[timestamps >= manifest.holdout_start].reset_index(drop=True)
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
            "gates": gates,
            "data": {
                "dataset_path": manifest.dataset_path,
                "holdout_start": manifest.holdout_start.isoformat(),
                "external_dataset_path": str(external),
            },
            "generation_three": {
                "manifest_hash": manifest.manifest_hash,
                "trial_spec": spec,
                "discovery_result": selected["result"],
                "holdout_opened_once": True,
            },
            "trades": [trade.as_dict() for trade in trades],
            "stressed_trade_count": len(stressed_trades),
            "disclaimer": (
                "Simulation only. Finalization does not approve or start paper trading."
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


class ResearchWorker:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.running = True

    def run_forever(self, poll_seconds: float = 0.5) -> None:
        import signal
        import time

        def stop(*_args: Any) -> None:
            self.running = False
            GenerationCoordinator(self.settings, self.store).request_control("stop")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        self.store.interrupt_running_generations("research worker restarted")
        self.store.recover_processing_commands("research.")
        while self.running:
            command = self.store.claim_next_command("research.")
            if command is None:
                time.sleep(poll_seconds)
                continue
            try:
                if command["type"] != "research.generation.run":
                    raise ValueError(f"unsupported research command: {command['type']}")
                payload = command["payload"]
                manifest = load_sealed_manifest(Path(payload["manifest_path"]))
                if payload.get("generation_id") != manifest.generation_id:
                    raise ValueError("queued generation ID does not match the sealed manifest")
                if payload.get("manifest_hash") != manifest.manifest_hash:
                    raise ValueError("queued manifest hash does not match the sealed manifest")
                result = GenerationCoordinator(self.settings, self.store).run(
                    Path(payload["manifest_path"])
                )
                self.store.complete_command(command["id"], result)
            except Exception as exc:
                self.store.complete_command(
                    command["id"], {"error": str(exc), "type": type(exc).__name__}, failed=True
                )


def research_worker_main() -> None:
    from gold_trader.config import get_settings

    settings = get_settings()
    store = Store(settings.database_url)
    store.initialize()
    ResearchWorker(settings, store).run_forever()
