from __future__ import annotations

import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from gold_trader.config import Settings, get_settings
from gold_trader.data import (
    AlphaVantageProvider,
    FailoverQuoteProvider,
    FinnhubProvider,
    LiveQuoteProvider,
    MetalsDevProvider,
    TwelveDataProvider,
    file_sha256,
    normalize_bars,
    resample_bars,
)
from gold_trader.database import Store
from gold_trader.domain import AccountSnapshot, Fill, Side
from gold_trader.generation import strategy_from_spec
from gold_trader.live_paper import LivePaperEngine
from gold_trader.ml import ChronologicalSignalMetaModel, MetaFilteredStrategy
from gold_trader.notifications import MacNotifier
from gold_trader.research import stable_hash
from gold_trader.secrets import get_secret
from gold_trader.simulator import DeterministicSimulator
from gold_trader.sleep import SleepLease
from gold_trader.strategies import strategy_registry

PROVIDER_REGISTRY: dict[str, Callable[..., LiveQuoteProvider]] = {
    TwelveDataProvider.name: TwelveDataProvider,
    AlphaVantageProvider.name: AlphaVantageProvider,
    MetalsDevProvider.name: MetalsDevProvider,
    FinnhubProvider.name: FinnhubProvider,
}


class PaperWorker:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        provider_factory: Callable[[], LiveQuoteProvider] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.sleep_lease = SleepLease()
        self.notifier = MacNotifier()
        self.running = True
        self.live_engine: LivePaperEngine | None = None
        self.next_live_poll = 0.0
        self.provider_factory = provider_factory or self._build_live_provider

    def _build_live_provider(self) -> LiveQuoteProvider:
        poll_seconds = max(int(self.settings.live_paper.poll_seconds), 1)
        configured: list[tuple[str, LiveQuoteProvider]] = []
        missing: list[str] = []
        for name in self.settings.live_paper.provider_priority:
            provider_class = PROVIDER_REGISTRY.get(name)
            if provider_class is None:
                continue
            api_key = get_secret(name)
            if not api_key:
                missing.append(name)
                continue
            configured.append((name, provider_class(api_key, poll_seconds=poll_seconds)))
        if not configured:
            missing_list = ", ".join(missing or self.settings.live_paper.provider_priority)
            raise ValueError(
                "no live quote provider has a configured Keychain secret; run "
                f"gold-trader secret set <name> for one of: {missing_list}"
            )
        return FailoverQuoteProvider(configured, notifier=self.notifier)

    def stop(self, *_args: Any) -> None:
        self.running = False
        self.sleep_lease.release()

    def run_forever(self, poll_seconds: float = 0.5) -> None:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        self.store.recover_processing_commands("paper.")
        self._restore_live_if_needed()
        while self.running:
            command = self.store.claim_next_command("paper.")
            if command is not None:
                try:
                    result = self._handle(command["type"], command["payload"])
                    self.store.complete_command(command["id"], result)
                except Exception as exc:
                    self.sleep_lease.release()
                    self.store.complete_command(
                        command["id"],
                        {"error": str(exc), "type": type(exc).__name__},
                        failed=True,
                    )
                    self.notifier.notify(
                        "Aurum worker error", "A safety command failed; inspect audit"
                    )
            paper = self.store.get_state("paper") or {}
            live_running = (
                self.live_engine is not None
                and paper.get("mode") == "live"
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

    def _handle(self, command_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        paper = self.store.get_state("paper") or {}
        if command_type == "paper.start":
            approval = self.store.valid_approval()
            if approval is None or approval["candidate_hash"] != payload["candidate_hash"]:
                raise ValueError("approval is absent or no longer matches")
            if paper.get("kill_switch"):
                raise ValueError("kill switch is active")
            mode = payload.get("mode", "replay")
            report = self.store.get_research_run(payload["candidate_hash"])
            if report is None:
                raise ValueError("approved research evidence is unavailable")
            self._validate_report_lineage(approval, report)
            strategy = self._resolve_approved_strategy(report)
            if strategy.strategy_hash != report["strategy_hash"]:
                self.store.invalidate_approvals()
                raise ValueError("approved strategy implementation hash changed")
            if mode == "live":
                self.live_engine = LivePaperEngine(
                    self.settings,
                    self.store,
                    self.provider_factory(),
                    report,
                    strategy,
                    notifier=self.notifier,
                )
            lease = self.sleep_lease.acquire()
            paper.update(
                {
                    "status": "running",
                    "mode": mode,
                    "candidate_hash": payload["candidate_hash"],
                    "feed_status": "replay-ready" if mode == "replay" else "connecting",
                    "entries_allowed": mode == "replay",
                    "sleep_lease": lease,
                    "reconciliation": "clean",
                }
            )
            self.store.set_state("paper", paper)
            if mode == "live":
                assert self.live_engine is not None
                prepared = self.live_engine.prepare(paper)
                prepared["sleep_lease"] = lease
                self.store.set_state("paper", prepared)
                self.next_live_poll = 0.0
                return prepared
            if mode == "replay":
                dataset_path = report.get("data", {}).get("dataset_path")
                if dataset_path:
                    return self._complete_historical_replay(paper, report, Path(dataset_path))
            return paper
        if command_type == "paper.stop":
            self.sleep_lease.release()
            if self.live_engine is not None:
                stopped = self.live_engine.stop(paper, "operator-stop")
                self.live_engine = None
                return stopped
            paper.update(
                {
                    "status": "stopped",
                    "entries_allowed": False,
                    "sleep_lease": False,
                    "feed_status": "stopped",
                }
            )
            self.store.set_state("paper", paper)
            return paper
        if command_type == "paper.kill":
            self.sleep_lease.release()
            if self.live_engine is not None:
                paper["kill_switch"] = True
                killed = self.live_engine.stop(
                    paper, payload.get("reason", "operator-kill"), killed=True
                )
                killed["reason"] = payload.get("reason", "operator activated")
                self.store.set_state("paper", killed)
                self.live_engine = None
                self.notifier.notify("Aurum kill switch", str(killed["reason"]))
                return killed
            paper.update(
                {
                    "status": "killed",
                    "kill_switch": True,
                    "entries_allowed": False,
                    "sleep_lease": False,
                    "pending_orders_cancelled": True,
                    "reason": payload.get("reason", "operator activated"),
                }
            )
            self.store.set_state("paper", paper)
            self.notifier.notify("Aurum kill switch", str(paper["reason"]))
            return paper
        raise ValueError(f"unsupported command type: {command_type}")

    def _complete_historical_replay(
        self, paper: dict[str, Any], report: dict[str, Any], dataset_path: Path
    ) -> dict[str, Any]:
        if not dataset_path.is_file():
            raise ValueError("approved replay dataset is no longer available")
        raw = (
            pd.read_parquet(dataset_path)
            if dataset_path.suffix == ".parquet"
            else pd.read_csv(dataset_path)
        )
        normalized, quality = normalize_bars(raw)
        if not quality.valid:
            raise ValueError("approved replay dataset no longer passes quality")
        strategy = self._resolve_approved_strategy(report)
        simulator_class: type[DeterministicSimulator] = DeterministicSimulator
        if report.get("generation_five"):
            from gold_trader.generation5 import Generation5Simulator, normalize_xau_session

            normalized, _session_evidence = normalize_xau_session(normalized)
            simulator_class = Generation5Simulator
        bars = resample_bars(normalized, strategy.timeframe)
        trades, metrics = simulator_class(
            self.settings.risk,
            self.settings.simulation,
            self.settings.starting_equity,
        ).run(bars, strategy)
        for trade in trades:
            entry_side = Side(trade.side)
            exit_side = Side.SELL if entry_side is Side.BUY else Side.BUY
            lineage = {
                "data_hash": report["dataset_hash"],
                "config_hash": report["configuration_hash"],
                "strategy_hash": report["strategy_hash"],
                "model_hash": report["model_hash"],
            }
            self.store.append_event(
                Fill(
                    order_id=f"{trade.trade_id}:entry",
                    side=entry_side,
                    lots=trade.lots,
                    price=trade.entry_price,
                    commission=trade.commission / 2,
                    slippage=self.settings.simulation.slippage_points,
                    filled_at=trade.entry_at,
                    **lineage,
                )
            )
            self.store.append_event(
                Fill(
                    order_id=f"{trade.trade_id}:exit",
                    side=exit_side,
                    lots=trade.lots,
                    price=trade.exit_price,
                    commission=trade.commission / 2,
                    slippage=self.settings.simulation.slippage_points,
                    filled_at=trade.exit_at,
                    **lineage,
                )
            )
        final_equity = self.settings.starting_equity + metrics["net_pnl"]
        self.store.append_event(
            AccountSnapshot(
                balance=final_equity,
                equity=final_equity,
                day_start_equity=final_equity,
                rolling_five_day_high=max(self.settings.starting_equity, final_equity),
                peak_equity=max(self.settings.starting_equity, final_equity),
                open_position_count=0,
                data_hash=report["dataset_hash"],
                config_hash=report["configuration_hash"],
                strategy_hash=report["strategy_hash"],
                model_hash=report["model_hash"],
            )
        )
        self.sleep_lease.release()
        paper.update(
            {
                "status": "replay-complete",
                "entries_allowed": False,
                "sleep_lease": False,
                "feed_status": "historical-replay-complete",
                "historical_replay_only": True,
                "historical_replay_trade_count": int(metrics["trade_count"]),
                "observation_trade_count": 0,
                "equity": final_equity,
                "metrics": metrics,
            }
        )
        self.store.set_state("paper", paper)
        return paper

    def _resolve_approved_strategy(self, report: dict[str, Any]) -> Any:
        generation_five = report.get("generation_five")
        if generation_five:
            from gold_trader.generation5 import generation5_strategy_from_spec

            generation5_strategy = generation5_strategy_from_spec(
                generation_five["trial_spec"]
            )
            if generation5_strategy.strategy_hash != report["strategy_hash"]:
                raise ValueError("Generation 5 strategy no longer matches its approved hash")
            return generation5_strategy
        generation_four = report.get("generation_four")
        if generation_four:
            from gold_trader.generation4 import generation4_strategy_from_spec

            generation4_strategy = generation4_strategy_from_spec(
                generation_four["trial_spec"]
            )
            if generation4_strategy.strategy_hash != report["strategy_hash"]:
                raise ValueError("Generation 4 strategy no longer matches its approved hash")
            return generation4_strategy
        generation_three = report.get("generation_three")
        if generation_three:
            generation3_strategy = strategy_from_spec(generation_three["trial_spec"])
            if generation3_strategy.strategy_hash != report["strategy_hash"]:
                raise ValueError("Generation 3 strategy no longer matches its approved hash")
            return generation3_strategy
        strategies = strategy_registry()
        model_name = report.get("model", {}).get("model", "rule-only")
        if model_name == "rule-only":
            strategy_key = report.get("variant_id", report["strategy_id"])
            if strategy_key not in strategies:
                raise ValueError("approved strategy variant is not registered")
            return strategies[strategy_key]
        if model_name != ChronologicalSignalMetaModel.model_id:
            raise ValueError("approved model type is not replayable in this runtime")

        base_variant_id = report.get("base_variant_id")
        if base_variant_id not in strategies:
            raise ValueError("approved model parent strategy is not registered")
        artifact_metadata = report.get("model_artifact") or {}
        artifact_value = artifact_metadata.get("path")
        expected_sha256 = artifact_metadata.get("sha256")
        if not artifact_value or not expected_sha256:
            raise ValueError("approved model artifact metadata is incomplete")
        artifact = Path(str(artifact_value)).resolve()
        allowed_root = (self.settings.project_root / "artifacts/models").resolve()
        expected_name = f"{report['candidate_hash']}.joblib"
        if artifact.parent != allowed_root or artifact.name != expected_name:
            raise ValueError("approved model artifact path is outside the trusted model store")
        if not artifact.is_file() or file_sha256(artifact) != expected_sha256:
            raise ValueError("approved model artifact is missing or failed checksum validation")
        loaded = joblib.load(artifact)
        if not isinstance(loaded, ChronologicalSignalMetaModel):
            raise ValueError("approved model artifact has an unexpected type")
        if loaded.model_hash != report["model_hash"]:
            raise ValueError("approved model artifact no longer matches the model hash")
        return MetaFilteredStrategy(strategies[base_variant_id], loaded)

    def _restore_live_if_needed(self) -> None:
        paper = self.store.get_state("paper") or {}
        if paper.get("mode") != "live" or paper.get("status") not in {
            "running",
            "risk-halted",
        }:
            return
        approval = self.store.valid_approval()
        candidate_hash = paper.get("candidate_hash")
        report = self.store.get_research_run(str(candidate_hash)) if candidate_hash else None
        if (
            approval is None
            or approval.get("candidate_hash") != candidate_hash
            or report is None
        ):
            paper.update(
                {
                    "status": "recovery-required",
                    "entries_allowed": False,
                    "feed_status": "blocked",
                    "reconciliation": "approval-data-or-key-missing",
                }
            )
            self.store.set_state("paper", paper)
            return
        try:
            self._validate_report_lineage(approval, report)
            strategy = self._resolve_approved_strategy(report)
            if strategy.strategy_hash != report["strategy_hash"]:
                raise ValueError("approved strategy implementation hash changed")
            self.live_engine = LivePaperEngine(
                self.settings,
                self.store,
                self.provider_factory(),
                report,
                strategy,
                notifier=self.notifier,
            )
            self.live_engine.prepare(paper)
            lease = self.sleep_lease.acquire()
            paper = self.store.get_state("paper") or paper
            paper.update(
                {
                    "sleep_lease": lease,
                    "reconciliation": "worker-restored-no-duplicate-fills",
                }
            )
            self.store.set_state("paper", paper)
            self.next_live_poll = 0.0
        except Exception as exc:
            paper.update(
                {
                    "status": "recovery-required",
                    "entries_allowed": False,
                    "feed_status": "blocked",
                    "reconciliation": f"restore-failed:{type(exc).__name__}",
                }
            )
            self.store.set_state("paper", paper)

    def _validate_report_lineage(
        self, approval: dict[str, Any], report: dict[str, Any]
    ) -> None:
        for key in (
            "candidate_hash",
            "dataset_hash",
            "configuration_hash",
            "strategy_hash",
            "model_hash",
        ):
            if approval.get(key) != report.get(key):
                self.store.invalidate_approvals()
                raise ValueError(f"approval no longer matches report {key}")
        valid_configuration_hashes = {
            stable_hash(self.settings.model_dump()),
            stable_hash(self.settings.model_dump(mode="json", exclude={"session_secret"})),
        }
        if report["configuration_hash"] not in valid_configuration_hashes:
            self.store.invalidate_approvals()
            raise ValueError("runtime configuration changed and invalidated approval")


def worker_main() -> None:
    settings = get_settings()
    store = Store(settings.database_url)
    store.initialize()
    PaperWorker(settings, store).run_forever()
