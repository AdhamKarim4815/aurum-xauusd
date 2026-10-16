from __future__ import annotations

import signal
import time
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

import pandas as pd

from gold_trader.config import Settings
from gold_trader.data import FailoverQuoteProvider, LiveQuoteProvider
from gold_trader.database import Store
from gold_trader.domain import DomainRecord, Quote
from gold_trader.live_paper import LivePaperEngine
from gold_trader.notifications import MacNotifier
from gold_trader.secrets import get_secret
from gold_trader.sleep import SleepLease
from gold_trader.strategies import BaseStrategy
from gold_trader.worker import PROVIDER_REGISTRY, PaperWorker

PAPER_LAB_STATE = "paper_lab"
PAPER_LAB_PREFIX = "paper_lab:account:"
PAPER_LAB_COMMAND_PREFIX = "paperlab."
ALLOWED_EXPERIMENTAL_FAILURES = frozenset({"trade_count", "monthly_activity"})


def paper_lab_account_key(candidate_hash: str) -> str:
    return f"{PAPER_LAB_PREFIX}{candidate_hash}"


def failed_gates(report: dict[str, Any]) -> set[str]:
    return {str(name) for name, passed in report.get("gates", {}).items() if not passed}


def validate_experimental_report(report: dict[str, Any]) -> None:
    evidence = (
        report.get("generation_six")
        or report.get("generation_five")
        or report.get("generation_four")
        or report.get("generation_three")
    )
    if not evidence or not evidence.get("trial_spec"):
        raise ValueError("live paper lab requires a finalized generation strategy")
    failures = failed_gates(report)
    disallowed = failures - ALLOWED_EXPERIMENTAL_FAILURES
    if disallowed:
        names = ", ".join(sorted(disallowed))
        raise ValueError(f"strategy failed non-overridable safety gates: {names}")
    if not report.get("eligible") and not failures:
        raise ValueError("ineligible strategy has no auditable failed-gate explanation")


class SharedCycleQuoteProvider:
    """Fetch one quote per coordinator cycle and give every strategy an independent record."""

    def __init__(self, provider: LiveQuoteProvider) -> None:
        self.provider = provider
        self._quote: Quote | None = None
        self._error: Exception | None = None

    @property
    def active_provider_name(self) -> str | None:
        return cast(str | None, getattr(self.provider, "active_provider_name", None))

    @property
    def name(self) -> str:
        return str(getattr(self.provider, "name", "shared-live-provider"))

    def begin_cycle(self) -> None:
        self._quote = None
        self._error = None

    async def quote(self, symbol: str = "XAUUSD") -> Quote:
        if self._error is not None:
            raise RuntimeError(str(self._error)) from self._error
        if self._quote is None:
            try:
                self._quote = await self.provider.quote(symbol)
            except Exception as exc:
                self._error = exc
                raise
        return self._quote.model_copy(update={"event_id": str(uuid4())})

    async def backfill(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return await self.provider.backfill(symbol, start, end)


class PaperLabStateStore:
    """Present one namespaced paper account to an unmodified LivePaperEngine."""

    def __init__(self, store: Store, candidate_hash: str, report: dict[str, Any]) -> None:
        self.store = store
        self.candidate_hash = candidate_hash
        self.report = report
        self.state_key = paper_lab_account_key(candidate_hash)
        self.correlation_id = f"paper-lab:{candidate_hash[:16]}"

    def get_state(self, key: str) -> dict[str, Any] | None:
        if key != "paper":
            return self.store.get_state(key)
        return self.store.get_state(self.state_key)

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        self.store.set_state(self.state_key if key == "paper" else key, value)

    def valid_approval(self) -> dict[str, Any]:
        return {
            key: self.report[key]
            for key in (
                "candidate_hash",
                "dataset_hash",
                "configuration_hash",
                "strategy_hash",
                "model_hash",
            )
        }

    def append_event(self, record: DomainRecord) -> None:
        self.store.append_event(record.model_copy(update={"correlation_id": self.correlation_id}))


class DebouncedLabNotifier(MacNotifier):
    def __init__(self) -> None:
        self.last_sent: dict[tuple[str, str], float] = {}

    def notify(self, title: str, message: str) -> bool:
        key = (title, message)
        now = time.monotonic()
        if now - self.last_sent.get(key, 0.0) < 300:
            return False
        self.last_sent[key] = now
        return super().notify(f"{title} · 4-strategy lab", message)


class PaperLabWorker:
    """Run four simultaneous, isolated, simulation-only strategy accounts."""

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.running = True
        self.sleep_lease = SleepLease()
        self.notifier = DebouncedLabNotifier()
        self.engines: dict[str, LivePaperEngine] = {}
        self.shared_provider: SharedCycleQuoteProvider | None = None
        self.next_poll = 0.0

    def _network_retry_delay(self, failure_cycles: int) -> float:
        exponent = min(max(failure_cycles - 1, 0), 6)
        return float(
            min(
                self.settings.live_paper.reconnect_initial_seconds * 2**exponent,
                self.settings.live_paper.reconnect_max_seconds,
            )
        )

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
            names = ", ".join(missing or self.settings.live_paper.provider_priority)
            raise ValueError(
                f"no live quote provider is configured in Keychain; configure one of: {names}"
            )
        return FailoverQuoteProvider(configured, notifier=self.notifier)

    def _resolve_strategy(self, report: dict[str, Any]) -> BaseStrategy:
        if report.get("generation_six"):
            from gold_trader.generation6 import generation6_strategy_from_spec

            strategy = generation6_strategy_from_spec(report["generation_six"]["trial_spec"])
            if strategy.strategy_hash != report["strategy_hash"]:
                raise ValueError("Generation 6 strategy hash changed")
            return strategy
        return cast(
            BaseStrategy,
            PaperWorker(self.settings, self.store)._resolve_approved_strategy(report),
        )

    def _account_seed(
        self, candidate_hash: str, report: dict[str, Any], *, resume: bool
    ) -> dict[str, Any]:
        existing = self.store.get_state(paper_lab_account_key(candidate_hash)) if resume else None
        trial = (
            report.get("generation_six")
            or report.get("generation_five")
            or report.get("generation_four")
            or report.get("generation_three")
            or {}
        ).get("trial_spec", {})
        seed = dict(existing or {})
        seed.update(
            {
                "candidate_hash": candidate_hash,
                "strategy_id": report["strategy_id"],
                "variant_id": report.get("variant_id", report["strategy_id"]),
                "timeframe": report["timeframe"],
                "trial_hash": trial.get("trial_hash"),
                "generation_id": trial.get("generation_id"),
                "experimental_override": True,
                "failed_gates": sorted(failed_gates(report)),
                "simulated_fills_only": True,
            }
        )
        return seed

    def start(self, candidate_hashes: list[str], *, resume: bool = False) -> dict[str, Any]:
        unique = list(dict.fromkeys(candidate_hashes))
        if len(unique) != 4:
            raise ValueError("the live paper lab requires exactly four unique strategies")
        canonical = self.store.get_state("paper") or {}
        if canonical.get("status") in {"running", "risk-halted"}:
            raise ValueError("stop the normal paper account before starting the four-strategy lab")

        reports: dict[str, dict[str, Any]] = {}
        for candidate_hash in unique:
            report = self.store.get_research_run(candidate_hash)
            if report is None:
                raise ValueError(f"unknown finalized candidate: {candidate_hash}")
            validate_experimental_report(report)
            reports[candidate_hash] = report

        self.shared_provider = SharedCycleQuoteProvider(self._build_live_provider())
        prepared_accounts: list[dict[str, Any]] = []
        self.engines = {}
        for candidate_hash, report in reports.items():
            strategy = self._resolve_strategy(report)
            namespaced = PaperLabStateStore(self.store, candidate_hash, report)
            engine = LivePaperEngine(
                self.settings,
                cast(Store, namespaced),
                self.shared_provider,
                report,
                strategy,
                notifier=self.notifier,
            )
            account = engine.prepare(self._account_seed(candidate_hash, report, resume=resume))
            account.update(
                {
                    "mode": "experimental-live-paper",
                    "sleep_lease": False,
                    "started_at": account.get("started_at") or datetime.now(UTC).isoformat(),
                }
            )
            self.store.set_state(paper_lab_account_key(candidate_hash), account)
            self.engines[candidate_hash] = engine
            prepared_accounts.append(account)

        lease = self.sleep_lease.acquire()
        for account in prepared_accounts:
            account["sleep_lease"] = lease
            self.store.set_state(paper_lab_account_key(str(account["candidate_hash"])), account)
        previous_state = self.store.get_state(PAPER_LAB_STATE) or {}
        state = {
            "status": "running",
            "connection_status": "checking",
            "mode": "four-strategy-experimental-live-paper",
            "candidate_hashes": unique,
            "strategy_count": 4,
            "sleep_lease": lease,
            "simulated_fills_only": True,
            "started_at": (
                previous_state.get("started_at")
                if resume and previous_state.get("started_at")
                else datetime.now(UTC).isoformat()
            ),
            "last_poll_at": None,
            "next_retry_at": None,
            "retry_seconds": None,
            "network_failure_cycles": 0,
            "network_pause_count": int(previous_state.get("network_pause_count", 0)),
            "shared_quote_requests_per_cycle": 1,
            "warning": (
                "Experimental paper-only override. These candidates did not meet the final "
                "trade-count requirement. Accounts and returns are independent and not additive."
            ),
        }
        self.store.set_state(PAPER_LAB_STATE, state)
        self.next_poll = 0.0
        return state

    def poll_once(self) -> dict[str, Any]:
        if self.shared_provider is None or not self.engines:
            raise RuntimeError("paper lab is not prepared")
        self.shared_provider.begin_cycle()
        accounts: list[dict[str, Any]] = []
        for candidate_hash, engine in self.engines.items():
            try:
                account = engine.poll_once()
            except Exception as exc:
                account = self.store.get_state(paper_lab_account_key(candidate_hash)) or {}
                account.update(
                    {
                        "status": "worker-error",
                        "feed_status": "blocked",
                        "entries_allowed": False,
                        "last_feed_error": type(exc).__name__,
                    }
                )
                self.store.set_state(paper_lab_account_key(candidate_hash), account)
            accounts.append(account)
        state = self.store.get_state(PAPER_LAB_STATE) or {}
        now = datetime.now(UTC)
        previous_status = str(state.get("status", "running"))
        healthy_feeds = sum(account.get("feed_status") == "healthy" for account in accounts)
        network_errors = sum(account.get("feed_status") == "network-error" for account in accounts)
        state.update(
            {
                "last_poll_at": now.isoformat(),
                "sleep_lease": self.sleep_lease.active,
                "healthy_feeds": healthy_feeds,
                "entries_allowed": sum(
                    bool(account.get("entries_allowed")) for account in accounts
                ),
                "open_positions": sum(bool(account.get("position")) for account in accounts),
                "closed_trades": sum(
                    int(account.get("observation_trade_count", 0)) for account in accounts
                ),
            }
        )
        if accounts and network_errors == len(accounts):
            failure_cycles = int(state.get("network_failure_cycles", 0)) + 1
            retry_seconds = self._network_retry_delay(failure_cycles)
            first_pause = previous_status != "paused-network"
            state.update(
                {
                    "status": "paused-network",
                    "connection_status": "offline",
                    "entries_allowed": 0,
                    "network_failure_cycles": failure_cycles,
                    "network_pause_count": int(state.get("network_pause_count", 0))
                    + int(first_pause),
                    "network_paused_at": (
                        now.isoformat() if first_pause else state.get("network_paused_at")
                    ),
                    "retry_seconds": retry_seconds,
                    "next_retry_at": (now + timedelta(seconds=retry_seconds)).isoformat(),
                    "pause_reason": "Live quote connection unavailable",
                    "reconciliation": "waiting-for-network-no-new-entries",
                }
            )
            for account in accounts:
                account.update(
                    {
                        "status": "paused-network",
                        "entries_allowed": False,
                        "pause_reason": "network-unavailable",
                        "retry_seconds": retry_seconds,
                        "next_retry_at": state["next_retry_at"],
                    }
                )
                self.store.set_state(paper_lab_account_key(str(account["candidate_hash"])), account)
        else:
            recovered = previous_status == "paused-network"
            state.update(
                {
                    "status": "running",
                    "connection_status": "online" if healthy_feeds else "checking",
                    "network_failure_cycles": 0,
                    "retry_seconds": None,
                    "next_retry_at": None,
                    "pause_reason": None,
                }
            )
            if recovered:
                state.update(
                    {
                        "resumed_at": now.isoformat(),
                        "reconciliation": (
                            "network-restored-backfill-checked-no-retroactive-trades"
                        ),
                    }
                )
            for account in accounts:
                if account.get("feed_status") == "healthy":
                    account["status"] = "running"
                    account.pop("pause_reason", None)
                    account.pop("retry_seconds", None)
                    account.pop("next_retry_at", None)
                    self.store.set_state(
                        paper_lab_account_key(str(account["candidate_hash"])), account
                    )
        self.store.set_state(PAPER_LAB_STATE, state)
        return state

    def stop(self, *, killed: bool = False, reason: str = "operator-stop") -> dict[str, Any]:
        for candidate_hash, engine in list(self.engines.items()):
            account = self.store.get_state(paper_lab_account_key(candidate_hash)) or {}
            engine.stop(account, reason, killed=killed)
        self.engines = {}
        self.shared_provider = None
        self.sleep_lease.release()
        state = self.store.get_state(PAPER_LAB_STATE) or {}
        state.update(
            {
                "status": "killed" if killed else "stopped",
                "sleep_lease": False,
                "stopped_at": datetime.now(UTC).isoformat(),
                "stop_reason": reason,
            }
        )
        self.store.set_state(PAPER_LAB_STATE, state)
        return state

    def status(self) -> dict[str, Any]:
        state = self.store.get_state(PAPER_LAB_STATE) or {
            "status": "stopped",
            "strategy_count": 0,
            "sleep_lease": False,
            "simulated_fills_only": True,
        }
        accounts = [
            self.store.get_state(paper_lab_account_key(candidate_hash)) or {}
            for candidate_hash in state.get("candidate_hashes", [])
        ]
        return {**state, "accounts": accounts}

    def restore_if_needed(self) -> None:
        state = self.store.get_state(PAPER_LAB_STATE) or {}
        if state.get("status") not in {"running", "paused-network"}:
            return
        hashes = [str(value) for value in state.get("candidate_hashes", [])]
        try:
            self.start(hashes, resume=True)
            restored = self.store.get_state(PAPER_LAB_STATE) or {}
            restored["reconciliation"] = "worker-restored-no-duplicate-fills"
            self.store.set_state(PAPER_LAB_STATE, restored)
        except Exception as exc:
            state.update(
                {
                    "status": "recovery-required",
                    "sleep_lease": False,
                    "reconciliation": f"restore-failed:{type(exc).__name__}",
                }
            )
            self.store.set_state(PAPER_LAB_STATE, state)

    def run_forever(self, poll_seconds: float = 0.5) -> None:
        def shutdown(*_args: Any) -> None:
            self.running = False
            self.sleep_lease.release()

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        self.store.recover_processing_commands(PAPER_LAB_COMMAND_PREFIX)
        self.restore_if_needed()
        while self.running:
            command = self.store.claim_next_command(PAPER_LAB_COMMAND_PREFIX)
            if command is not None:
                try:
                    if command["type"] == "paperlab.start":
                        result = self.start(
                            [str(value) for value in command["payload"]["candidate_hashes"]]
                        )
                    elif command["type"] == "paperlab.stop":
                        result = self.stop()
                    elif command["type"] == "paperlab.kill":
                        result = self.stop(
                            killed=True,
                            reason=str(command["payload"].get("reason", "operator-kill")),
                        )
                    else:
                        raise ValueError(f"unsupported paper lab command: {command['type']}")
                    self.store.complete_command(command["id"], result)
                except Exception as exc:
                    self.sleep_lease.release()
                    error = {"error": str(exc), "type": type(exc).__name__}
                    state = self.store.get_state(PAPER_LAB_STATE) or {}
                    state.update(
                        {
                            "status": "error",
                            "sleep_lease": False,
                            "last_error": error,
                        }
                    )
                    self.store.set_state(PAPER_LAB_STATE, state)
                    self.store.complete_command(command["id"], error, failed=True)

            state = self.store.get_state(PAPER_LAB_STATE) or {}
            if (
                self.engines
                and state.get("status") in {"running", "paused-network"}
                and time.monotonic() >= self.next_poll
            ):
                updated = self.poll_once()
                if updated.get("status") == "paused-network":
                    delay = float(
                        updated.get(
                            "retry_seconds",
                            self.settings.live_paper.reconnect_initial_seconds,
                        )
                    )
                else:
                    accounts = self.status()["accounts"]
                    delay = max(
                        (
                            float(
                                account.get(
                                    "next_retry_seconds",
                                    self.settings.live_paper.poll_seconds,
                                )
                            )
                            for account in accounts
                        ),
                        default=self.settings.live_paper.poll_seconds,
                    )
                self.next_poll = time.monotonic() + delay
            time.sleep(poll_seconds)


def paper_lab_worker_main() -> None:
    from gold_trader.config import get_settings

    settings = get_settings()
    store = Store(settings.database_url)
    store.initialize()
    PaperLabWorker(settings, store).run_forever()
