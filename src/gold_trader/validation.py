from __future__ import annotations

import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gold_trader.config import Settings
from gold_trader.database import Store
from gold_trader.service import ApplicationService


def validation_state_key(trial_hash: str) -> str:
    return f"validation:{trial_hash}"


class ValidationWorker:
    """Consume final-validation commands outside the dashboard request process."""

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.running = True

    def process_next(self) -> bool:
        command = self.store.claim_next_command("validation.")
        if command is None:
            return False
        payload = command["payload"]
        trial_hash = str(payload.get("trial_hash", ""))
        state_key = validation_state_key(trial_hash)
        started_at = datetime.now(UTC).isoformat()
        state = {
            **payload,
            "command_id": command["id"],
            "status": "running",
            "stage": "Running sealed holdout and independent-source simulations",
            "progress": 25,
            "started_at": started_at,
        }
        self.store.set_state(state_key, state)
        try:
            if command["type"] != "validation.finalize":
                raise ValueError(f"unsupported validation command: {command['type']}")
            report = ApplicationService(self.store, self.settings).finalize_strategy_choice(
                str(payload["generation_id"]),
                trial_hash,
                Path(str(payload["external_dataset"])),
            )
            candidate_hash = str(report["candidate_hash"])
            failed_gates = [name for name, passed in report.get("gates", {}).items() if not passed]
            result = {
                "candidate_hash": candidate_hash,
                "eligible": bool(report.get("eligible")),
                "failed_gates": failed_gates,
                "report_path": str(
                    self.settings.project_root / "var/reports" / f"{candidate_hash}.json"
                ),
            }
            self.store.set_state(
                state_key,
                {
                    **state,
                    **result,
                    "status": "completed",
                    "stage": "Final validation complete",
                    "progress": 100,
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            self.store.complete_command(command["id"], result)
        except Exception as exc:
            error = {"error": str(exc), "type": type(exc).__name__}
            self.store.set_state(
                state_key,
                {
                    **state,
                    **error,
                    "status": "failed",
                    "stage": "Final validation failed",
                    "progress": 100,
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            self.store.complete_command(command["id"], error, failed=True)
        return True

    def run_forever(self, poll_seconds: float = 0.5) -> None:
        def stop(*_args: Any) -> None:
            self.running = False

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        self.store.recover_processing_commands("validation.")
        while self.running:
            if not self.process_next():
                time.sleep(poll_seconds)


def validation_worker_main() -> None:
    from gold_trader.config import get_settings

    settings = get_settings()
    store = Store(settings.database_url)
    store.initialize()
    ValidationWorker(settings, store).run_forever()
