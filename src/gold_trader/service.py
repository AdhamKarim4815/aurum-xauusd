from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from gold_trader.config import Settings, get_settings
from gold_trader.database import Store
from gold_trader.domain import Approval, KillSwitchEvent
from gold_trader.generation import GenerationCoordinator, load_sealed_manifest
from gold_trader.generation4 import (
    Generation4Coordinator,
    load_generation4_manifest,
)
from gold_trader.generation5 import (
    Generation5Coordinator,
    load_generation5_manifest,
)
from gold_trader.generation6 import (
    Generation6Coordinator,
    load_generation6_manifest,
)
from gold_trader.paper_lab import (
    PAPER_LAB_STATE,
    paper_lab_account_key,
    validate_experimental_report,
)
from gold_trader.research import stable_hash
from gold_trader.secrets import get_secret


def _validation_state_key(trial_hash: str) -> str:
    return f"validation:{trial_hash}"


class ApplicationService:
    def __init__(self, store: Store, settings: Settings | None = None) -> None:
        self.store = store
        self.settings = settings or get_settings()

    def status(self) -> dict[str, Any]:
        paper = self.store.get_state("paper") or {}
        approval = self.store.valid_approval()
        runs = self.store.list_research_runs(limit=5)
        generation = self.store.get_generation()
        compact_runs = [
            {
                key: run.get(key)
                for key in (
                    "strategy_id",
                    "variant_id",
                    "timeframe",
                    "candidate_hash",
                    "status",
                    "eligible",
                    "metrics",
                )
            }
            for run in runs
        ]
        compact_generation = None
        if generation is not None:
            compact_generation = {
                key: generation.get(key)
                for key in ("generation_id", "manifest_hash", "status", "summary", "updated_at")
            }
        provider_chain = [
            {"name": name, "configured": bool(get_secret(name))}
            for name in self.settings.live_paper.provider_priority
        ]
        return {
            "paper": paper,
            "approval": approval,
            "research_runs": compact_runs,
            "generation": compact_generation,
            "eligible_candidates": sum(1 for run in runs if run.get("eligible")),
            "live_execution_available": False,
            "continuous_live_paper_available": True,
            "equity_history": self.store.list_account_snapshots(limit=300),
            "provider_chain": provider_chain,
            "live_paper_stale_after_seconds": self.settings.live_paper.stale_after_seconds,
        }

    def approve_candidate(self, candidate_hash: str, confirmation: str | None = None) -> Approval:
        if confirmation is not None and confirmation != candidate_hash:
            raise ValueError("typed candidate hash does not match")
        run = self.store.get_research_run(candidate_hash)
        if run is None:
            raise ValueError("unknown candidate")
        if not run.get("eligible"):
            raise ValueError("candidate did not pass the robustness gate")
        approval = Approval(
            candidate_hash=candidate_hash,
            dataset_hash=run["dataset_hash"],
            configuration_hash=run["configuration_hash"],
            strategy_hash=run["strategy_hash"],
            model_hash=run["model_hash"],
        )
        self.store.create_approval(approval)
        return approval

    def start_paper(self, mode: str = "replay") -> int:
        approval = self.store.valid_approval()
        if approval is None:
            raise ValueError("a valid eligible candidate approval is required")
        paper = self.store.get_state("paper") or {}
        if paper.get("kill_switch"):
            raise ValueError("kill switch is active")
        if mode not in {"replay", "live"}:
            raise ValueError("mode must be replay or live")
        report = self.store.get_research_run(str(approval["candidate_hash"]))
        prefix = "paper6." if report and report.get("generation_six") else "paper."
        return self.store.enqueue_command(
            f"{prefix}start",
            {"mode": mode, "candidate_hash": approval["candidate_hash"]},
            f"start:{approval['candidate_hash']}:{mode}:{uuid4()}",
        )

    def stop_paper(self) -> int:
        paper = self.store.get_state("paper") or {}
        prefix = "paper6." if paper.get("worker_prefix") == "paper6." else "paper."
        return self.store.enqueue_command(f"{prefix}stop", {}, f"stop:{uuid4()}")

    def activate_kill_switch(self, reason: str) -> int:
        event = KillSwitchEvent(active=True, reason=reason or "operator activated")
        self.store.append_event(event)
        paper = self.store.get_state("paper") or {}
        paper.update({"kill_switch": True, "status": "killed", "reason": event.reason})
        self.store.set_state("paper", paper)
        prefix = "paper6." if paper.get("worker_prefix") == "paper6." else "paper."
        command_id = self.store.enqueue_command(
            f"{prefix}kill", {"reason": event.reason}, f"kill:{event.event_id}"
        )
        lab = self.store.get_state(PAPER_LAB_STATE) or {}
        if lab.get("status") == "running":
            self.store.enqueue_command(
                "paperlab.kill",
                {"reason": event.reason},
                f"paperlab-kill:{event.event_id}",
            )
        return command_id

    def reset_kill_switch(self) -> None:
        paper = self.store.get_state("paper") or {}
        if paper.get("status") not in {"stopped", "killed"}:
            raise ValueError("paper worker must be stopped before resetting the kill switch")
        lab = self.store.get_state(PAPER_LAB_STATE) or {}
        if lab.get("status") == "running":
            raise ValueError("four-strategy paper lab must be stopped before resetting kill")
        event = KillSwitchEvent(active=False, reason="operator reset after review")
        self.store.append_event(event)
        paper.update({"kill_switch": False, "status": "stopped", "reason": None})
        self.store.set_state("paper", paper)

    def paper_lab_status(self) -> dict[str, Any]:
        state = self.store.get_state(PAPER_LAB_STATE) or {
            "status": "stopped",
            "strategy_count": 0,
            "sleep_lease": False,
            "simulated_fills_only": True,
        }
        accounts = [
            self.store.get_state(paper_lab_account_key(str(candidate_hash))) or {}
            for candidate_hash in state.get("candidate_hashes", [])
        ]
        return {**state, "accounts": accounts}

    def start_paper_lab(self, candidate_hashes: list[str], *, acknowledged: bool) -> int:
        if not acknowledged:
            raise ValueError("paper-only experimental-risk acknowledgement is required")
        unique = list(dict.fromkeys(candidate_hashes))
        if len(unique) != 4:
            raise ValueError("select exactly four unique finalized strategies")
        paper = self.store.get_state("paper") or {}
        if paper.get("kill_switch"):
            raise ValueError("global kill switch is active")
        if paper.get("status") in {"running", "risk-halted"}:
            raise ValueError("stop the normal paper account before starting the strategy lab")
        lab = self.store.get_state(PAPER_LAB_STATE) or {}
        if lab.get("status") == "running":
            raise ValueError("four-strategy paper lab is already running")
        for candidate_hash in unique:
            report = self.store.get_research_run(candidate_hash)
            if report is None:
                raise ValueError(f"unknown finalized candidate: {candidate_hash}")
            validate_experimental_report(report)
        return self.store.enqueue_command(
            "paperlab.start",
            {"candidate_hashes": unique},
            f"paperlab-start:{stable_hash(unique)}:{uuid4()}",
        )

    def stop_paper_lab(self) -> int:
        return self.store.enqueue_command("paperlab.stop", {}, f"paperlab-stop:{uuid4()}")

    def kill_paper_lab(self, reason: str) -> int:
        return self.store.enqueue_command(
            "paperlab.kill",
            {"reason": reason or "operator activated lab kill"},
            f"paperlab-kill:{uuid4()}",
        )

    def start_generation(self, manifest_path: Path) -> int:
        manifest = load_sealed_manifest(manifest_path.resolve())
        current = self.store.get_generation(manifest.generation_id)
        if current and current["status"] == "running":
            raise ValueError("Generation 3 is already running")
        GenerationCoordinator(self.settings, self.store).request_control("run")
        return self.store.enqueue_command(
            "research.generation.run",
            {
                "manifest_path": str(manifest_path.resolve()),
                "generation_id": manifest.generation_id,
                "manifest_hash": manifest.manifest_hash,
            },
            f"generation:{manifest.manifest_hash}:{uuid4()}",
        )

    def control_generation(self, action: str) -> None:
        if action == "pause":
            GenerationCoordinator(self.settings, self.store).request_control("pause")
        elif action == "stop":
            GenerationCoordinator(self.settings, self.store).request_control("stop")
        else:
            raise ValueError("generation action must be pause or stop")

    def start_generation4(self, manifest_path: Path) -> int:
        manifest = load_generation4_manifest(manifest_path.resolve())
        current = self.store.get_generation(manifest.generation_id)
        if current and current["status"] == "running":
            raise ValueError("Generation 4 is already running")
        Generation4Coordinator(self.settings, self.store).request_control("run")
        return self.store.enqueue_command(
            "generation4.run",
            {
                "manifest_path": str(manifest_path.resolve()),
                "generation_id": manifest.generation_id,
                "manifest_hash": manifest.manifest_hash,
            },
            f"generation4:{manifest.manifest_hash}:{uuid4()}",
        )

    def control_generation4(self, action: str) -> None:
        coordinator = Generation4Coordinator(self.settings, self.store)
        if action == "pause":
            coordinator.request_control("pause")
        elif action == "stop":
            coordinator.request_control("stop")
        else:
            raise ValueError("Generation 4 action must be pause or stop")

    def start_generation5(self, manifest_path: Path) -> int:
        manifest = load_generation5_manifest(manifest_path.resolve())
        current = self.store.get_generation(manifest.generation_id)
        if current and current["status"] == "running":
            raise ValueError("Generation 5 is already running")
        Generation5Coordinator(self.settings, self.store).request_control("run")
        return self.store.enqueue_command(
            "generation5.run",
            {
                "manifest_path": str(manifest_path.resolve()),
                "generation_id": manifest.generation_id,
                "manifest_hash": manifest.manifest_hash,
            },
            f"generation5:{manifest.manifest_hash}:{uuid4()}",
        )

    def control_generation5(self, action: str) -> None:
        coordinator = Generation5Coordinator(self.settings, self.store)
        if action == "pause":
            coordinator.request_control("pause")
        elif action == "stop":
            coordinator.request_control("stop")
        else:
            raise ValueError("Generation 5 action must be pause or stop")

    def start_generation6(self, manifest_path: Path) -> int:
        manifest = load_generation6_manifest(manifest_path.resolve())
        current = self.store.get_generation(manifest.generation_id)
        if current and current["status"] == "running":
            raise ValueError("Generation 6 is already running")
        Generation6Coordinator(self.settings, self.store).request_control("run")
        return self.store.enqueue_command(
            "generation6.run",
            {
                "manifest_path": str(manifest_path.resolve()),
                "generation_id": manifest.generation_id,
                "manifest_hash": manifest.manifest_hash,
            },
            f"generation6:{manifest.manifest_hash}:{uuid4()}",
        )

    def control_generation6(self, action: str) -> None:
        coordinator = Generation6Coordinator(self.settings, self.store)
        if action == "pause":
            coordinator.request_control("pause")
        elif action == "stop":
            coordinator.request_control("stop")
        else:
            raise ValueError("Generation 6 action must be pause or stop")

    def finalize_strategy_choice(
        self, generation_id: str, trial_hash: str, external_dataset: Path
    ) -> dict[str, Any]:
        generation = self.store.get_generation(generation_id)
        if generation is None:
            raise ValueError("unknown strategy generation")
        if generation["status"] != "completed":
            raise ValueError("the generation must finish before final validation")
        manifest_path = Path(str(generation["summary"].get("manifest_path", "")))
        if not manifest_path.is_absolute():
            manifest_path = self.settings.project_root / manifest_path
        if generation_id.startswith("generation-6"):
            return Generation6Coordinator(self.settings, self.store).finalize(
                manifest_path, trial_hash, external_dataset
            )
        if generation_id.startswith("generation-5"):
            return Generation5Coordinator(self.settings, self.store).finalize(
                manifest_path, trial_hash, external_dataset
            )
        if generation_id.startswith("generation-4"):
            return Generation4Coordinator(self.settings, self.store).finalize(
                manifest_path, trial_hash, external_dataset
            )
        return GenerationCoordinator(self.settings, self.store).finalize(
            manifest_path, trial_hash, external_dataset
        )

    def queue_strategy_validation(
        self, generation_id: str, trial_hash: str, external_dataset: Path
    ) -> int:
        """Queue final validation and persist its operator-visible progress state."""
        generation = self.store.get_generation(generation_id)
        if generation is None:
            raise ValueError("unknown strategy generation")
        if generation["status"] != "completed":
            raise ValueError("the generation must finish before final validation")
        selected = next(
            (
                item
                for item in self.store.list_trials(generation_id)
                if item["spec"].get("trial_hash") == trial_hash
            ),
            None,
        )
        if selected is None or selected["result"].get("status") != "shortlisted":
            raise ValueError("only a completed shortlist can be finalized")

        data_root = (self.settings.project_root / "var/data").resolve()
        external = external_dataset.resolve()
        if (
            not external.is_file()
            or external.parent != data_root
            or external.suffix.lower() not in {".csv", ".parquet"}
        ):
            raise ValueError("external validation dataset must be a CSV or Parquet in var/data")

        state_key = _validation_state_key(trial_hash)
        current = self.store.get_state(state_key) or {}
        if current.get("status") in {"queued", "running"}:
            return int(current["command_id"])

        for run in self.store.list_research_runs(limit=500):
            evidence = (
                run.get("generation_six")
                or run.get("generation_five")
                or run.get("generation_four")
                or run.get("generation_three")
                or {}
            )
            finalized_trial = evidence.get("trial_spec", {}).get("trial_hash")
            if finalized_trial == trial_hash:
                self.store.set_state(
                    state_key,
                    {
                        "generation_id": generation_id,
                        "trial_hash": trial_hash,
                        "external_dataset": str(external),
                        "status": "completed",
                        "stage": "Final validation complete",
                        "progress": 100,
                        "candidate_hash": run["candidate_hash"],
                        "eligible": bool(run.get("eligible")),
                        "failed_gates": [
                            name for name, passed in run.get("gates", {}).items() if not passed
                        ],
                        "report_path": str(
                            self.settings.project_root
                            / "var/reports"
                            / f"{run['candidate_hash']}.json"
                        ),
                    },
                )
                return 0

        payload = {
            "generation_id": generation_id,
            "trial_hash": trial_hash,
            "external_dataset": str(external),
        }
        command_id = self.store.enqueue_command(
            "validation.finalize", payload, f"validation:{trial_hash}:{uuid4()}"
        )
        self.store.set_state(
            state_key,
            {
                **payload,
                "command_id": command_id,
                "status": "queued",
                "stage": "Waiting for the validation worker",
                "progress": 0,
            },
        )
        return command_id
