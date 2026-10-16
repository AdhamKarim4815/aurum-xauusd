from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    event,
    func,
    select,
    update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from gold_trader.domain import Approval, CommandStatus, DomainRecord


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(120), index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    correlation_id: Mapped[str] = mapped_column(String(64), index=True)
    causation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[str] = mapped_column(Text)


class CommandRow(Base):
    __tablename__ = "commands"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    command_type: Mapped[str] = mapped_column(String(80), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class StateRow(Base):
    __tablename__ = "app_state"

    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ApprovalRow(Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    candidate_hash: Mapped[str] = mapped_column(String(128), index=True)
    dataset_hash: Mapped[str] = mapped_column(String(128))
    configuration_hash: Mapped[str] = mapped_column(String(128))
    strategy_hash: Mapped[str] = mapped_column(String(128))
    model_hash: Mapped[str] = mapped_column(String(128))
    valid: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    approved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ResearchRunRow(Base):
    __tablename__ = "research_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    strategy_id: Mapped[str] = mapped_column(String(120), index=True)
    candidate_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    dataset_hash: Mapped[str] = mapped_column(String(128))
    configuration_hash: Mapped[str] = mapped_column(String(128))
    strategy_hash: Mapped[str] = mapped_column(String(128))
    model_hash: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(24), index=True)
    eligible: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    metrics_json: Mapped[str] = mapped_column(Text)
    report_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class GenerationRow(Base):
    __tablename__ = "generations"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    manifest_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    manifest_json: Mapped[str] = mapped_column(Text)
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TrialRow(Base):
    __tablename__ = "generation_trials"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    generation_id: Mapped[str] = mapped_column(String(80), index=True)
    family: Mapped[str] = mapped_column(String(80), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), index=True)
    score: Mapped[str] = mapped_column(String(64), default="0")
    spec_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Store:
    def __init__(self, database_url: str) -> None:
        if database_url.startswith("sqlite:///"):
            path = Path(database_url.removeprefix("sqlite:///"))
            path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(database_url, future=True)
        if database_url.startswith("sqlite"):
            event.listen(self.engine, "connect", self._configure_sqlite)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    @staticmethod
    def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)
        if self.get_state("paper") is None:
            self.set_state(
                "paper",
                {
                    "status": "stopped",
                    "kill_switch": False,
                    "feed_status": "unconfigured",
                    "sleep_lease": False,
                },
            )

    def append_event(self, record: DomainRecord) -> None:
        with self.sessions.begin() as session:
            session.add(
                EventRow(
                    id=record.event_id,
                    event_type=record.__class__.__name__,
                    occurred_at=record.occurred_at,
                    correlation_id=record.correlation_id,
                    causation_id=record.causation_id,
                    payload_json=record.model_dump_json(),
                )
            )

    def list_account_snapshots(self, limit: int = 300) -> list[dict[str, Any]]:
        """Return the most recent equity observations in chronological order."""
        with self.sessions() as session:
            rows = session.scalars(
                select(EventRow)
                .where(EventRow.event_type == "AccountSnapshot")
                .order_by(EventRow.occurred_at.desc())
                .limit(limit)
            ).all()
        return [
            {
                "occurred_at": row.occurred_at.isoformat(),
                "equity": json.loads(row.payload_json)["equity"],
            }
            for row in reversed(rows)
        ]

    def audit(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.sessions() as session:
            rows = session.scalars(
                select(EventRow).order_by(EventRow.occurred_at.desc()).limit(limit)
            ).all()
        return [
            {
                "id": row.id,
                "type": row.event_type,
                "occurred_at": row.occurred_at,
                "payload": json.loads(row.payload_json),
            }
            for row in rows
        ]

    def enqueue_command(
        self, command_type: str, payload: dict[str, Any], idempotency_key: str
    ) -> int:
        with self.sessions.begin() as session:
            existing = session.scalar(
                select(CommandRow).where(CommandRow.idempotency_key == idempotency_key)
            )
            if existing is not None:
                return existing.id
            row = CommandRow(
                command_type=command_type,
                idempotency_key=idempotency_key,
                status=CommandStatus.PENDING.value,
                payload_json=json.dumps(payload, sort_keys=True),
            )
            session.add(row)
            session.flush()
            return row.id

    def claim_next_command(self, prefix: str | None = None) -> dict[str, Any] | None:
        with self.sessions.begin() as session:
            statement = select(CommandRow).where(CommandRow.status == CommandStatus.PENDING.value)
            if prefix is not None:
                statement = statement.where(CommandRow.command_type.like(f"{prefix}%"))
            row = session.scalar(statement.order_by(CommandRow.id.asc()).limit(1))
            if row is None:
                return None
            row.status = CommandStatus.PROCESSING.value
            row.processed_at = utc_now()
            return {
                "id": row.id,
                "type": row.command_type,
                "payload": json.loads(row.payload_json),
            }

    def complete_command(
        self, command_id: int, result: dict[str, Any], failed: bool = False
    ) -> None:
        with self.sessions.begin() as session:
            row = session.get(CommandRow, command_id)
            if row is None:
                raise KeyError(f"unknown command {command_id}")
            row.status = CommandStatus.FAILED.value if failed else CommandStatus.COMPLETED.value
            row.result_json = json.dumps(result, sort_keys=True)
            row.processed_at = utc_now()

    def recover_processing_commands(self, prefix: str) -> int:
        """Return interrupted commands to their isolated queue on worker startup."""
        with self.sessions.begin() as session:
            rows = session.scalars(
                select(CommandRow).where(
                    CommandRow.status == CommandStatus.PROCESSING.value,
                    CommandRow.command_type.like(f"{prefix}%"),
                )
            ).all()
            for row in rows:
                row.status = CommandStatus.PENDING.value
                row.processed_at = None
            return len(rows)

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        with self.sessions.begin() as session:
            row = session.get(StateRow, key)
            if row is None:
                session.add(StateRow(key=key, value_json=json.dumps(value), updated_at=utc_now()))
            else:
                row.value_json = json.dumps(value)
                row.updated_at = utc_now()

    def get_state(self, key: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.get(StateRow, key)
            return json.loads(row.value_json) if row is not None else None

    def list_states(self, prefix: str) -> list[dict[str, Any]]:
        """Return durable application states whose keys share an isolated prefix."""
        with self.sessions() as session:
            rows = session.scalars(
                select(StateRow)
                .where(StateRow.key.like(f"{prefix}%"))
                .order_by(StateRow.updated_at.desc())
            ).all()
        return [
            {
                "key": row.key,
                "updated_at": row.updated_at.isoformat(),
                **json.loads(row.value_json),
            }
            for row in rows
        ]

    def save_research_run(self, report: dict[str, Any]) -> None:
        with self.sessions.begin() as session:
            row = ResearchRunRow(
                id=report["run_id"],
                strategy_id=report["strategy_id"],
                candidate_hash=report["candidate_hash"],
                dataset_hash=report["dataset_hash"],
                configuration_hash=report["configuration_hash"],
                strategy_hash=report["strategy_hash"],
                model_hash=report["model_hash"],
                status=report["status"],
                eligible=report["eligible"],
                metrics_json=json.dumps(report["metrics"], sort_keys=True),
                report_json=json.dumps(report, sort_keys=True, default=str),
            )
            session.merge(row)

    def list_research_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.sessions() as session:
            rows = session.scalars(
                select(ResearchRunRow).order_by(ResearchRunRow.created_at.desc()).limit(limit)
            ).all()
        return [json.loads(row.report_json) for row in rows]

    def get_research_run(self, candidate_hash: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.scalar(
                select(ResearchRunRow).where(ResearchRunRow.candidate_hash == candidate_hash)
            )
            return json.loads(row.report_json) if row is not None else None

    def create_approval(self, approval: Approval) -> None:
        with self.sessions.begin() as session:
            session.execute(
                update(ApprovalRow)
                .where(ApprovalRow.valid.is_(True))
                .values(valid=False, invalidated_at=utc_now())
            )
            session.add(
                ApprovalRow(
                    id=approval.event_id,
                    candidate_hash=approval.candidate_hash,
                    dataset_hash=approval.dataset_hash,
                    configuration_hash=approval.configuration_hash,
                    strategy_hash=approval.strategy_hash,
                    model_hash=approval.model_hash,
                    valid=approval.valid,
                    approved_at=approval.approved_at,
                )
            )
        self.append_event(approval)

    def valid_approval(self) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = session.scalar(
                select(ApprovalRow)
                .where(ApprovalRow.valid.is_(True))
                .order_by(ApprovalRow.approved_at.desc())
                .limit(1)
            )
            if row is None:
                return None
            return {
                "candidate_hash": row.candidate_hash,
                "dataset_hash": row.dataset_hash,
                "configuration_hash": row.configuration_hash,
                "strategy_hash": row.strategy_hash,
                "model_hash": row.model_hash,
                "approved_at": row.approved_at.isoformat(),
            }

    def invalidate_approvals(self) -> int:
        with self.sessions.begin() as session:
            rows = session.scalars(select(ApprovalRow).where(ApprovalRow.valid.is_(True))).all()
            for row in rows:
                row.valid = False
                row.invalidated_at = utc_now()
            return len(rows)

    def save_generation(
        self,
        generation_id: str,
        manifest_hash: str,
        status: str,
        manifest: dict[str, Any],
        summary: dict[str, Any] | None = None,
    ) -> None:
        with self.sessions.begin() as session:
            row = session.get(GenerationRow, generation_id)
            if row is None:
                session.add(
                    GenerationRow(
                        id=generation_id,
                        manifest_hash=manifest_hash,
                        status=status,
                        manifest_json=json.dumps(manifest, sort_keys=True, default=str),
                        summary_json=json.dumps(summary or {}, sort_keys=True, default=str),
                    )
                )
            else:
                if row.manifest_hash != manifest_hash:
                    trial_count = session.scalar(
                        select(func.count())
                        .select_from(TrialRow)
                        .where(TrialRow.generation_id == generation_id)
                    )
                    if trial_count or row.status == "running":
                        raise ValueError("a generation with durable trials cannot be replaced")
                    row.manifest_hash = manifest_hash
                    row.manifest_json = json.dumps(manifest, sort_keys=True, default=str)
                row.status = status
                row.summary_json = json.dumps(summary or {}, sort_keys=True, default=str)
                row.updated_at = utc_now()

    def get_generation(self, generation_id: str | None = None) -> dict[str, Any] | None:
        with self.sessions() as session:
            row = (
                session.get(GenerationRow, generation_id)
                if generation_id is not None
                else session.scalar(
                    select(GenerationRow).order_by(GenerationRow.updated_at.desc()).limit(1)
                )
            )
            if row is None:
                return None
            return {
                "generation_id": row.id,
                "manifest_hash": row.manifest_hash,
                "status": row.status,
                "manifest": json.loads(row.manifest_json),
                "summary": json.loads(row.summary_json),
                "updated_at": row.updated_at.isoformat(),
            }

    def list_generations(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.sessions() as session:
            rows = session.scalars(
                select(GenerationRow).order_by(GenerationRow.updated_at.desc()).limit(limit)
            ).all()
        return [
            {
                "generation_id": row.id,
                "manifest_hash": row.manifest_hash,
                "status": row.status,
                "summary": json.loads(row.summary_json),
                "updated_at": row.updated_at.isoformat(),
            }
            for row in rows
        ]

    def interrupt_running_generations(self, reason: str) -> int:
        """Mark orphaned runs interrupted before a research worker recovers its queue."""
        interrupted_at = utc_now()
        with self.sessions.begin() as session:
            rows = session.scalars(
                select(GenerationRow).where(GenerationRow.status == "running")
            ).all()
            for row in rows:
                summary = json.loads(row.summary_json)
                summary.update(
                    {
                        "interrupted_at": interrupted_at.isoformat(),
                        "interruption_reason": reason,
                    }
                )
                row.status = "interrupted"
                row.summary_json = json.dumps(summary, sort_keys=True, default=str)
                row.updated_at = interrupted_at
            return len(rows)

    def save_trial(
        self,
        generation_id: str,
        spec: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        trial_id = str(spec["trial_hash"])
        with self.sessions.begin() as session:
            row = TrialRow(
                id=trial_id,
                generation_id=generation_id,
                family=str(spec["family"]),
                ordinal=int(spec["ordinal"]),
                status=str(result.get("status", "completed")),
                score=str(result.get("score", 0.0)),
                spec_json=json.dumps(spec, sort_keys=True, default=str),
                result_json=json.dumps(result, sort_keys=True, default=str),
                completed_at=utc_now(),
            )
            session.merge(row)

    def list_trials(
        self,
        generation_id: str | None = None,
        limit: int = 100_000,
        *,
        sort_by_return: bool = False,
    ) -> list[dict[str, Any]]:
        if generation_id is None:
            generation = self.get_generation()
            if generation is None:
                return []
            generation_id = str(generation["generation_id"])
        with self.sessions() as session:
            statement = (
                select(TrialRow)
                .where(TrialRow.generation_id == generation_id)
                .order_by(TrialRow.ordinal.asc())
            )
            if not sort_by_return:
                statement = statement.limit(limit)
            rows = session.scalars(statement).all()
        items = [
            {"spec": json.loads(row.spec_json), "result": json.loads(row.result_json)}
            for row in rows
        ]
        if sort_by_return:
            items.sort(
                key=lambda item: float(
                    item["result"].get("metrics", {}).get("net_return", float("-inf"))
                ),
                reverse=True,
            )
            return items[:limit]
        return items
