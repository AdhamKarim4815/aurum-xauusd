"""Initial audit, command, approval, research, and state tables."""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Runtime metadata and this migration intentionally share the same schema.
    from gold_trader.database import Base

    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    for table in ("research_runs", "approvals", "app_state", "commands", "events"):
        op.drop_table(table)
