"""Add Generation 3 manifest and trial checkpoint tables.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    from gold_trader.database import GenerationRow, TrialRow

    bind = op.get_bind()
    GenerationRow.__table__.create(bind=bind, checkfirst=True)
    TrialRow.__table__.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    op.drop_table("generation_trials")
    op.drop_table("generations")
