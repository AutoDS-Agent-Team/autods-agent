"""Mark legacy dataset header state as unknown.

Revision ID: 20260921_03
Revises: 20260921_02
Create Date: 2026-09-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260921_03"
down_revision: str | None = "20260921_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "datasets",
        "has_header",
        existing_type=sa.Boolean(),
        nullable=True,
    )
    op.execute(sa.text("UPDATE datasets SET has_header = NULL"))


def downgrade() -> None:
    op.execute(sa.text("UPDATE datasets SET has_header = TRUE WHERE has_header IS NULL"))
    op.alter_column(
        "datasets",
        "has_header",
        existing_type=sa.Boolean(),
        nullable=False,
    )
