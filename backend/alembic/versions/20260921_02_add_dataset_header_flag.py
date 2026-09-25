"""Add dataset header detection metadata.

Revision ID: 20260921_02
Revises: 20260921_01
Create Date: 2026-09-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260921_02"
down_revision: str | None = "20260921_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "datasets",
        sa.Column(
            "has_header",
            sa.Boolean(),
            server_default=sa.true(),
            nullable=False,
        ),
    )
    op.alter_column("datasets", "has_header", server_default=None)


def downgrade() -> None:
    op.drop_column("datasets", "has_header")
