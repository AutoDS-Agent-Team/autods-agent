"""Create experiments table.

Revision ID: 20260922_04
Revises: 20260921_03
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260922_04"
down_revision: str | None = "20260921_03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("dataset_id", sa.String(length=36), nullable=False),
        sa.Column("user_objective", sa.Text(), nullable=False),
        sa.Column("detected_task_type", sa.String(length=32), nullable=True),
        sa.Column("suggested_target_column", sa.String(length=255), nullable=True),
        sa.Column("confirmed_task_type", sa.String(length=32), nullable=True),
        sa.Column("confirmed_target_column", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_experiments_dataset_id"),
        "experiments",
        ["dataset_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_experiments_dataset_id"), table_name="experiments")
    op.drop_table("experiments")
