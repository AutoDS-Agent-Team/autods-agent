"""Create model runs table.

Revision ID: 20260922_06
Revises: 20260922_05
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260922_06"
down_revision: str | None = "20260922_05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("pipeline_plan_id", sa.String(length=36), nullable=False),
        sa.Column("model_name", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("training_duration_seconds", sa.Float(), nullable=False),
        sa.Column("failure_information", sa.Text(), nullable=True),
        sa.Column("artifact_filename", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["pipeline_plan_id"], ["pipeline_plans.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_model_runs_experiment_id"),
        "model_runs",
        ["experiment_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_model_runs_experiment_id"), table_name="model_runs")
    op.drop_table("model_runs")
