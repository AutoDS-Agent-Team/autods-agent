"""Add evaluation, optimization, prediction, and report metadata.

Revision ID: 20260922_07
Revises: 20260922_06
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260922_07"
down_revision: str | None = "20260922_06"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("experiments", sa.Column("selected_model_run_id", sa.String(length=36), nullable=True))
    op.create_foreign_key("fk_experiments_selected_model_run", "experiments", "model_runs", ["selected_model_run_id"], ["id"], ondelete="SET NULL")
    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("selected_model_run_id", sa.String(length=36), nullable=False),
        sa.Column("primary_metric", sa.String(length=32), nullable=False),
        sa.Column("validation_comparison", sa.JSON(), nullable=False),
        sa.Column("final_test_metrics", sa.JSON(), nullable=False),
        sa.Column("final_test_confusion_matrix", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["selected_model_run_id"], ["model_runs.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("experiment_id"),
    )
    op.create_table(
        "optimization_results",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("baseline_model_run_id", sa.String(length=36), nullable=False),
        sa.Column("optimized_model_run_id", sa.String(length=36), nullable=True),
        sa.Column("model_name", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("trial_count", sa.Integer(), nullable=False),
        sa.Column("best_parameters", sa.JSON(), nullable=False),
        sa.Column("best_validation_score", sa.Float(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("failure_information", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["baseline_model_run_id"], ["model_runs.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["optimized_model_run_id"], ["model_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("experiment_id"),
    )
    op.create_table(
        "prediction_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("model_run_id", sa.String(length=36), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("artifact_filename", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_run_id"], ["model_runs.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_prediction_runs_experiment_id"), "prediction_runs", ["experiment_id"], unique=False)
    op.create_table(
        "reports",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("artifact_filename", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_reports_experiment_id"), "reports", ["experiment_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_reports_experiment_id"), table_name="reports")
    op.drop_table("reports")
    op.drop_index(op.f("ix_prediction_runs_experiment_id"), table_name="prediction_runs")
    op.drop_table("prediction_runs")
    op.drop_table("optimization_results")
    op.drop_table("evaluation_results")
    op.drop_constraint("fk_experiments_selected_model_run", "experiments", type_="foreignkey")
    op.drop_column("experiments", "selected_model_run_id")
