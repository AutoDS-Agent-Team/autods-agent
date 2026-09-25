"""Create bounded Ask AutoDS conversation context.

Revision ID: 20260925_14
Revises: 20260925_13
"""
from alembic import op
import sqlalchemy as sa


revision = "20260925_14"
down_revision = "20260925_13"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assistant_turns",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=36), nullable=True),
        sa.Column("experiment_id", sa.String(length=36), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("resolved_question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("evidence_type", sa.String(length=32), nullable=False),
        sa.Column("referenced_columns", sa.JSON(), nullable=False),
        sa.Column("analytical_topic", sa.String(length=255), nullable=True),
        sa.Column("source_types", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_assistant_turns_user_id", "assistant_turns", ["user_id"])
    op.create_index("ix_assistant_turns_conversation_id", "assistant_turns", ["conversation_id"])
    op.create_index("ix_assistant_turns_dataset_id", "assistant_turns", ["dataset_id"])
    op.create_index("ix_assistant_turns_experiment_id", "assistant_turns", ["experiment_id"])
    op.create_index("ix_assistant_turns_created_at", "assistant_turns", ["created_at"])


def downgrade() -> None:
    op.drop_table("assistant_turns")
