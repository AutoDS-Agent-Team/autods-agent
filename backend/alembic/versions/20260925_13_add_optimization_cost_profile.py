"""Persist measured optimization search-cost details.

Revision ID: 20260925_13
Revises: 20260923_12
"""
from alembic import op
import sqlalchemy as sa

revision = "20260925_13"
down_revision = "20260923_12"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("optimization_results", sa.Column("cost_profile", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("optimization_results", "cost_profile")
