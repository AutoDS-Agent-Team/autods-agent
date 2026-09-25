"""add Excel dataset metadata

Revision ID: 20260923_11
Revises: 20260922_10
"""
from alembic import op
import sqlalchemy as sa

revision = "20260923_11"
down_revision = "20260922_10"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("source_format", sa.String(length=16), nullable=False, server_default="csv"))
    op.add_column("datasets", sa.Column("worksheet_name", sa.String(length=255), nullable=True))
    op.alter_column("datasets", "source_format", server_default=None)


def downgrade() -> None:
    op.drop_column("datasets", "worksheet_name")
    op.drop_column("datasets", "source_format")
