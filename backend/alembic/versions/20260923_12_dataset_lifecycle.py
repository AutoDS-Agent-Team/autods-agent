"""add dataset lifecycle metadata

Revision ID: 20260923_12
Revises: 20260923_11
"""
from alembic import op
import sqlalchemy as sa

revision = "20260923_12"
down_revision = "20260923_11"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("status", sa.String(length=32), nullable=False, server_default="READY"))
    op.add_column("datasets", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")))
    op.alter_column("datasets", "status", server_default=None)
    op.alter_column("datasets", "updated_at", server_default=None)


def downgrade() -> None:
    op.drop_column("datasets", "updated_at")
    op.drop_column("datasets", "status")
