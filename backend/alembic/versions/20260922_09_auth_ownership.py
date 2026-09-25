"""Add authentication users and ownership without deleting legacy records."""
from collections.abc import Sequence
import sqlalchemy as sa
from alembic import op

revision: str = "20260922_09"
down_revision: str | None = "20260922_08"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LEGACY_USER_ID = "00000000-0000-0000-0000-000000000001"

def upgrade() -> None:
    op.create_table("users", sa.Column("id", sa.String(36), primary_key=True), sa.Column("email", sa.String(320), nullable=False, unique=True), sa.Column("password_hash", sa.String(512), nullable=False), sa.Column("is_active", sa.Boolean(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_users_email", "users", ["email"])
    op.execute(sa.text("INSERT INTO users (id, email, password_hash, is_active, created_at, updated_at) VALUES (:id, :email, :password_hash, false, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)").bindparams(id=LEGACY_USER_ID, email="legacy-owner@local.invalid", password_hash="legacy-record-no-login"))
    op.add_column("datasets", sa.Column("user_id", sa.String(36), nullable=True))
    op.add_column("experiments", sa.Column("user_id", sa.String(36), nullable=True))
    op.execute(sa.text("UPDATE datasets SET user_id = :id WHERE user_id IS NULL").bindparams(id=LEGACY_USER_ID))
    op.execute(sa.text("UPDATE experiments SET user_id = :id WHERE user_id IS NULL").bindparams(id=LEGACY_USER_ID))
    op.alter_column("datasets", "user_id", nullable=False)
    op.alter_column("experiments", "user_id", nullable=False)
    op.create_foreign_key("fk_datasets_user", "datasets", "users", ["user_id"], ["id"], ondelete="RESTRICT")
    op.create_foreign_key("fk_experiments_user", "experiments", "users", ["user_id"], ["id"], ondelete="RESTRICT")
    op.create_index("ix_datasets_user_id", "datasets", ["user_id"])
    op.create_index("ix_experiments_user_id", "experiments", ["user_id"])

def downgrade() -> None:
    op.drop_index("ix_experiments_user_id", table_name="experiments"); op.drop_index("ix_datasets_user_id", table_name="datasets")
    op.drop_constraint("fk_experiments_user", "experiments", type_="foreignkey"); op.drop_constraint("fk_datasets_user", "datasets", type_="foreignkey")
    op.drop_column("experiments", "user_id"); op.drop_column("datasets", "user_id")
    op.drop_index("ix_users_email", table_name="users"); op.drop_table("users")
