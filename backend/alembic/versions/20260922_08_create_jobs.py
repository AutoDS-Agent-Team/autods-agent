"""Create persistent background jobs."""
from collections.abc import Sequence
import sqlalchemy as sa
from alembic import op
revision: str = "20260922_08"
down_revision: str | None = "20260922_07"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None
def upgrade() -> None:
    op.create_table("jobs", sa.Column("id", sa.String(36), primary_key=True), sa.Column("experiment_id", sa.String(36), nullable=False), sa.Column("job_type", sa.String(32), nullable=False), sa.Column("status", sa.String(32), nullable=False), sa.Column("stage", sa.String(32), nullable=False), sa.Column("error_information", sa.Text()), sa.Column("result_reference", sa.String(36)), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.Column("started_at", sa.DateTime(timezone=True)), sa.Column("completed_at", sa.DateTime(timezone=True)), sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="CASCADE"))
    op.create_index("ix_jobs_experiment_id", "jobs", ["experiment_id"])
def downgrade() -> None:
    op.drop_index("ix_jobs_experiment_id", table_name="jobs"); op.drop_table("jobs")
