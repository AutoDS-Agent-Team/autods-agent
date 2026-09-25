from collections.abc import Sequence
import sqlalchemy as sa
from alembic import op
revision='20260922_10'; down_revision='20260922_09'; branch_labels: str | Sequence[str] | None=None; depends_on: str | Sequence[str] | None=None
def upgrade():
    op.alter_column('users','password_hash',nullable=True); op.add_column('users',sa.Column('google_subject',sa.String(255),nullable=True)); op.create_unique_constraint('uq_users_google_subject','users',['google_subject'])
def downgrade():
    op.drop_constraint('uq_users_google_subject','users',type_='unique'); op.drop_column('users','google_subject'); op.alter_column('users','password_hash',nullable=False)
