from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_objective: Mapped[str] = mapped_column(Text, nullable=False)
    detected_task_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    suggested_target_column: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    confirmed_task_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confirmed_target_column: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    selected_model_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_runs.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
