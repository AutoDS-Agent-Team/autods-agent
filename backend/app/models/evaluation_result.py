from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    selected_model_run_id: Mapped[str] = mapped_column(
        ForeignKey("model_runs.id", ondelete="RESTRICT"), nullable=False
    )
    primary_metric: Mapped[str] = mapped_column(String(32), nullable=False)
    validation_comparison: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    final_test_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    final_test_confusion_matrix: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
