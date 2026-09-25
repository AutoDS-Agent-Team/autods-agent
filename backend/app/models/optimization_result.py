from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class OptimizationResult(Base):
    __tablename__ = "optimization_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, unique=True)
    baseline_model_run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.id", ondelete="RESTRICT"), nullable=False)
    optimized_model_run_id: Mapped[str | None] = mapped_column(ForeignKey("model_runs.id", ondelete="SET NULL"), nullable=True)
    model_name: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    trial_count: Mapped[int] = mapped_column(nullable=False)
    best_parameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    best_validation_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    cost_profile: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    failure_information: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
