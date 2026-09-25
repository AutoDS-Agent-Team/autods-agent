import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "autods-backend"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:5173"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    database_url: str = "postgresql+psycopg://autods:autods@localhost:5432/autods"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    celery_task_always_eager: bool = False
    celery_task_max_retries: int = Field(default=1, ge=0, le=3)
    expected_migration_revision: str = "20260925_14"
    jwt_secret_key: str = Field(min_length=32, default="change-this-development-secret-before-production")
    jwt_algorithm: str = "HS256"
    jwt_access_token_minutes: int = Field(default=60, ge=5, le=1440)
    google_client_id: str | None = None
    rate_limit_auth_per_minute: int = Field(default=10, ge=1, le=120)
    rate_limit_write_per_minute: int = Field(default=30, ge=1, le=300)
    data_retention_days: int = Field(default=0, ge=0, le=3650)
    assistant_max_experiments: int = Field(default=3, ge=1, le=20)
    assistant_max_context_chars: int = Field(default=12000, ge=1000, le=50000)
    assistant_max_turns: int = Field(default=8, ge=1, le=20)
    max_upload_size_mb: int = Field(default=25, gt=0)
    dataset_storage_path: Path = Path(__file__).resolve().parents[3] / "storage" / "datasets"
    model_storage_path: Path = Path(__file__).resolve().parents[3] / "storage" / "models"
    prediction_storage_path: Path = Path(__file__).resolve().parents[3] / "storage" / "predictions"
    report_storage_path: Path = Path(__file__).resolve().parents[3] / "storage" / "reports"
    profile_top_values_limit: int = Field(default=10, gt=0, le=100)
    text_tfidf_max_features: int = Field(default=2000, ge=100, le=10000)
    training_random_state: int = 42
    training_validation_size: float = Field(default=0.2, gt=0, lt=0.5)
    training_test_size: float = Field(default=0.2, gt=0, lt=0.5)
    optuna_n_trials: int = Field(default=10, ge=1, le=20)
    optuna_timeout_seconds: int = Field(default=120, ge=1, le=600)
    shap_max_samples: int = Field(default=200, ge=1, le=1000)

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_max_retries: int = Field(default=1, ge=0, le=5)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if stripped.startswith("["):
            return json.loads(stripped)
        return [origin.strip() for origin in stripped.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
