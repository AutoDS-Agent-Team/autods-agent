from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.dependencies import get_db
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="healthy", service="autods-backend")


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness_check(database: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> ReadinessResponse:
    try:
        database.execute(text("SELECT 1"))
        from redis import Redis

        redis = Redis.from_url(settings.redis_url)
        redis.ping()
        redis.close()
        migration = database.execute(text("SELECT version_num FROM alembic_version LIMIT 1")).scalar_one_or_none()
        if migration != settings.expected_migration_revision:
            raise RuntimeError("Database migration is not at the required revision")
        from app.worker import celery_app
        
        workers = celery_app.control.ping(timeout=1.0) or []
        if not workers:
            raise RuntimeError("No Celery worker responded")
    except Exception as error:
        raise HTTPException(status_code=503, detail="Backend dependencies are not ready.") from error
    return ReadinessResponse(status="healthy", service="autods-backend", dependencies={"postgres": "ready", "redis": "ready", "migrations": "ready", "worker": "ready"})

