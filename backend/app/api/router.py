from fastapi import APIRouter

from app.api.routes.datasets import router as datasets_router
from app.api.routes.experiments import router as experiments_router
from app.api.routes.health import router as health_router
from app.api.routes.artifacts import router as artifacts_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.auth import router as auth_router
from app.api.routes.assistant import router as assistant_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(assistant_router)
api_router.include_router(datasets_router)
api_router.include_router(experiments_router)
api_router.include_router(artifacts_router)
api_router.include_router(jobs_router)
