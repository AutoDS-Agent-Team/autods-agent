from typing import Annotated
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.dependencies import get_db
from app.schemas.job import JobResponse
from app.services.job_service import get_job
from app.models.user import User
from app.services.security import get_current_user, require_experiment_owner

router = APIRouter(prefix="/jobs", tags=["jobs"])
@router.get("/{job_id}", response_model=JobResponse)
def get_job_definition(job_id: str, database: Annotated[Session, Depends(get_db)], user: Annotated[User, Depends(get_current_user)]) -> JobResponse:
    job = get_job(job_id, database)
    require_experiment_owner(job.experiment_id, user, database)
    return job
