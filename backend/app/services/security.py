from typing import Annotated
import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError, ExperimentNotFoundError
from app.db.dependencies import get_db
from app.models.user import User

_bearer = HTTPBearer(auto_error=False)

def get_current_user(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)], database: Annotated[Session, Depends(get_db)], settings: Annotated[Settings, Depends(get_settings)]) -> User:
    if credentials is None:
        raise AuthenticationError("Authentication is required.")
    try:
        subject = jwt.decode(credentials.credentials, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]).get("sub")
    except jwt.PyJWTError as error:
        raise AuthenticationError("Invalid or expired access token.") from error
    user = database.get(User, subject)
    if user is None or not user.is_active:
        raise AuthenticationError("Invalid or expired access token.")
    return user

def require_experiment_owner(experiment_id: str, user: User, database: Session):
    from app.models.experiment import Experiment
    experiment = database.get(Experiment, experiment_id)
    if experiment is None or experiment.user_id != user.id:
        raise ExperimentNotFoundError("Experiment not found.")
    return experiment
