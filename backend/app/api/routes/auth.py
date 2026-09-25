from typing import Annotated
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.core.config import Settings, get_settings
from app.db.dependencies import get_db
from app.schemas.auth import GoogleLoginRequest, LoginRequest, RegisterRequest, TokenResponse, UserResponse
from app.services.auth_service import authenticate, authenticate_google, register_user
from app.services.security import get_current_user
from app.models.user import User
from app.core.rate_limit import limit

router = APIRouter(prefix="/auth", tags=["auth"])

def _token(token: str, user: User) -> TokenResponse:
    return TokenResponse(access_token=token, user=UserResponse(id=user.id, email=user.email))

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(limit("auth"))])
def register(request: RegisterRequest, database: Annotated[Session, Depends(get_db)], settings: Annotated[Settings, Depends(get_settings)]) -> TokenResponse:
    user = register_user(request.email, request.password, database)
    token, _ = authenticate(request.email, request.password, database, settings)
    return _token(token, user)

@router.post("/login", response_model=TokenResponse, dependencies=[Depends(limit("auth"))])
def login(request: LoginRequest, database: Annotated[Session, Depends(get_db)], settings: Annotated[Settings, Depends(get_settings)]) -> TokenResponse:
    token, user = authenticate(request.email, request.password, database, settings)
    return _token(token, user)

@router.post("/google", response_model=TokenResponse, dependencies=[Depends(limit("auth"))])
def google_login(request: GoogleLoginRequest, database: Annotated[Session, Depends(get_db)], settings: Annotated[Settings, Depends(get_settings)]) -> TokenResponse:
    token, user = authenticate_google(request.credential, database, settings)
    return _token(token, user)

@router.get("/me", response_model=UserResponse)
def me(user: Annotated[User, Depends(get_current_user)]) -> UserResponse:
    return UserResponse(id=user.id, email=user.email)
