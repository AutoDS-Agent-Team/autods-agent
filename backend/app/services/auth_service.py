from datetime import datetime, timedelta, timezone

import jwt
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import ApplicationError
from app.models.user import User

_hasher = PasswordHasher()
_DUMMY_PASSWORD_HASH = _hasher.hash("autods-invalid-login-dummy-password")


def register_user(email: str, password: str, database: Session) -> User:
    normalized = email.strip().lower()
    if database.scalar(select(User).where(User.email == normalized)):
        raise ApplicationError("An account with this email already exists.")
    user = User(email=normalized, password_hash=_hasher.hash(password))
    database.add(user); database.commit(); database.refresh(user)
    return user


def authenticate(email: str, password: str, database: Session, settings: Settings) -> tuple[str, User]:
    user = database.scalar(select(User).where(User.email == email.strip().lower()))
    try:
        candidate_hash = user.password_hash if user and user.password_hash else _DUMMY_PASSWORD_HASH
        valid = _hasher.verify(candidate_hash, password) and user is not None and user.is_active and user.password_hash is not None
    except (VerifyMismatchError, InvalidHashError):
        valid = False
    if not valid:
        raise ApplicationError("Invalid email or password.")
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_token_minutes)
    token = jwt.encode({"sub": user.id, "exp": expires, "iat": datetime.now(timezone.utc)}, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token, user

def authenticate_google(credential: str, database: Session, settings: Settings) -> tuple[str, User]:
    if not settings.google_client_id: raise ApplicationError("Google sign-in is not configured.")
    try: payload = id_token.verify_oauth2_token(credential, google_requests.Request(), settings.google_client_id)
    except Exception as error: raise ApplicationError("Google credential could not be verified.") from error
    subject, email = payload.get("sub"), payload.get("email")
    if not subject or not email or not payload.get("email_verified"): raise ApplicationError("Google credential is missing a verified identity.")
    user = database.scalar(select(User).where(User.google_subject == subject)) or database.scalar(select(User).where(User.email == email.lower()))
    if user is None:
        user = User(email=email.lower(), password_hash=None, google_subject=subject); database.add(user); database.commit(); database.refresh(user)
    elif user.google_subject not in (None, subject): raise ApplicationError("Google identity cannot be linked to this account.")
    elif user.google_subject is None:
        user.google_subject = subject; database.commit(); database.refresh(user)
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_token_minutes)
    return jwt.encode({"sub": user.id, "exp": expires}, settings.jwt_secret_key, algorithm=settings.jwt_algorithm), user
