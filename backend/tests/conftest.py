from collections.abc import Generator
from pathlib import Path
import shutil
from uuid import uuid4

import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.dependencies import get_db
from app.main import app
from app import models as _models  # noqa: F401
from app.models.user import User
from app.services.security import get_current_user


@pytest.fixture
def test_storage_path() -> Generator[Path, None, None]:
    path = Path(__file__).parent / ".test-data" / uuid4().hex
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def test_app(test_storage_path: Path) -> Generator[FastAPI, None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
    Base.metadata.create_all(engine)

    settings = Settings(
        app_env="test",
        google_client_id="test-google-client-id",
        database_url="sqlite+pysqlite:///:memory:",
        dataset_storage_path=test_storage_path / "datasets",
        model_storage_path=test_storage_path / "models",
        prediction_storage_path=test_storage_path / "predictions",
        report_storage_path=test_storage_path / "reports",
        max_upload_size_mb=1,
        profile_top_values_limit=3,
    )
    session = testing_session()
    test_user = User(id="11111111-1111-1111-1111-111111111111", email="test@example.com", password_hash="test-hash")
    session.add(test_user); session.commit(); session.close()

    def override_get_db() -> Generator[Session, None, None]:
        database = testing_session()
        try:
            yield database
        finally:
            database.close()

    def override_get_settings() -> Settings:
        return settings

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = override_get_settings
    app.dependency_overrides[get_current_user] = lambda: test_user
    app.state.testing_session_factory = testing_session
    app.state.testing_settings = settings
    yield app
    app.dependency_overrides.clear()
    del app.state.testing_session_factory
    del app.state.testing_settings
    Base.metadata.drop_all(engine)
    engine.dispose()
