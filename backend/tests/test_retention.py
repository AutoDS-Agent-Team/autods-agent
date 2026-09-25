from datetime import datetime, timedelta, timezone

from app.core.config import Settings
from app.services.retention_service import cleanup_expired_data


def test_retention_zero_is_non_destructive(test_app) -> None:
    settings = test_app.state.testing_settings
    settings.data_retention_days = 0
    assert cleanup_expired_data(test_app.state.testing_session_factory(), settings) == {"datasets": 0, "experiments": 0, "model_artifacts": 0, "prediction_artifacts": 0, "report_artifacts": 0}


def test_retention_dry_run_is_safe(test_app) -> None:
    settings = test_app.state.testing_settings
    settings.data_retention_days = 1
    database = test_app.state.testing_session_factory()
    try:
        result = cleanup_expired_data(database, settings, now=datetime.now(timezone.utc), dry_run=True)
        assert result["datasets"] == 0 and result["experiments"] == 0
    finally:
        database.close()
