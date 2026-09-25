import asyncio
import pytest
from starlette.requests import Request
from app.core.config import get_settings
from app.core.rate_limit import _windows, limit
from app.core.exceptions import ApplicationError

def request(host: str) -> Request:
    return Request({"type":"http", "method":"POST", "path":"/x", "headers":[], "client":(host, 1234), "scheme":"http", "server":("test",80)})

def test_auth_and_write_limits_use_independent_buckets() -> None:
    settings = get_settings(); previous_env, previous_auth, previous_write = settings.app_env, settings.rate_limit_auth_per_minute, settings.rate_limit_write_per_minute
    settings.app_env="development"; settings.rate_limit_auth_per_minute=1; settings.rate_limit_write_per_minute=1; _windows.clear()
    try:
        asyncio.run(limit("auth")(request("rate-test")))
        asyncio.run(limit("write")(request("rate-test")))
        with pytest.raises(ApplicationError) as auth_error: asyncio.run(limit("auth")(request("rate-test")))
        with pytest.raises(ApplicationError) as write_error: asyncio.run(limit("write")(request("rate-test")))
        assert auth_error.value.status_code == write_error.value.status_code == 429
        asyncio.run(limit("write")(request("another-user")))
    finally:
        settings.app_env, settings.rate_limit_auth_per_minute, settings.rate_limit_write_per_minute = previous_env, previous_auth, previous_write; _windows.clear()
