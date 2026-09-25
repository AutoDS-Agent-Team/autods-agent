from collections import defaultdict, deque
from time import monotonic, time
from fastapi import Request
from app.core.config import get_settings
from app.core.exceptions import ApplicationError

_windows: dict[str, deque[float]] = defaultdict(deque)

def limit(scope: str):
    async def check(request: Request) -> None:
        settings = get_settings()
        if settings.app_env == "test":
            return
        host = request.client.host if request.client else "unknown"
        if settings.app_env == "development" and host in {"127.0.0.1", "::1", "testclient"}:
            return
        maximum = settings.rate_limit_auth_per_minute if scope == "auth" else settings.rate_limit_write_per_minute
        try:
            from redis.asyncio import Redis
        except ImportError:
            key = f"{scope}:{request.client.host if request.client else 'unknown'}"
            now = monotonic(); window = _windows[key]
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= maximum:
                error = ApplicationError("Too many requests. Please try again shortly."); error.status_code = 429
                raise error
            window.append(now)
            return

        bucket = int(time() // 60)
        redis = Redis.from_url(settings.redis_url)
        try:
            key = f"autods:rate:{scope}:{host}:{bucket}"
            count = await redis.incr(key)
            if count == 1:
                await redis.expire(key, 61)
            if count > maximum:
                error = ApplicationError("Too many requests. Please try again shortly."); error.status_code = 429
                raise error
        finally:
            await redis.aclose()
    return check
