"""
Shared rate limiter for VANI using SlowAPI.
This ensures a single Limiter instance is used across the app and routers.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.settings import settings

def _default_rate_limit() -> str:
    """Build a human-readable default rate limit string from settings.
    Examples: '100/minute', '20/second', '50/hour', or '100 per 45 seconds'.
    """
    try:
        reqs = int(getattr(settings, 'rate_limit_requests', 100))
        window = int(getattr(settings, 'rate_limit_window', 60))
    except Exception:
        reqs, window = 100, 60

    if window == 1:
        return f"{reqs}/second"
    if window == 60:
        return f"{reqs}/minute"
    if window == 3600:
        return f"{reqs}/hour"
    # Generic fallback for arbitrary windows
    return f"{reqs} per {window} seconds"

# Single shared limiter instance with default limits from settings
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[_default_rate_limit()]
)

def endpoint_rate_limit() -> str:
    """Callable used by @limiter.limit for dynamic per-endpoint limits from settings."""
    return _default_rate_limit()
