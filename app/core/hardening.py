"""Phase 6 API hardening: security headers + fixed-window rate limiting.

The limiter is in-memory (per-process). That is adequate for the single-node
EC2 deployment in Phase 7; a multi-node deployment would move the counters to
Redis without changing the middleware contract.
"""

import hashlib
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    # The API serves JSON only; a restrictive CSP neutralises any reflected HTML.
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "X-Permitted-Cross-Domain-Policies": "none",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response


class FixedWindowLimiter:
    """Counts requests per key per 60s window. Pure logic, unit-testable."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._counts: dict[tuple[str, int], int] = {}

    def hit(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Returns (allowed, remaining) and evicts stale windows."""
        ts = now if now is not None else time.time()
        window = int(ts // 60)
        self._counts = {k: v for k, v in self._counts.items() if k[1] >= window}
        bucket = (key, window)
        count = self._counts.get(bucket, 0) + 1
        self._counts[bucket] = count
        return count <= self.limit, max(self.limit - count, 0)


# Write-heavy or expensive endpoints get a tighter budget than reads.
# Email-related endpoints are also strict to prevent abuse / enumeration.
STRICT_PATHS = (
    "/api/v1/uploads",
    "/api/v1/forecasts/retrain",
    "/api/v1/reports/generate",
    "/api/v1/auth/forgot-password",
    "/api/v1/auth/reset-password",
    "/api/v1/auth/signup",
    "/api/v1/auth/login",
    "/api/v1/admin/email/test",
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit_per_minute: int = 240, strict_per_minute: int = 20) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.default = FixedWindowLimiter(limit_per_minute)
        self.strict = FixedWindowLimiter(strict_per_minute)

    @staticmethod
    def _key(request: Request) -> str:
        auth = request.headers.get("authorization")
        if auth:
            return hashlib.sha256(auth.encode()).hexdigest()[:16]
        client = request.client.host if request.client else "unknown"
        return f"ip:{client}"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        limiter = self.strict if request.method == "POST" and request.url.path in STRICT_PATHS else self.default
        allowed, remaining = limiter.hit(self._key(request))
        if not allowed:
            return JSONResponse(
                {"detail": "Rate limit exceeded. Try again shortly."},
                status_code=429,
                headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
