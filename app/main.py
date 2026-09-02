from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.audit import AuditMiddleware
from app.core.config import get_settings
from app.core.database import dispose_engine
from app.core.hardening import RateLimitMiddleware, SecurityHeadersMiddleware
from app.core.logging import setup_logging
from app.core.request_context import RequestContextMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    if get_settings().env not in ("test", "ci"):
        from app.workers.scheduler import start_scheduler, stop_scheduler

        await start_scheduler()
        yield
        stop_scheduler()
    else:
        yield
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="InsightFlow API — BI & Decision Support",
        version="0.1.0",
        description=(
            "Backend for InsightFlow — AI-Driven Cloud BI & Decision Support Dashboard "
            "(FYP — Sairash Budhathoki, NP069813)."
        ),
        lifespan=lifespan,
        # Hide framework details from attackers
        docs_url="/docs" if not settings.is_prod else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not settings.is_prod else "/openapi.json",
    )

    # Global unhandled-exception sanitiser — never leak stack traces or
    # internal paths to clients in prod. Dev keeps the full traceback via
    # FastAPI's debug response (when env!=prod we still sanitise 500s).
    from fastapi import Request
    from fastapi.responses import JSONResponse
    import logging

    logger = logging.getLogger(__name__)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # type: ignore[no-untyped-def]
        # Let HTTPException and Starlette HTTPException pass through
        from fastapi import HTTPException as FastAPIHTTPException
        from starlette.exceptions import HTTPException as StarletteHTTPException

        if isinstance(exc, (FastAPIHTTPException, StarletteHTTPException)):
            raise exc
        logger.exception("unhandled exception on %s %s", request.method, request.url.path, exc_info=exc)
        # In prod, hide details; in dev, include type for debugging but still no stack.
        detail = "Internal server error" if settings.is_prod else f"{type(exc).__name__}: request failed"
        return JSONResponse({"detail": detail}, status_code=500)

    # RequestContext must be the outermost middleware: its request_id contextvar
    # must stay live while every inner middleware (audit, rate limit) runs.
    app.add_middleware(AuditMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, limit_per_minute=settings.rate_limit_per_minute)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept", "X-Request-ID"],
        expose_headers=["X-RateLimit-Remaining", "X-Request-ID"],
        max_age=600,
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
