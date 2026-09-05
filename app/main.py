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
    # Self-heal missing is_personal column (personal workspaces migration)
    # — makes fresh deploys / not-yet-migrated DBs work without manual alembic run.
    if get_settings().env not in ("test", "ci"):
        try:
            from sqlalchemy import text

            from app.core.database import get_engine

            eng = get_engine()
            async with eng.begin() as conn:
                await conn.execute(
                    text(
                        "ALTER TABLE organizations ADD COLUMN IF NOT EXISTS is_personal BOOLEAN NOT NULL DEFAULT false"
                    )
                )
        except Exception:  # noqa: BLE001 — best-effort self-heal, migration is source of truth
            pass
        # Self-heal RBAC: ensure compare:view permission exists and is granted to
        # analyst / manager / admin (covers deployments that were migrated before
        # the Compare feature). Idempotent — uses ON CONFLICT DO NOTHING.
        try:
            import uuid as _uuid

            from sqlalchemy import text as sa_text

            from app.core.database import get_engine as _get_eng
            from app.core.rbac_defaults import DEFAULT_GRANTS, DEFAULT_PERMISSIONS

            eng2 = _get_eng()
            async with eng2.begin() as conn:
                # Insert missing permissions
                for perm in DEFAULT_PERMISSIONS:
                    await conn.execute(
                        sa_text(
                            "INSERT INTO permissions (id, key, label, description, group_label, sort_order, is_system) "
                            "VALUES (:id, :key, :label, :desc, :grp, :ord, true) "
                            "ON CONFLICT (key) DO NOTHING"
                        ),
                        {
                            "id": str(_uuid.uuid4()),
                            "key": perm["key"],
                            "label": perm["label"],
                            "desc": perm["description"],
                            "grp": perm["group_label"],
                            "ord": perm["sort_order"],
                        },
                    )
                # Grant missing grants to default roles (and any other missing grant)
                for role_name, perms in DEFAULT_GRANTS.items():
                    for pk in perms:
                        await conn.execute(
                            sa_text(
                                "INSERT INTO role_permissions (id, role_id, permission_id) "
                                "SELECT :id, r.id, p.id FROM roles r, permissions p "
                                "WHERE r.name = :role AND p.key = :perm "
                                "ON CONFLICT (role_id, permission_id) DO NOTHING"
                            ),
                            {"id": str(_uuid.uuid4()), "role": role_name, "perm": pk},
                        )
        except Exception:  # noqa: BLE001 — best-effort, RBAC sync endpoint is the manual fallback
            pass
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
    import logging

    from fastapi import Request
    from fastapi.responses import JSONResponse

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
