import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

# Must be set before any app import so cached Settings pick them up.
TEST_DB_URL = "postgresql+asyncpg://postgres:postgres@localhost:54329/bi_test"
os.environ["ENV"] = "test"
os.environ["DATABASE_URL"] = TEST_DB_URL
os.environ["SUPABASE_JWT_SECRET"] = "test-jwt-secret-0123456789abcdef-0123456789abcdef"
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_SERVICE_KEY"] = ""
os.environ["RATE_LIMIT_PER_MINUTE"] = "100000"  # limiter logic is unit-tested directly

import jwt
import psycopg
import pytest
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from alembic import command
from app.core.database import dispose_engine, get_session_factory
from app.main import app
from app.models import Profile

ADMIN_DSN = "postgresql://postgres:postgres@localhost:54329/postgres"


def pytest_configure(config):  # noqa: C901
    try:
        with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = 'bi_test'").fetchone()
            if not exists:
                conn.execute("CREATE DATABASE bi_test")
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
    except Exception as e:  # pragma: no cover - DB not available in some envs
        # Allow unit tests to run without DB; integration tests will be skipped
        import warnings

        warnings.warn(f"DB unavailable, skipping DB setup: {e}", UserWarning, stacklevel=2)
        config.option.markexpr = "not integration"


@pytest.fixture(autouse=True)
async def _clean_tables() -> AsyncIterator[None]:
    yield
    from app.services.ai.circuit import reset_circuits

    reset_circuits()
    try:
        async with get_session_factory()() as session:
            result = await session.execute(
                text(
                    "SELECT schemaname, tablename FROM pg_tables "
                    "WHERE schemaname IN ('public', 'staging') "
                    "AND tablename != 'alembic_version'"
                )
            )
            tables = ", ".join(f'{s}."{t}"' for s, t in result)
            if tables:
                await session.execute(text(f"TRUNCATE {tables} CASCADE"))
                await session.commit()
    except Exception:
        pass  # DB not available — skip cleanup for unit-only runs


@pytest.fixture(scope="session", autouse=True)
async def _dispose() -> AsyncIterator[None]:
    yield
    await dispose_engine()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def mint_token(
    user_id: uuid.UUID,
    role: str = "analyst",
    *,
    org_id: uuid.UUID | None = None,
    expires_in: int = 3600,
    secret: str = "test-jwt-secret-0123456789abcdef-0123456789abcdef",
    audience: str = "authenticated",
) -> str:
    now = datetime.now(UTC)
    app_meta: dict[str, str] = {"role": role}
    if org_id is not None:
        app_meta["org_id"] = str(org_id)
    payload = {
        "sub": str(user_id),
        "aud": audience,
        "email": f"{role}@example.com",
        "app_metadata": app_meta,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
    }
    if org_id is not None:
        payload["org_id"] = str(org_id)
    return jwt.encode(payload, secret, algorithm="HS256")


async def _ensure_test_org() -> uuid.UUID:
    """Get or create a legacy org for tests (so org_id is not NULL after migration)."""
    from sqlalchemy import text as sa_text

    org_id = None
    try:
        async with get_session_factory()() as session:
            # Try legacy org first
            row = await session.execute(sa_text("SELECT id FROM organizations WHERE is_legacy = true LIMIT 1"))
            org_id = row.scalar_one_or_none()
            if org_id is None:
                row = await session.execute(sa_text("SELECT id FROM organizations LIMIT 1"))
                org_id = row.scalar_one_or_none()
            if org_id is None:
                # Create one
                new_id = uuid.uuid4()
                await session.execute(
                    sa_text(
                        "INSERT INTO organizations (id, name, slug, is_legacy, created_at, updated_at) "
                        "VALUES (:id, 'Test Legacy Org', 'test-legacy', true, now(), now())"
                    ),
                    {"id": str(new_id)},
                )
                await session.commit()
                org_id = new_id
            if isinstance(org_id, str):
                org_id = uuid.UUID(org_id)
            return org_id
    except Exception:
        # If DB not available or org table missing (pre-migration), return None (profile will have NULL)
        return None


async def create_profile(role: str, *, is_active: bool = True, org_id: uuid.UUID | None = None) -> Profile:
    if org_id is None:
        org_id = await _ensure_test_org()
    profile = Profile(
        id=uuid.uuid4(),
        email=f"{role}-{uuid.uuid4().hex[:6]}@example.com",
        full_name=f"Test {role.title()}",
        role=role,
        is_active=is_active,
        org_id=org_id,
    )
    try:
        async with get_session_factory()() as session:
            session.add(profile)
            await session.commit()
    except Exception as e:  # DB not available — skip for unit-only envs
        pytest.skip(f"DB not available, skipping integration fixture: {e}")
    return profile


@pytest.fixture
async def user_token() -> tuple[Profile, str]:
    profile = await create_profile("analyst")
    return profile, mint_token(profile.id, "analyst")


@pytest.fixture
async def manager_token() -> tuple[Profile, str]:
    """Lowest role allowed to upload / run ETL (see uploads + etl routers)."""
    profile = await create_profile("manager")
    return profile, mint_token(profile.id, "manager")


@pytest.fixture
async def admin_token() -> tuple[Profile, str]:
    profile = await create_profile("admin")
    return profile, mint_token(profile.id, "admin")


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
