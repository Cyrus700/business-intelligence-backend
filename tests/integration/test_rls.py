"""Row Level Security tests, run locally against the same policies Supabase
enforces in production.

The RLS migration installs `auth.jwt()` / `auth.uid()` stubs that read the
`request.jwt.claims` setting — the exact mechanism Supabase/PostgREST uses.
Here we assume a non-owner role (`rls_probe`) and set claims per test, so the
policies are exercised without needing a Supabase project. The backend's own
connection (table owner / service role) bypasses RLS by design.
"""

import json
import uuid
from datetime import date

from sqlalchemy import text

from app.core.database import get_session_factory
from app.models import AuditLog, DataSource, KpiSnapshot, Notification

SETUP = [
    """DO $$ BEGIN
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rls_probe') THEN
            CREATE ROLE rls_probe NOLOGIN;
        END IF;
    END $$""",
    "GRANT USAGE ON SCHEMA public TO rls_probe",
    "GRANT USAGE ON SCHEMA auth TO rls_probe",  # Supabase grants this to authenticated
    "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rls_probe",
]


async def seed() -> tuple[uuid.UUID, uuid.UUID]:
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    async with get_session_factory()() as db:
        for stmt in SETUP:
            await db.execute(text(stmt))
        db.add(KpiSnapshot(snapshot_date=date(2026, 6, 30), metric="revenue", value=100))
        db.add(AuditLog(action="login", entity="session"))
        await db.commit()
    return user_a, user_b


async def count_as(role: str | None, sub: uuid.UUID, table: str) -> int:
    claims = json.dumps({"sub": str(sub), "app_metadata": {"role": role}} if role else {})
    async with get_session_factory()() as db:
        await db.execute(text("SET LOCAL ROLE rls_probe"))
        await db.execute(
            text("SELECT set_config('request.jwt.claims', :claims, true)"), {"claims": claims}
        )
        n = (await db.execute(text(f"SELECT COUNT(*) FROM {table}"))).scalar_one()
        await db.rollback()
    return int(n)


async def test_warehouse_readable_by_any_role_but_not_anon():
    await seed()
    me = uuid.uuid4()
    assert await count_as("analyst", me, "kpi_snapshots") == 1
    assert await count_as("manager", me, "kpi_snapshots") == 1
    assert await count_as(None, me, "kpi_snapshots") == 0  # anon sees nothing


async def test_audit_logs_admin_only():
    await seed()
    me = uuid.uuid4()
    assert await count_as("admin", me, "audit_logs") >= 1
    assert await count_as("analyst", me, "audit_logs") == 0
    assert await count_as("manager", me, "audit_logs") == 0


async def test_notifications_own_rows_only():
    user_a, user_b = await seed()
    async with get_session_factory()() as db:
        await db.execute(
            text(
                "INSERT INTO profiles (id, email, full_name, role, is_active) VALUES "
                "(:a, 'a@x.com', 'A', 'analyst', true), (:b, 'b@x.com', 'B', 'analyst', true)"
            ),
            {"a": user_a, "b": user_b},
        )
        db.add(Notification(user_id=user_a, title="for A"))
        db.add(Notification(user_id=user_b, title="for B"))
        await db.commit()

    assert await count_as("analyst", user_a, "notifications") == 1
    assert await count_as("analyst", user_b, "notifications") == 1


async def test_etl_surface_admin_only():
    await seed()
    me = uuid.uuid4()
    async with get_session_factory()() as db:
        db.add(DataSource(name="src", kind="csv_upload", target_domain="sales"))
        await db.commit()
    assert await count_as("admin", me, "data_sources") == 1
    assert await count_as("analyst", me, "data_sources") == 0
