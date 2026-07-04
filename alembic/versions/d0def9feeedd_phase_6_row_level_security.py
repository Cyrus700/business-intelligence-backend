"""phase 6 row level security

Enables RLS on every application table with policies driven by the Supabase
JWT role claim (``auth.jwt() -> 'app_metadata' ->> 'role'``). The FastAPI
service connects as the table owner (local dev) or with the Supabase
service-role key (production), both of which bypass RLS by design — RLS here
is defence-in-depth for any client-side / anon-key access path.

On local Postgres (no Supabase) the ``auth`` schema does not exist, so this
migration creates compatible ``auth.jwt()`` / ``auth.uid()`` stubs reading
``request.jwt.claims`` — the same mechanism Supabase uses — only when absent.

Revision ID: d0def9feeedd
Revises: 7659907cb5db
Create Date: 2026-07-04
"""

from collections.abc import Sequence

from alembic import op

revision: str = "d0def9feeedd"
down_revision: str | None = "7659907cb5db"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ROLE = "(auth.jwt() -> 'app_metadata' ->> 'role')"
ANY_ROLE = f"{ROLE} IN ('admin', 'manager', 'analyst')"
MANAGER_UP = f"{ROLE} IN ('admin', 'manager')"
ADMIN = f"{ROLE} = 'admin'"

# table -> list of (policy_name, command, using_expr, check_expr | None)
POLICIES: dict[str, list[tuple[str, str, str, str | None]]] = {
    # profiles: own row, or admin sees/manages everyone
    "profiles": [
        ("profiles_read_own_or_admin", "SELECT", f"id = auth.uid() OR {ADMIN}", None),
        ("profiles_admin_write", "ALL", ADMIN, ADMIN),
    ],
    # warehouse: any authenticated role reads; writes only via service role
    "products": [("read_any_role", "SELECT", ANY_ROLE, None)],
    "customers": [("read_any_role", "SELECT", ANY_ROLE, None)],
    "sales_transactions": [("read_any_role", "SELECT", ANY_ROLE, None)],
    "expenses": [("read_any_role", "SELECT", ANY_ROLE, None)],
    "inventory_levels": [("read_any_role", "SELECT", ANY_ROLE, None)],
    "kpi_snapshots": [("read_any_role", "SELECT", ANY_ROLE, None)],
    # ETL surface: admin only
    "data_sources": [("admin_only", "ALL", ADMIN, ADMIN)],
    "etl_jobs": [("admin_only", "ALL", ADMIN, ADMIN)],
    "staging.raw_uploads": [("admin_only", "ALL", ADMIN, ADMIN)],
    # ML / decision outputs: all roles read, manager+ updates status fields
    "ml_models": [("read_any_role", "SELECT", ANY_ROLE, None)],
    "forecasts": [("read_any_role", "SELECT", ANY_ROLE, None)],
    "anomalies": [
        ("read_any_role", "SELECT", ANY_ROLE, None),
        ("manager_update", "UPDATE", MANAGER_UP, MANAGER_UP),
    ],
    "insights": [
        ("read_any_role", "SELECT", ANY_ROLE, None),
        ("manager_update", "UPDATE", MANAGER_UP, MANAGER_UP),
    ],
    "alert_rules": [("manager_all", "ALL", MANAGER_UP, MANAGER_UP)],
    "reports": [("read_any_role", "SELECT", ANY_ROLE, None)],
    # notifications: strictly own rows
    "notifications": [
        ("own_rows_select", "SELECT", "user_id = auth.uid()", None),
        ("own_rows_update", "UPDATE", "user_id = auth.uid()", "user_id = auth.uid()"),
    ],
    # audit log: admin read; inserts happen via service role only
    "audit_logs": [("admin_read", "SELECT", ADMIN, None)],
}

AUTH_STUBS = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'auth' AND p.proname = 'jwt'
    ) THEN
        CREATE FUNCTION auth.jwt() RETURNS jsonb LANGUAGE sql STABLE AS
        $f$ SELECT COALESCE(current_setting('request.jwt.claims', true), '{}')::jsonb $f$;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'auth' AND p.proname = 'uid'
    ) THEN
        CREATE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS
        $f$ SELECT NULLIF(auth.jwt() ->> 'sub', '')::uuid $f$;
    END IF;
END $$;
"""


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS auth")
    op.execute(AUTH_STUBS)
    for table, policies in POLICIES.items():
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        for name, command, using, check in policies:
            with_check = f" WITH CHECK ({check})" if check else ""
            op.execute(
                f"CREATE POLICY {name} ON {table} FOR {command} "
                f"TO PUBLIC USING ({using}){with_check}"
            )


def downgrade() -> None:
    for table, policies in POLICIES.items():
        for name, _, _, _ in policies:
            op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
