"""supabase profile trigger

Creates a trigger on Supabase's auth.users that inserts a matching public.profiles
row on signup (default role: analyst). Guarded by a check for the auth schema so the
migration is a clean no-op on plain Postgres (local dev / CI databases).

Revision ID: 7659907cb5db
Revises: 2e93fabef7dd
Create Date: 2026-07-02 11:48:12.430510
"""

from collections.abc import Sequence

from alembic import op

revision: str = "7659907cb5db"
down_revision: str | None = "2e93fabef7dd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CREATE = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'auth') THEN
        CREATE OR REPLACE FUNCTION public.handle_new_auth_user()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER SET search_path = public
        AS $fn$
        BEGIN
            INSERT INTO public.profiles (id, email, full_name, role)
            VALUES (
                NEW.id,
                NEW.email,
                NEW.raw_user_meta_data ->> 'full_name',
                COALESCE(NEW.raw_app_meta_data ->> 'role', 'analyst')
            )
            ON CONFLICT (id) DO NOTHING;
            RETURN NEW;
        END;
        $fn$;

        DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
        CREATE TRIGGER on_auth_user_created
            AFTER INSERT ON auth.users
            FOR EACH ROW EXECUTE FUNCTION public.handle_new_auth_user();
    END IF;
END
$$;
"""

DROP = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'auth') THEN
        DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
        DROP FUNCTION IF EXISTS public.handle_new_auth_user();
    END IF;
END
$$;
"""


def upgrade() -> None:
    op.execute(CREATE)


def downgrade() -> None:
    op.execute(DROP)
