"""Promote (or create) an admin user against the configured Supabase project.

Usage:
    uv run python scripts/create_admin.py admin@example.com  # promote existing profile
    uv run python scripts/create_admin.py admin@example.com --create --password 'S3cure!'

Requires SUPABASE_URL + SUPABASE_SERVICE_KEY in .env.
"""

import argparse
import asyncio

from sqlalchemy import select

from app.core.database import get_session_factory
from app.models import Profile
from app.services.supabase_admin import SupabaseAdmin


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("email")
    parser.add_argument("--create", action="store_true", help="create the auth user first")
    parser.add_argument("--password", help="password when using --create")
    args = parser.parse_args()

    admin_api = SupabaseAdmin()

    if args.create:
        if not args.password:
            raise SystemExit("--create requires --password")
        user_id = await admin_api.create_user(args.email, args.password, "admin", "Administrator")
        print(f"created auth user {user_id}")

    async with get_session_factory()() as session:
        profile = (
            await session.execute(select(Profile).where(Profile.email == args.email))
        ).scalar_one_or_none()
        if profile is None:
            raise SystemExit(
                f"no profile for {args.email} — sign the user up first or pass --create"
            )
        profile.role = "admin"
        await admin_api.set_role(profile.id, "admin")
        await session.commit()
        print(f"{args.email} is now admin (profile + JWT app_metadata)")


if __name__ == "__main__":
    asyncio.run(main())
