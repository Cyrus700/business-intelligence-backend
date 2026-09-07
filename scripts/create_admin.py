"""Promote (or create) an admin user locally.

Usage:
    uv run python scripts/create_admin.py                    # use ADMIN_EMAIL from .env
    uv run python scripts/create_admin.py admin@example.com  # promote existing profile
    uv run python scripts/create_admin.py admin@example.com --create --password 'S3cure!'
"""

import argparse
import asyncio
from uuid import NAMESPACE_URL, uuid5

import bcrypt
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import get_session_factory
from app.models import Profile


async def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "email", nargs="?", default=settings.admin_email.split(",")[0].strip() if settings.admin_email else ""
    )
    parser.add_argument("--create", action="store_true", help="create the user first")
    parser.add_argument("--password", default=settings.admin_password, help="password when using --create")
    args = parser.parse_args()

    if args.create:
        if not args.password:
            raise SystemExit("--create requires --password")
        pw_hash = bcrypt.hashpw(args.password.encode(), bcrypt.gensalt(rounds=12)).decode()
        async with get_session_factory()() as session:
            existing = (
                await session.execute(select(Profile).where(Profile.email == args.email.strip().lower()))
            ).scalar_one_or_none()
            if existing:
                print(f"profile for {args.email} already exists — promoting")
            else:
                profile = Profile(
                    id=uuid5(NAMESPACE_URL, f"email://{args.email.strip().lower()}"),
                    email=args.email.strip().lower(),
                    password_hash=pw_hash,
                    role="admin",
                    is_active=True,
                    email_verified=True,
                )
                session.add(profile)
                await session.commit()
                print(f"created user {args.email} (id={profile.id})")

    async with get_session_factory()() as session:
        profile = (
            await session.execute(select(Profile).where(Profile.email == args.email.strip().lower()))
        ).scalar_one_or_none()
        if profile is None:
            raise SystemExit(f"no profile for {args.email} — sign the user up first or pass --create")
        profile.role = "admin"
        profile.is_super_admin = True
        profile.is_active = True
        profile.token_version = (profile.token_version or 0) + 1
        await session.commit()
        print(f"{args.email} is now admin (role=admin, super_admin=true, token_version={profile.token_version})")


if __name__ == "__main__":
    asyncio.run(main())
