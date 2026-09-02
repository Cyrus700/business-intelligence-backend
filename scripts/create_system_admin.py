"""Seed (or reset) a platform System Admin that signs in with a password.

This is the operator account: ``is_super_admin`` — the one that approves or
rejects new businesses and can see every org. Unlike ``create_admin.py`` it does
not go through Supabase; it writes the bcrypt hash straight into ``profiles``,
which is what ``POST /auth/login`` checks.

Usage:
    uv run python scripts/create_system_admin.py you@example.com --password 'S3cure!'
    uv run python scripts/create_system_admin.py            # ADMIN_EMAIL/ADMIN_PASSWORD from .env

Re-running on an existing account resets the password and re-asserts the flags;
``token_version`` is bumped so any session issued before the reset stops working
on its next request.
"""

import argparse
import asyncio
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import bcrypt
from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.database import get_session_factory
from app.models import Organization, Profile


def _check_password(password: str) -> None:
    if len(password) < 8:
        raise SystemExit("password must be at least 8 characters")
    if not (any(c.isalpha() for c in password) and any(c.isdigit() for c in password)):
        raise SystemExit("password must contain at least one letter and one number")


async def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("email", nargs="?", default=settings.admin_email)
    parser.add_argument("--password", default=getattr(settings, "admin_password", None))
    parser.add_argument("--full-name", default="System Admin")
    args = parser.parse_args()

    email = args.email.strip().lower()
    if not args.password:
        raise SystemExit("--password is required (or set ADMIN_PASSWORD in .env)")
    _check_password(args.password)

    pw_hash = bcrypt.hashpw(args.password.encode(), bcrypt.gensalt(rounds=12)).decode()
    now = datetime.now(UTC).replace(tzinfo=None)

    async with get_session_factory()() as session:
        profile = (
            await session.execute(select(Profile).where(func.lower(Profile.email) == email))
        ).scalar_one_or_none()

        # A super admin sees every org, but the app still wants an org to render
        # tenant-scoped screens against — the legacy org is the shared one.
        legacy = (
            await session.execute(select(Organization).where(Organization.is_legacy.is_(True)))
        ).scalar_one_or_none()

        if profile is None:
            profile = Profile(
                id=uuid5(NAMESPACE_URL, f"email://{email}"),
                email=email,
                full_name=args.full_name,
                org_id=legacy.id if legacy else None,
            )
            session.add(profile)
            action = "created"
        else:
            action = "updated"
            # Old sessions must not survive a credential reset.
            profile.token_version = (profile.token_version or 0) + 1
            if profile.org_id is None and legacy is not None:
                profile.org_id = legacy.id

        profile.password_hash = pw_hash
        profile.role = "admin"
        profile.is_super_admin = True
        profile.is_active = True
        profile.email_verified = True
        profile.email_verified_at = profile.email_verified_at or now

        await session.commit()
        await session.refresh(profile)

    print(
        f"{action} system admin {profile.email} "
        f"(role={profile.role}, super_admin={profile.is_super_admin}, "
        f"org_id={profile.org_id}, token_version={profile.token_version})"
    )


if __name__ == "__main__":
    asyncio.run(main())
