"""Seed the database with users for all three roles.

Creates or updates:
  - 1 admin  (email from ADMIN_EMAIL env / Settings)
  - 1 manager (manager@sairash.com)
  - 2 analysts (analyst1@sairash.com, analyst2@sairash.com)

Idempotent — safe to run multiple times.
Usage:
    uv run python seeds/seed_users.py
"""

import asyncio
import sys
from pathlib import Path

import bcrypt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import get_session_factory
from app.models import Profile
from app.services.supabase_admin import SupabaseAdmin, SupabaseAdminError

SEED_USERS = [
    {
        "email": "sairash@gmail.com",  # filled from settings
        "password": "Admin@123456",
        "role": "admin",
        "full_name": "Admin User",
        "department": "Engineering",
    },
    {
        "email": "manager@sairash.com",
        "password": "Manager@123456",
        "role": "manager",
        "full_name": "Manager User",
        "department": "Operations",
    },
    {
        "email": "analyst1@sairash.com",
        "password": "Analyst@123456",
        "role": "analyst",
        "full_name": "Analyst One",
        "department": "Sales",
    },
    {
        "email": "analyst2@sairash.com",
        "password": "Analyst@123456",
        "role": "analyst",
        "full_name": "Analyst Two",
        "department": "Marketing",
    },
]


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


async def main() -> None:
    settings = get_settings()

    # In prod, refuse to seed weak demo passwords unless explicitly forced.
    if settings.is_prod:
        demo_weak = any(u["password"] in {"Manager@123456", "Analyst@123456"} for u in SEED_USERS[1:])
        if demo_weak:
            print("Refusing to seed demo users with weak passwords in prod. Set ENV=dev or update passwords.")
            print("Only the admin user will be seeded from ADMIN_EMAIL/ADMIN_PASSWORD.")
            # Keep only admin
            SEED_USERS[:] = SEED_USERS[:1]

    SEED_USERS[0]["email"] = settings.admin_email
    SEED_USERS[0]["password"] = settings.admin_password

    admin_api: SupabaseAdmin | None = None
    try:
        admin_api = SupabaseAdmin()
    except SupabaseAdminError:
        print("Supabase admin API not available — will seed DB profiles only (no auth users)")

    async with get_session_factory()() as session:
        for user in SEED_USERS:
            existing = (
                await session.execute(select(Profile).where(Profile.email == user["email"]))
            ).scalar_one_or_none()

            if existing:
                changed = False
                if existing.role != user["role"]:
                    existing.role = user["role"]
                    changed = True
                if not existing.password_hash:
                    existing.password_hash = _hash_password(user["password"])
                    changed = True
                if changed:
                    await session.commit()
                    print(f"Updated {user['email']} (role={user['role']}, password set)")
                else:
                    print(f"Skipped {user['email']} (already exists)")
                continue

            user_id = None
            if admin_api:
                try:
                    user_id = await admin_api.create_user(
                        email=user["email"],
                        password=user["password"],
                        role=user["role"],
                        full_name=user["full_name"],
                    )
                    print(f"Created auth user {user['email']} (id={user_id})")
                except SupabaseAdminError as e:
                    print(f"Failed to create auth user {user['email']}: {e}")
                    continue

            if user_id is None:
                import uuid

                user_id = uuid.uuid5(uuid.NAMESPACE_URL, f"https://seed/{user['email']}")

            profile = Profile(
                id=user_id,
                email=user["email"],
                password_hash=_hash_password(user["password"]),
                full_name=user["full_name"],
                role=user["role"],
                department=user["department"],
                is_active=True,
            )
            session.add(profile)
            await session.commit()
            print(f"Seeded profile {user['email']} (role={user['role']})")

    print("\nUser seeding complete.")


if __name__ == "__main__":
    asyncio.run(main())
