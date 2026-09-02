#!/usr/bin/env python3
"""One-command setup for the BI backend.

Usage:
    uv run python scripts/setup.py             # full setup (prompts for seed)
    uv run python scripts/setup.py --seed       # full setup + seed demo data
    uv run python scripts/setup.py --quick      # skip seeds, skip prompts
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import bcrypt

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], **kwargs) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="BI Backend setup")
    parser.add_argument("--seed", action="store_true", help="Seed demo data")
    parser.add_argument("--quick", action="store_true", help="Skip all prompts")
    args = parser.parse_args()

    # 1. .env
    env_path = ROOT / ".env"
    if not env_path.exists():
        shutil.copyfile(ROOT / ".env.example", env_path)
        print("Created .env from .env.example")
    else:
        print(".env already exists")

    # 2. Install deps
    run(["uv", "sync"])

    # 3. Check Postgres
    env = {**os.environ, "PGPASSWORD": "postgres"}
    try:
        subprocess.check_call(
            ["psql", "-U", "postgres", "-h", "localhost", "-c", "SELECT 1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
    except Exception:
        print("\nWARNING: Postgres not reachable at localhost:5432 user=postgres.")
        print("  Start it or set DATABASE_URL in .env and re-run.\n")
        return

    # 4. Create databases
    for db in ["bi_dev", "bi_test"]:
        subprocess.call(
            ["psql", "-U", "postgres", "-h", "localhost", "-c", f"CREATE DATABASE {db}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

    # 5. Migrations
    run(["uv", "run", "alembic", "upgrade", "head"])

    # 6. Create admin profile via raw SQL + generate a dev JWT
    print("\n>>> Creating dev admin profile & JWT")

    admin_id = "00000000-0000-0000-0000-000000000001"
    now = datetime.now(UTC)

    # Insert profile if not exists
    pw_hash = bcrypt.hashpw(b"admin123", bcrypt.gensalt()).decode()
    insert_sql = (
        f"INSERT INTO profiles (id, email, password_hash, full_name, role, department, is_active, created_at, updated_at) "
        f"VALUES ('{admin_id}', 'admin@sairash.com', '{pw_hash}', 'Admin', 'admin', 'Engineering', TRUE, '{now}', '{now}') "
        f"ON CONFLICT (id) DO NOTHING;"
    )
    subprocess.check_call(
        ["psql", "-U", "postgres", "-h", "localhost", "-d", "bi_dev", "-c", insert_sql],
        stdout=subprocess.DEVNULL,
        env=env,
    )
    print("Admin profile ready")

    # Generate JWT
    import jwt as pyjwt

    secret = "dev-secret-do-not-use-in-production"
    token = pyjwt.encode(
        {
            "sub": admin_id,
            "email": "admin@sairash.com",
            "app_metadata": {"role": "admin"},
            "aud": "authenticated",
            "exp": now + timedelta(days=365),
            "iat": now,
        },
        secret,
        algorithm="HS256",
    )

    # Write the dev token to a file for the frontend to reference
    (ROOT / ".dev-token").write_text(token)
    print("\nDev JWT written to .dev-token")
    print("Set this in the frontend .env.local as:")
    print(f"  NEXT_PUBLIC_DEV_API_TOKEN={token}")

    # 7. Seed demo data
    if args.seed:
        print("\n>>> Generating & seeding demo data")
        run([sys.executable, "seeds/generate_demo_data.py"])
        run([sys.executable, "scripts/seed_db.py"])
        print("\nDemo data seeded")
    elif not args.quick:
        resp = input("\nSeed demo data? (y/N): ").strip().lower()
        if resp == "y":
            run([sys.executable, "seeds/generate_demo_data.py"])
            run([sys.executable, "scripts/seed_db.py"])

    # 8. Print summary
    print("\n" + "=" * 50)
    print("  Setup complete!")
    print("=" * 50)
    print("\n  Start backend:   uv run uvicorn app.main:app --reload")
    print("  Start frontend:  cd ../business-intelligence-frontend && bun dev")
    print("  API docs:        http://localhost:8000/docs")
    print()


if __name__ == "__main__":
    main()
