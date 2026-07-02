# Business Intelligence Backend

FastAPI backend for the **AI-Driven Cloud-Based Business Intelligence and Decision Support
Dashboard** (FYP — Sairash Budhathoki, NP069813). Project-wide docs and the phase-by-phase plan
live in [`../docs/`](../docs/README.md).

## Stack
Python 3.12 · FastAPI · SQLAlchemy 2 (async) + Alembic · Supabase PostgreSQL · Supabase Auth (JWT)
· pytest · ruff + mypy · Docker · managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync                        # install deps (creates .venv)
cp .env.example .env           # fill in Supabase/AWS values
docker compose up -d db        # local Postgres (dev/test) on :54329
uv run alembic upgrade head    # apply schema
uv run uvicorn app.main:app --reload   # API on :8000, docs at /docs
```

Against the real Supabase dev project instead of local Postgres: set `DATABASE_URL` in `.env` to
the Supabase session-pooler connection string and rerun `alembic upgrade head`.

## Tests & quality

```bash
docker compose up -d db   # tests need the local Postgres (they create/use bi_test)
uv run pytest --cov=app   # full suite + coverage
uv run ruff check . && uv run ruff format --check .
uv run mypy app
```

## Layout

```
app/
  core/      config, database, security (JWT), audit middleware, logging
  api/v1/    routers (health, auth, users, audit-logs; later-phase stubs return 501)
  models/    SQLAlchemy ORM (identity, integration, warehouse, ml, decision)
  schemas/   Pydantic request/response models
  services/  supabase_admin; etl/ml/insights arrive in Phases 2–5
  workers/   APScheduler jobs (Phase 2+)
alembic/     migrations — the versioned schema history
scripts/     create_admin.py, export_openapi.py
tests/       unit/ + integration/ (run against dockerised Postgres)
```

## Conventions
- Roles: `admin` > `manager` > `analyst`; endpoints declare the minimum role
  (`require_role("admin")`). The `profiles` table is authoritative; JWT `app_metadata.role` is
  mirrored for Supabase RLS (Phase 6).
- All schema changes go through Alembic. Migrations must downgrade cleanly.
- Mutating authenticated requests are recorded in `audit_logs` by middleware.
