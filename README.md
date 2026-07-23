# Business Intelligence Backend

FastAPI backend for the AI-Driven Business Intelligence Dashboard.

## Prerequisites

- **Python 3.12+** (`python3 --version`)
- **uv** (`uv --version`) — install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **PostgreSQL** running on `localhost:5432` with user `postgres`

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Create env file
cp .env.example .env

# 3. Full setup (DB, migrations, admin user, dev JWT)
uv run python scripts/setup.py --seed

# 4. Start server
uv run uvicorn app.main:app --reload
```

The API is now at **http://localhost:8000** — docs at **http://localhost:8000/docs**

## Setup options

```bash
uv run python scripts/setup.py          # prompts for demo data
uv run python scripts/setup.py --seed   # auto-seed demo data
uv run python scripts/setup.py --quick  # no prompts, no seed
```

## Start only

```bash
uv run uvicorn app.main:app --reload
```

## Tests

```bash
uv run pytest --cov=app
```
