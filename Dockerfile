# syntax=docker/dockerfile:1.7
# ── Backend — FastAPI + Uvicorn — free-tier optimized (t2.micro 1GB) ──
# Build on GitHub (7GB) not on EC2 (1GB). Final image ~180MB.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Cache deps first (max layer reuse)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy source
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./

# Install project (no dev deps) — uses cache
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ── Runtime (slim, non-root, read-only) ──
FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app

# Runtime deps + non-root user (no build tools)
RUN groupadd -r api --gid 1001 && useradd -r -g api --uid 1001 api \
 && apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /app/var/uploads && chown -R api:api /app

# Copy venv + app from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app /app/app
COPY --from=builder /app/alembic /app/alembic
COPY --from=builder /app/alembic.ini /app/alembic.ini
RUN chown -R api:api /app/.venv /app/app /app/alembic /app/alembic.ini

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENV=prod

USER api
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
