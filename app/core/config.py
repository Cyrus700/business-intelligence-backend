"""Central application settings — hardened, validated, env-aware.

Design principles (totos.md §29-32, §34):
- Fail fast in production if any secret is weak or missing.
- Never log, expose or echo a secret.
- ``ENV=dev`` stays ergonomic (sensible defaults), ``ENV=prod`` is strict.
- Every value is validated at import time so a bad deploy aborts before it serves traffic.
"""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_ENV_VALUES = ("dev", "test", "ci", "prod", "production")

_WEAK_SECRETS = {
    "dev-secret-do-not-use-in-production",
    "change-me",
    "changeme",
    "secret",
    "password",
    "",
}

# Matches a plausible strong secret: >=32 chars, mix of classes
_STRONG_RE = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{32,}$")


def _is_strong_secret(v: str) -> bool:
    if v in _WEAK_SECRETS:
        return False
    if len(v) < 32:
        return False
    # In prod we strongly encourage the mix check, but we don't hard-fail on it
    # to avoid blocking legitimate 64-char hex secrets that are still strong.
    return True


def _mask(v: str) -> str:
    if not v:
        return "(empty)"
    if len(v) <= 8:
        return "***"
    return f"{v[:3]}***{v[-3:]}"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # ── Environment ───────────────────────────────────────────────────────
    env: str = Field(default="dev", description="Runtime env: dev|test|ci|prod")
    database_url: str = Field(default="", description="Postgres DSN (postgresql+asyncpg://)")

    supabase_url: str = Field(default="", description="Supabase project URL")
    supabase_jwt_secret: str = Field(default="", description="HS256 secret for JWT mint/verify")
    supabase_service_key: str = Field(default="", description="Supabase service_role key (server-only)")
    supabase_anon_key: str = Field(default="", description="Supabase anon key")

    jwt_audience: str = Field(default="authenticated")
    jwt_expiry_hours: int = Field(default=24, ge=1, le=720, description="Auth token TTL in hours (prod default 24h)")
    jwt_reset_expiry_minutes: int = Field(default=30, ge=5, le=120, description="Password-reset token TTL")

    aws_region: str = Field(default="ap-south-1")
    s3_bucket: str = Field(default="bi-fyp-dev-uploads")

    frontend_origins: str = Field(default="http://localhost:3000", description="Comma-separated CORS allowlist")

    google_client_id: str = Field(default="")
    google_client_secret: str = Field(default="")
    google_redirect_uri: str = Field(default="http://localhost:8000/api/v1/auth/google/callback")

    admin_email: str = Field(default="admin@example.com")
    admin_password: str = Field(default="")

    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-2.0-flash")

    # ── Email / SMTP ──────────────────────────────────────────────────────
    smtp_host: str = Field(default="", description="SMTP host (empty disables email)")
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    smtp_from: str = Field(default="InsightFlow <alerts@insightflow.local>")
    smtp_timeout: int = Field(default=15, ge=5, le=60, description="SMTP socket timeout (seconds)")
    smtp_max_retries: int = Field(default=2, ge=0, le=5)
    email_rate_limit_per_minute: int = Field(default=10, ge=1, le=100, description="Max emails per recipient per minute")

    # ── Rate limiting ─────────────────────────────────────────────────────
    rate_limit_per_minute: int = Field(default=240, ge=10, le=100_000)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("env")
    @classmethod
    def _validate_env(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in _ENV_VALUES:
            raise ValueError(f"ENV must be one of {_ENV_VALUES}, got {v!r}")
        # normalise "production" → "prod"
        return "prod" if v == "production" else v

    @field_validator("frontend_origins")
    @classmethod
    def _validate_origins(cls, v: str) -> str:
        origins = [o.strip() for o in v.split(",") if o.strip()]
        if not origins:
            raise ValueError("FRONTEND_ORIGINS must contain at least one origin")
        for o in origins:
            if not o.startswith(("http://", "https://")):
                raise ValueError(f"CORS origin must be http(s): {o!r}")
            if "*" in o:
                raise ValueError(f"Wildcard origin not allowed: {o!r}")
        return ", ".join(origins)

    @field_validator("smtp_port")
    @classmethod
    def _validate_smtp_port(cls, v: int) -> int:
        if v not in (25, 465, 587, 2525):
            warnings.warn(f"Unusual SMTP port {v}; expected 25, 465, 587 or 2525")
        return v

    @model_validator(mode="after")
    def _harden(self) -> "Settings":
        is_prod = self.env == "prod"
        # ── Database ──────────────────────────────────────────────────
        if is_prod and not self.database_url:
            raise ValueError("DATABASE_URL is required in prod")
        if self.database_url and "localhost" in self.database_url and is_prod:
            warnings.warn("DATABASE_URL points to localhost in prod — is this intended?")

        # ── JWT secret ────────────────────────────────────────────────
        if not self.supabase_jwt_secret and is_prod:
            raise ValueError("SUPABASE_JWT_SECRET is required in prod")
        if self.supabase_jwt_secret and not _is_strong_secret(self.supabase_jwt_secret):
            msg = "SUPABASE_JWT_SECRET is weak (min 32 chars)."
            if is_prod:
                raise ValueError(msg + " Use: openssl rand -hex 32")
            warnings.warn(msg + " — ok in dev, will fail in prod")

        # ── Admin password ────────────────────────────────────────────
        if not self.admin_password and is_prod:
            raise ValueError("ADMIN_PASSWORD is required in prod")
        if self.admin_password and len(self.admin_password) < 12 and is_prod:
            raise ValueError("ADMIN_PASSWORD must be >=12 chars in prod")
        if self.admin_password and self.admin_password in _WEAK_SECRETS:
            raise ValueError("ADMIN_PASSWORD is a well-known weak value")

        # ── SMTP sanity ───────────────────────────────────────────────
        if self.smtp_host and not self.smtp_from:
            raise ValueError("SMTP_FROM is required when SMTP_HOST is set")
        if self.smtp_host and "@" not in self.smtp_from:
            raise ValueError(f"SMTP_FROM must contain an email address, got {self.smtp_from!r}")
        # If SMTP is configured but no auth, warn — many hosts (Mailpit) are open.
        if self.smtp_host and not self.smtp_user:
            warnings.warn("SMTP_HOST set but SMTP_USER empty — assuming open relay / Mailpit")

        # ── Rate limit sanity ─────────────────────────────────────────
        if self.rate_limit_per_minute > 1000 and is_prod:
            warnings.warn(f"RATE_LIMIT_PER_MINUTE={self.rate_limit_per_minute} is very high for prod")

        return self

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def is_prod(self) -> bool:
        return self.env == "prod"

    @property
    def is_dev(self) -> bool:
        return self.env in ("dev", "test", "ci")

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.frontend_origins.split(",") if o.strip()]

    @property
    def frontend_url(self) -> str:
        return self.cors_origins[0] if self.cors_origins else "http://localhost:3000"

    def masked_summary(self) -> dict[str, str]:
        """Safe-to-log summary — secrets are masked."""
        return {
            "env": self.env,
            "database_url": _mask(self.database_url),
            "supabase_url": self.supabase_url or "(empty)",
            "supabase_jwt_secret": _mask(self.supabase_jwt_secret),
            "supabase_service_key": _mask(self.supabase_service_key),
            "frontend_origins": self.frontend_origins,
            "smtp_host": self.smtp_host or "(disabled)",
            "smtp_port": str(self.smtp_port),
            "smtp_from": self.smtp_from,
            "rate_limit_per_minute": str(self.rate_limit_per_minute),
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
