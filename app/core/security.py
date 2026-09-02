from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt

from app.core.config import get_settings

LEEWAY_SECONDS = 30  # tolerate small clock skew between Supabase and this host


class AuthError(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True)
class TokenClaims:
    user_id: UUID
    email: str | None
    role: str | None  # application role from app_metadata, NOT Supabase's postgres role
    token_version: int | None = None  # "ver" claim — compared to profiles.token_version
    org_id: UUID | None = None
    purpose: str | None = None


def _jwt_secret() -> str:
    settings = get_settings()
    secret = settings.supabase_jwt_secret or "dev-secret-do-not-use-in-production"
    # In prod a weak secret would have already aborted at Settings validation,
    # but defend-in-depth here as well.
    if len(secret) < 32 and settings.is_prod:
        raise AuthError("JWT secret is not configured securely")
    return secret


def sign_token(
    user_id: UUID,
    email: str,
    role: str = "analyst",
    token_version: int = 0,
    *,
    purpose: str = "auth",
    org_id: UUID | None = None,
    expiry_hours: int | None = None,
) -> str:
    """Create a signed JWT.

    ``purpose`` isolates token classes (``auth`` vs ``reset``) so a long-lived
    auth token cannot be replayed as a password-reset token and vice-versa.
    ``expiry_hours`` overrides the default from settings for special-purpose
    tokens (e.g. short-lived reset tokens).
    ``org_id`` is embedded in app_metadata for tenant scoping (trusted but DB is authoritative).
    """
    settings = get_settings()
    secret = _jwt_secret()
    ttl = timedelta(hours=expiry_hours if expiry_hours is not None else settings.jwt_expiry_hours)
    app_meta: dict[str, str] = {"role": role}
    if org_id is not None:
        app_meta["org_id"] = str(org_id)
    payload = {
        "sub": str(user_id),
        "email": email,
        "app_metadata": app_meta,
        "aud": settings.jwt_audience,
        "exp": datetime.now(UTC) + ttl,
        "iat": datetime.now(UTC),
        "ver": token_version,
        "purpose": purpose,
    }
    # Top-level org_id for RLS / external consumers (PostgREST style)
    if org_id is not None:
        payload["org_id"] = str(org_id)
    return jwt.encode(payload, secret, algorithm="HS256")


def sign_reset_token(user_id: UUID, email: str, role: str = "analyst", token_version: int = 0, org_id: UUID | None = None) -> str:
    """Short-lived password-reset token (30 min by default, isolated purpose)."""
    settings = get_settings()
    ttl_min = settings.jwt_reset_expiry_minutes
    # Reuse sign_token but with minute granularity and purpose=reset
    secret = _jwt_secret()
    app_meta: dict[str, str] = {"role": role}
    if org_id is not None:
        app_meta["org_id"] = str(org_id)
    payload = {
        "sub": str(user_id),
        "email": email,
        "app_metadata": app_meta,
        "aud": settings.jwt_audience,
        "exp": datetime.now(UTC) + timedelta(minutes=ttl_min),
        "iat": datetime.now(UTC),
        "ver": token_version,
        "purpose": "reset",
    }
    if org_id is not None:
        payload["org_id"] = str(org_id)
    return jwt.encode(payload, secret, algorithm="HS256")


def verify_token(token: str, *, expected_purpose: str | None = None) -> TokenClaims:
    """Verify a Supabase-issued JWT (HS256, aud=authenticated) and extract claims.

    ``expected_purpose`` — when given, the ``purpose`` claim must match.
    Legacy tokens without a ``purpose`` claim are accepted only when
    ``expected_purpose`` is ``None`` or ``"auth"`` (backwards compat).
    """
    # In dev/test we allow the fallback dev secret so local `uvicorn` works without SUPABASE_JWT_SECRET;
    # in prod `get_settings()` already hard-fails if the secret is weak/missing.
    settings = get_settings()
    secret = _jwt_secret()
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience=settings.jwt_audience,
            leeway=LEEWAY_SECONDS,
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise AuthError("Token has expired") from e
    except jwt.InvalidTokenError as e:
        raise AuthError("Invalid token") from e

    purpose = payload.get("purpose")
    if expected_purpose is not None:
        # Strict check for purpose-bound tokens (e.g. reset)
        if purpose != expected_purpose:
            raise AuthError(f"Invalid token purpose (expected {expected_purpose!r})")
    elif purpose is not None and purpose != "auth":
        # An auth endpoint that didn't ask for a specific purpose should not
        # accept a reset token — prevents purpose confusion.
        if purpose == "reset":
            raise AuthError("Invalid token purpose")

    try:
        user_id = UUID(payload["sub"])
    except (ValueError, KeyError) as e:
        raise AuthError("Invalid subject claim") from e

    app_metadata = payload.get("app_metadata") or {}
    org_raw = app_metadata.get("org_id") or payload.get("org_id")
    org_id: UUID | None = None
    if org_raw:
        try:
            org_id = UUID(str(org_raw))
        except ValueError:
            org_id = None
    return TokenClaims(
        user_id=user_id,
        email=payload.get("email"),
        role=app_metadata.get("role"),
        token_version=payload.get("ver"),
        org_id=org_id,
        purpose=purpose,
    )


def verify_reset_token(token: str) -> TokenClaims:
    """Verify a short-lived password-reset token."""
    return verify_token(token, expected_purpose="reset")
