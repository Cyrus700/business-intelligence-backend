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


def sign_token(user_id: UUID, email: str, role: str = "analyst", token_version: int = 0) -> str:
    """Create a signed JWT compatible with the existing verify_token flow."""
    settings = get_settings()
    secret = settings.supabase_jwt_secret or "dev-secret-do-not-use-in-production"
    payload = {
        "sub": str(user_id),
        "email": email,
        "app_metadata": {"role": role},
        "aud": settings.jwt_audience,
        "exp": datetime.now(UTC) + timedelta(days=90),
        "iat": datetime.now(UTC),
        "ver": token_version,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def verify_token(token: str) -> TokenClaims:
    """Verify a Supabase-issued JWT (HS256, aud=authenticated) and extract claims."""
    settings = get_settings()
    if not settings.supabase_jwt_secret:
        raise AuthError("Server auth is not configured")
    try:
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience=settings.jwt_audience,
            leeway=LEEWAY_SECONDS,
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise AuthError("Token has expired") from e
    except jwt.InvalidTokenError as e:
        raise AuthError("Invalid token") from e

    try:
        user_id = UUID(payload["sub"])
    except (ValueError, KeyError) as e:
        raise AuthError("Invalid subject claim") from e

    app_metadata = payload.get("app_metadata") or {}
    return TokenClaims(
        user_id=user_id,
        email=payload.get("email"),
        role=app_metadata.get("role"),
        token_version=payload.get("ver"),
    )
