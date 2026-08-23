import logging
from uuid import NAMESPACE_URL, uuid5

import bcrypt
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, EmailStr
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.core.config import get_settings
from app.core.security import sign_reset_token, sign_token
from app.models import Profile
from app.schemas.identity import ProfileOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


def _google_redirect_uri() -> str:
    s = get_settings()
    return s.google_redirect_uri or f"{s.frontend_url}/api/v1/auth/google/callback"


@router.get("/google/login")
async def google_login():
    s = get_settings()
    if not s.google_client_id:
        raise HTTPException(503, "Google OAuth not configured")
    params = (
        f"client_id={s.google_client_id}"
        f"&redirect_uri={_google_redirect_uri()}"
        f"&response_type=code"
        f"&scope=openid%20email%20profile"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{params}")


@router.get("/google/callback")
async def google_callback(
    db: DbSession,
    code: str = Query(...),
    error: str | None = Query(None),
):
    if error:
        raise HTTPException(400, f"Google OAuth error: {error}")

    s = get_settings()
    if not s.google_client_id or not s.google_client_secret:
        raise HTTPException(503, "Google OAuth not configured")

    import httpx

    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": s.google_client_id,
                "client_secret": s.google_client_secret,
                "redirect_uri": _google_redirect_uri(),
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
        )
        if token_res.status_code != 200:
            raise HTTPException(401, "Failed to exchange Google code for token")

        token_data = token_res.json()
        access_token = token_data.get("access_token")

        user_res = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if user_res.status_code != 200:
            raise HTTPException(401, "Failed to fetch Google user info")

        google_user = user_res.json()

    google_id = str(google_user["id"])
    email = google_user.get("email", "")
    name = google_user.get("name", email.split("@")[0] if email else "User")

    profile_id = uuid5(NAMESPACE_URL, f"https://google.com/user/{google_id}")

    existing = await db.get(Profile, profile_id)
    if existing is None:
        existing = (
            await db.execute(select(Profile).where(Profile.email == email))
        ).scalar_one_or_none()

    # ADMIN_EMAIL from env is the source of truth — any Google login matching it becomes admin
    # (case-insensitive). This makes home1051ab@gmail.com admin via "Continue with Google" without manual DB edit.
    admin_email = (s.admin_email or "").strip().lower()
    is_admin_login = bool(admin_email and email.strip().lower() == admin_email)

    if existing is None:
        profile = Profile(
            id=profile_id,
            email=email,
            full_name=name,
            role="admin" if is_admin_login else "analyst",
            is_active=True,
        )
        db.add(profile)
        await db.commit()
        await db.refresh(profile)
        if is_admin_login:
            logger.info("Google signup promoted %s to admin via ADMIN_EMAIL", email)
    else:
        profile = existing
        # Keep display name in sync; auto-promote to admin if ADMIN_EMAIL now matches but role was lower
        updated = False
        if profile.full_name != name:
            profile.full_name = name
            updated = True
        if is_admin_login and profile.role != "admin":
            profile.role = "admin"
            # Bump token_version so permission change takes effect immediately (old tokens invalidated)
            profile.token_version = (profile.token_version or 0) + 1
            updated = True
            logger.info("Google login auto-promoted %s to admin via ADMIN_EMAIL", email)
        if updated:
            await db.commit()
            await db.refresh(profile)

    token = sign_token(profile.id, profile.email, profile.role, token_version=profile.token_version)

    frontend_url = s.frontend_url
    return RedirectResponse(
        f"{frontend_url}/auth/callback?token={token}",
        status_code=302,
    )


@router.get("/me", response_model=ProfileOut)
async def me(user: CurrentUser) -> ProfileOut:
    return ProfileOut.model_validate(user)


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    full_name: str | None = None
    department: str | None = None


@router.patch("/me", response_model=ProfileOut)
async def update_me(body: ProfileUpdate, user: CurrentUser, db: DbSession) -> ProfileOut:
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    await db.commit()
    await db.refresh(user)
    return ProfileOut.model_validate(user)


class PreferencesOut(BaseModel):
    two_factor: bool = True
    anomaly_alerts: bool = True
    weekly_digest: bool = False


class PreferencesIn(BaseModel):
    two_factor: bool | None = None
    anomaly_alerts: bool | None = None
    weekly_digest: bool | None = None


@router.get("/me/preferences", response_model=PreferencesOut)
async def get_preferences(user: CurrentUser) -> PreferencesOut:
    prefs = user.preferences or {}
    return PreferencesOut(**PreferencesOut(**prefs).model_dump())


@router.patch("/me/preferences", response_model=PreferencesOut)
async def update_preferences(body: PreferencesIn, user: CurrentUser, db: DbSession) -> PreferencesOut:
    current = (user.preferences or {})
    current.update(body.model_dump(exclude_unset=True))
    user.preferences = current
    await db.commit()
    await db.refresh(user)
    return PreferencesOut(**user.preferences)


# ── Email / Password Auth (local dev fallback, no Supabase needed) ──


class LoginBody(BaseModel):
    email: EmailStr
    password: str


class SignupBody(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None


class AuthOut(BaseModel):
    token: str
    user: ProfileOut


class ForgotPasswordBody(BaseModel):
    email: EmailStr


@router.post("/login", response_model=AuthOut)
async def login(body: LoginBody, db: DbSession, request: Request) -> AuthOut:
    from app.core.request_context import current_request_id
    from app.models import AuditLog

    def audit(action: str, user_id=None) -> None:
        db.add(
            AuditLog(
                user_id=user_id,
                action=action,
                entity="auth",
                detail={
                    "email": body.email,
                    "ip_address": request.client.host if request.client else None,
                    "request_id": current_request_id(),
                },
            )
        )

    result = await db.execute(select(Profile).where(Profile.email == body.email))
    profile = result.scalar_one_or_none()
    if profile is None:
        await db.flush()
        audit("auth.login_failed")
        await db.commit()
        raise HTTPException(401, "No account found with this email")
    if not profile.is_active:
        await db.flush()
        audit("auth.login_failed", profile.id)
        await db.commit()
        raise HTTPException(403, "Account is disabled")
    if not profile.password_hash:
        raise HTTPException(401, "This account uses Google sign-in. Please sign in with Google.")
    if not bcrypt.checkpw(body.password.encode(), profile.password_hash.encode()):
        await db.flush()
        audit("auth.login_failed", profile.id)
        await db.commit()
        raise HTTPException(401, "Incorrect password")

    audit("auth.login", profile.id)
    await db.commit()
    token = sign_token(profile.id, profile.email, profile.role, token_version=profile.token_version)
    return AuthOut(token=token, user=ProfileOut.model_validate(profile))


@router.post("/signup", response_model=AuthOut, status_code=201)
async def signup(body: SignupBody, db: DbSession, background_tasks: BackgroundTasks) -> AuthOut:
    existing = await db.execute(select(Profile).where(Profile.email == body.email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(409, "An account with this email already exists")

    if len(body.password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters")
    if not (any(c.isalpha() for c in body.password) and any(c.isdigit() for c in body.password)):
        raise HTTPException(422, "Password must contain at least one letter and one number")
    if body.password.lower() in {"password", "12345678", "admin123", "qwerty123"}:
        raise HTTPException(422, "Password is too common")

    # ADMIN_EMAIL owns the admin role — even email/password signup honours it
    s = get_settings()
    admin_email = (s.admin_email or "").strip().lower()
    assigned_role = "admin" if admin_email and body.email.strip().lower() == admin_email else "analyst"

    profile_id = uuid5(NAMESPACE_URL, f"email://{body.email}")
    pw_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt(rounds=12)).decode()

    profile = Profile(
        id=profile_id,
        email=body.email,
        password_hash=pw_hash,
        full_name=body.full_name,
        role=assigned_role,
        is_active=True,
    )
    db.add(profile)
    await db.commit()
    await db.refresh(profile)

    token = sign_token(profile.id, profile.email, profile.role, token_version=profile.token_version)

    # Best-effort welcome email — never blocks signup or fails the request.
    try:
        from app.services.email.service import send_welcome_email

        background_tasks.add_task(send_welcome_email, profile.email, profile.full_name)
    except Exception:
        logger.exception("failed to queue welcome email for %s", profile.email)

    return AuthOut(token=token, user=ProfileOut.model_validate(profile))


@router.post("/forgot-password")
async def forgot_password(
    body: ForgotPasswordBody, db: DbSession, background_tasks: BackgroundTasks
) -> dict[str, str]:
    result = await db.execute(select(Profile).where(Profile.email == body.email))
    profile = result.scalar_one_or_none()
    if profile is None:
        return {"message": "If an account exists for this email, a reset link has been sent."}

    # Short-lived, purpose-bound reset token (30 min, purpose=reset)
    reset_token = sign_reset_token(profile.id, profile.email, profile.role, token_version=profile.token_version)

    settings = get_settings()
    if settings.smtp_host:
        try:
            from app.services.email.service import send_password_reset_email

            # Background delivery so the endpoint returns quickly and timing
            # doesn't leak whether the address was real.
            background_tasks.add_task(
                send_password_reset_email, profile.email, reset_token, profile.full_name
            )
        except Exception:
            logger.exception("failed to queue reset email for %s", profile.email)
            # Don't fail the request — the user-facing contract is always the
            # same success message regardless of delivery outcome.
    else:
        # Dev fallback — only in non-prod. In prod this branch is never taken
        # because SMTP must be configured (Settings validation).
        if get_settings().is_prod:
            logger.warning("Password reset requested for %s but SMTP not configured (prod)", body.email)
        else:
            logger.info("Password reset token for %s: %s", body.email, reset_token)
            print(f"Password reset token for {body.email}: {reset_token}")

    return {"message": "If an account exists for this email, a reset link has been sent."}


@router.post("/reset-password")
async def reset_password(token: str, new_password: str, db: DbSession) -> AuthOut:
    from app.core.security import verify_reset_token

    try:
        claims = verify_reset_token(token)
    except Exception:
        raise HTTPException(401, "Invalid or expired reset token")

    if len(new_password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters")
    # Enforce basic complexity: at least one letter and one number
    if not (any(c.isalpha() for c in new_password) and any(c.isdigit() for c in new_password)):
        raise HTTPException(422, "Password must contain at least one letter and one number")

    profile = await db.get(Profile, claims.user_id)
    if profile is None:
        raise HTTPException(404, "User not found")
    # Token version must still match — prevents replay after a prior reset
    if claims.token_version is not None and profile.token_version != claims.token_version:
        raise HTTPException(401, "Reset token has been invalidated — please request a new link")

    profile.password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt(rounds=12)).decode()
    # Bump token version so any previously-issued reset tokens are invalidated
    # and existing sessions are rotated (single-use semantics).
    profile.token_version = (profile.token_version or 0) + 1
    await db.commit()
    await db.refresh(profile)

    new_token = sign_token(profile.id, profile.email, profile.role, token_version=profile.token_version)
    return AuthOut(token=new_token, user=ProfileOut.model_validate(profile))


PERMISSIONS_MATRIX: dict[str, list[str]] = {
    "analyst": [
        "dashboard:view",
        "kpis:view",
        "timeseries:view",
        "sales:view",
        "expenses:view",
        "inventory:view",
        "forecasts:view",
        "anomalies:view",
        "trends:view",
        "insights:view",
        "notifications:view",
        "notifications:read",
        "reports:view",
        "reports:download",
        "uploads:create",
    ],
    "manager": [
        "dashboard:view",
        "kpis:view",
        "timeseries:view",
        "sales:view",
        "expenses:view",
        "pnl:view",
        "inventory:view",
        "forecasts:view",
        "anomalies:view",
        "anomalies:update",
        "trends:view",
        "insights:view",
        "insights:pin",
        "alert-rules:manage",
        "notifications:view",
        "notifications:read",
        "reports:view",
        "reports:download",
        "reports:generate",
        "uploads:create",
    ],
    "admin": [
        "dashboard:view",
        "kpis:view",
        "timeseries:view",
        "sales:view",
        "expenses:view",
        "pnl:view",
        "inventory:view",
        "forecasts:view",
        "forecasts:retrain",
        "anomalies:view",
        "anomalies:update",
        "trends:view",
        "insights:view",
        "insights:pin",
        "insights:generate",
        "alert-rules:manage",
        "notifications:view",
        "notifications:read",
        "reports:view",
        "reports:download",
        "reports:generate",
        "uploads:create",
        "users:manage",
        "data-sources:manage",
        "etl:manage",
        "audit-logs:view",
    ],
}


class PermissionsOut(BaseModel):
    role: str
    permissions: list[str]


@router.get("/me/permissions", response_model=PermissionsOut)
async def my_permissions(user: CurrentUser) -> PermissionsOut:
    return PermissionsOut(
        role=user.role,
        permissions=PERMISSIONS_MATRIX.get(user.role, PERMISSIONS_MATRIX["analyst"]),
    )
