from uuid import NAMESPACE_URL, uuid5

import bcrypt
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, ConfigDict
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.core.config import get_settings
from app.core.security import sign_token
from app.models import Profile
from app.schemas.identity import ProfileOut

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

    if existing is None:
        profile = Profile(
            id=profile_id,
            email=email,
            full_name=name,
            role="analyst",
            is_active=True,
        )
        db.add(profile)
        await db.commit()
        await db.refresh(profile)
    else:
        profile = existing
        if profile.full_name != name:
            profile.full_name = name
        await db.commit()
        await db.refresh(profile)

    token = sign_token(profile.id, profile.email, profile.role)

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
async def login(body: LoginBody, db: DbSession) -> AuthOut:
    result = await db.execute(select(Profile).where(Profile.email == body.email))
    profile = result.scalar_one_or_none()
    if profile is None:
        raise HTTPException(401, "No account found with this email")
    if not profile.is_active:
        raise HTTPException(403, "Account is disabled")
    if not profile.password_hash:
        raise HTTPException(401, "This account uses Google sign-in. Please sign in with Google.")
    if not bcrypt.checkpw(body.password.encode(), profile.password_hash.encode()):
        raise HTTPException(401, "Incorrect password")

    token = sign_token(profile.id, profile.email, profile.role)
    return AuthOut(token=token, user=ProfileOut.model_validate(profile))


@router.post("/signup", response_model=AuthOut, status_code=201)
async def signup(body: SignupBody, db: DbSession) -> AuthOut:
    existing = await db.execute(select(Profile).where(Profile.email == body.email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(409, "An account with this email already exists")

    if len(body.password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters")

    profile_id = uuid5(NAMESPACE_URL, f"email://{body.email}")
    pw_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()

    profile = Profile(
        id=profile_id,
        email=body.email,
        password_hash=pw_hash,
        full_name=body.full_name,
        role="analyst",
        is_active=True,
    )
    db.add(profile)
    await db.commit()
    await db.refresh(profile)

    token = sign_token(profile.id, profile.email, profile.role)
    return AuthOut(token=token, user=ProfileOut.model_validate(profile))


@router.post("/forgot-password")
async def forgot_password(body: ForgotPasswordBody, db: DbSession) -> dict[str, str]:
    result = await db.execute(select(Profile).where(Profile.email == body.email))
    profile = result.scalar_one_or_none()
    if profile is None:
        return {"message": "If an account exists for this email, a reset link has been sent."}

    settings = get_settings()
    reset_token = sign_token(profile.id, profile.email, profile.role)

    if settings.smtp_host:
        try:
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(
                f"Reset your password here:\n{settings.frontend_url}/reset-password?token={reset_token}"
            )
            msg["Subject"] = "Insightful — Password Reset"
            msg["From"] = settings.smtp_from
            msg["To"] = body.email

            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                if settings.smtp_user:
                    server.login(settings.smtp_user, settings.smtp_password)
                server.send_message(msg)
        except Exception:
            raise HTTPException(502, "Failed to send reset email")
    else:
        print(f"Password reset token for {body.email}: {reset_token}")

    return {"message": "If an account exists for this email, a reset link has been sent."}


@router.post("/reset-password")
async def reset_password(token: str, new_password: str, db: DbSession) -> AuthOut:
    from app.core.security import verify_token

    try:
        claims = verify_token(token)
    except Exception:
        raise HTTPException(401, "Invalid or expired reset token")

    if len(new_password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters")

    profile = await db.get(Profile, claims.user_id)
    if profile is None:
        raise HTTPException(404, "User not found")

    profile.password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    await db.commit()
    await db.refresh(profile)

    new_token = sign_token(profile.id, profile.email, profile.role)
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


@router.get("/permissions", response_model=PermissionsOut)
async def permissions(user: CurrentUser) -> PermissionsOut:
    return PermissionsOut(
        role=user.role,
        permissions=PERMISSIONS_MATRIX.get(user.role, PERMISSIONS_MATRIX["analyst"]),
    )
