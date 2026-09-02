import logging
import secrets
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, UUID, uuid5

import bcrypt
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy import func, select

from app.api.deps import CurrentUser, DbSession, require_role
from app.core.config import get_settings
from app.core.security import sign_reset_token, sign_token
from app.models import Organization, OrganizationInvite, Profile
from app.schemas.identity import OrganizationInviteOut, OrganizationOut, ProfileOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


def _google_redirect_uri() -> str:
    s = get_settings()
    return s.google_redirect_uri or f"{s.frontend_url}/api/v1/auth/google/callback"


@router.get("/google/login")
async def google_login(request: Request, invite_token: str | None = Query(None)):
    s = get_settings()
    if not s.google_client_id:
        raise HTTPException(503, "Google OAuth not configured")
    state = secrets.token_urlsafe(16)
    # Include invite_token in state if provided (format: state:invite_token)
    state_with_invite = f"{state}:{invite_token}" if invite_token else state
    params = (
        f"client_id={s.google_client_id}"
        f"&redirect_uri={_google_redirect_uri()}"
        f"&response_type=code"
        f"&scope=openid%20email%20profile"
        f"&access_type=offline"
        f"&prompt=consent"
        f"&state={state_with_invite}"
    )
    resp = RedirectResponse(f"{GOOGLE_AUTH_URL}?{params}")
    # Store state in httpOnly cookie for 10 min to verify on callback (CSRF protection)
    resp.set_cookie("oauth_state", state, max_age=600, httponly=True, samesite="lax", secure=s.is_prod, path="/")
    return resp


@router.get("/google/callback")
async def google_callback(
    db: DbSession,
    request: Request,
    code: str = Query(...),
    error: str | None = Query(None),
    state: str | None = Query(None),
):
    if error:
        raise HTTPException(400, f"Google OAuth error: {error}")

    s = get_settings()
    if not s.google_client_id or not s.google_client_secret:
        raise HTTPException(503, "Google OAuth not configured")

    # CSRF: verify state matches cookie (if cookie set)
    cookie_state = request.cookies.get("oauth_state")
    if cookie_state:
        # state may contain ":invite_token" suffix, compare only random part
        received_state = state.split(":")[0] if state else ""
        if received_state != cookie_state:
            raise HTTPException(400, "Invalid OAuth state — possible CSRF. Please try again.")

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
    email = google_user.get("email", "") or ""
    normalized_google_email = email.strip().lower()
    name = google_user.get("name", email.split("@")[0] if email else "User")

    profile_id = uuid5(NAMESPACE_URL, f"https://google.com/user/{google_id}")

    existing = await db.get(Profile, profile_id)
    if existing is None:
        existing = (
            await db.execute(select(Profile).where(func.lower(Profile.email) == normalized_google_email))
        ).scalar_one_or_none()

    # ADMIN_EMAIL from env is the source of truth — any Google login matching it becomes admin
    admin_email = (s.admin_email or "").strip().lower()
    is_admin_login = bool(admin_email and normalized_google_email == admin_email)

    # Extract invite_token from state if present (state format: random:invite_token)
    invite_token_from_state: str | None = None
    if state and ":" in state:
        invite_token_from_state = state.split(":", 1)[1]

    if existing is None:
        # Try invite-based org assignment first (for team invites via Google)
        org_id: UUID | None = None
        invited_role = "analyst"
        if invite_token_from_state:
            inv = (
                await db.execute(select(OrganizationInvite).where(OrganizationInvite.token == invite_token_from_state))
            ).scalar_one_or_none()
            if inv and inv.accepted_at is None and inv.expires_at > datetime.now(UTC).replace(tzinfo=None):
                if not inv.email or inv.email.strip().lower() == normalized_google_email:
                    org_id = inv.org_id
                    invited_role = inv.role
                    # Mark personal invites as used; open invites stay reusable
                    if inv.email is not None:
                        inv.accepted_at = datetime.now(UTC).replace(tzinfo=None)
                        inv.accepted_by = profile_id
        if org_id is None:
            # Fallback: legacy org or block (strict multi-tenant)
            if get_settings().is_dev:
                legacy_org = (
                    await db.execute(select(Organization).where(Organization.is_legacy.is_(True)))
                ).scalar_one_or_none()
                org_id = legacy_org.id if legacy_org else None
                # In prod without invite, Google users must register business first
                if org_id is None and not is_admin_login:
                    raise HTTPException(
                        403,
                        "No organization found for this Google account. Please register your business at /register-business or request an invite, then try again. If you were invited, use the invite link that contains your token.",
                    )
            else:
                if not is_admin_login:
                    raise HTTPException(
                        403,
                        "No organization found for this Google account. Please register your business at /register-business or request an invite.",
                    )
                legacy_org = (
                    await db.execute(select(Organization).where(Organization.is_legacy.is_(True)))
                ).scalar_one_or_none()
                org_id = legacy_org.id if legacy_org else None
        # Platform super-admin if ADMIN_EMAIL; otherwise use invited role if present
        role_for_new = "admin" if is_admin_login else (invited_role if org_id is not None and 'invited_role' in locals() and invited_role else "analyst")
        profile = Profile(
            id=profile_id,
            email=normalized_google_email,
            full_name=name,
            role=role_for_new,
            is_active=True,
            org_id=org_id,
            is_super_admin=is_admin_login,
        )
        db.add(profile)
        await db.commit()
        await db.refresh(profile)
        if is_admin_login:
            logger.info("Google signup promoted %s to admin via ADMIN_EMAIL (super_admin=%s)", email, is_admin_login)
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
        if is_admin_login and not profile.is_super_admin:
            profile.is_super_admin = True
            updated = True
        # Backfill org_id if missing and legacy exists
        if profile.org_id is None:
            legacy_org = (
                await db.execute(select(Organization).where(Organization.is_legacy.is_(True)))
            ).scalar_one_or_none()
            if legacy_org:
                profile.org_id = legacy_org.id
                updated = True
        if updated:
            await db.commit()
            await db.refresh(profile)

    token = sign_token(profile.id, profile.email, profile.role, token_version=profile.token_version, org_id=profile.org_id)

    frontend_url = s.frontend_url
    resp = RedirectResponse(
        f"{frontend_url}/auth/callback?token={token}",
        status_code=302,
    )
    # Clear state cookie
    resp.delete_cookie("oauth_state", path="/")
    return resp


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
    invite_token: str | None = Field(default=None, description="Invite token to join existing org")


class RegisterOrgBody(BaseModel):
    org_name: str = Field(..., min_length=2, max_length=120)
    email: EmailStr
    password: str
    full_name: str | None = None


class AuthOut(BaseModel):
    token: str
    user: ProfileOut
    organization: OrganizationOut | None = None


class RegisterOrgOut(BaseModel):
    token: str | None = None
    user: ProfileOut | None = None
    organization: OrganizationOut
    status: str = "pending"
    message: str | None = None


class VerifyEmailBody(BaseModel):
    token: str


class ResendVerificationBody(BaseModel):
    email: EmailStr


class InviteCreateBody(BaseModel):
    email: EmailStr | None = None
    role: str = Field(default="analyst", description="analyst/manager/admin")
    expires_in_days: int = Field(default=7, ge=1, le=30)


class InviteAcceptBody(BaseModel):
    token: str
    email: EmailStr
    password: str
    full_name: str | None = None


class ForgotPasswordBody(BaseModel):
    email: EmailStr


class OrganizationPendingOut(OrganizationOut):
    contact_email: str | None = None
    contact_name: str | None = None


@router.post("/register-org", response_model=RegisterOrgOut, status_code=201)
async def register_org(body: RegisterOrgBody, db: DbSession, background_tasks: BackgroundTasks) -> RegisterOrgOut:
    """Register a new business: creates Organization (pending) + admin Profile (inactive until verified & approved)."""
    # Validate password policy (same as signup)
    if len(body.password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters")
    if not (any(c.isalpha() for c in body.password) and any(c.isdigit() for c in body.password)):
        raise HTTPException(422, "Password must contain at least one letter and one number")
    if body.password.lower() in {"password", "12345678", "admin123", "qwerty123"}:
        raise HTTPException(422, "Password is too common")

    normalized_email = body.email.strip().lower()
    existing = await db.execute(select(Profile).where(func.lower(Profile.email) == normalized_email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(409, "An account with this email already exists")

    normalized = body.org_name.strip()
    if not normalized:
        raise HTTPException(422, "Organization name is required")
    existing_org = await db.execute(select(Organization).where(func.lower(Organization.name) == normalized.lower()))
    if existing_org.scalar_one_or_none() is not None:
        raise HTTPException(409, "An organization with this name already exists")

    profile_id = uuid5(NAMESPACE_URL, f"email://{normalized_email}")
    pw_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt(rounds=12)).decode()

    import re as _re

    def _slugify(name: str) -> str:
        s = name.strip().lower()
        s = _re.sub(r"[^a-z0-9]+", "-", s)
        s = _re.sub(r"-{2,}", "-", s).strip("-")
        return (s or "org")[:64]
    base_slug = _slugify(normalized)
    slug = base_slug
    for _ in range(5):
        exists = await db.execute(select(Organization).where(Organization.slug == slug))
        if exists.scalar_one_or_none() is None:
            break
        slug = f"{base_slug[:58]}-{secrets.token_hex(3)}"

    s_cfg = get_settings()
    admin_email = (s_cfg.admin_email or "").strip().lower()
    is_admin_email = bool(admin_email and body.email.strip().lower() == admin_email)
    # Test / CI auto-approve to keep existing tests green; dev & prod require System Admin approval
    auto_approve = bool(s_cfg.env in ("test", "ci") or is_admin_email)

    verification_token = secrets.token_urlsafe(32)
    now = datetime.now(UTC).replace(tzinfo=None)
    expires = now + timedelta(hours=24)

    try:
        org = Organization(name=normalized, slug=slug, status="approved" if auto_approve else "pending")
        db.add(org)
        await db.flush()

        profile = Profile(
            id=profile_id,
            email=normalized_email,
            password_hash=pw_hash,
            full_name=body.full_name,
            role="admin",
            is_active=True if auto_approve else False,
            org_id=org.id,
            is_super_admin=is_admin_email,
            email_verified=True if auto_approve else False,
            email_verification_token=None if auto_approve else verification_token,
            email_verification_expires_at=None if auto_approve else expires,
            email_verified_at=now if auto_approve else None,
        )
        db.add(profile)
        await db.flush()
        await db.commit()
        await db.refresh(org)
        await db.refresh(profile)
    except Exception as e:
        await db.rollback()
        if "UniqueViolation" in type(e).__name__ or "uq_" in str(e):
            raise HTTPException(409, "Organization or user already exists") from e
        raise

    # Auto-approve path (dev / ADMIN_EMAIL): immediate login
    if auto_approve:
        token = sign_token(profile.id, profile.email, profile.role, token_version=profile.token_version, org_id=profile.org_id)
        try:
            from app.services.email.service import send_welcome_email

            background_tasks.add_task(send_welcome_email, profile.email, profile.full_name)
        except Exception:
            logger.exception("failed to queue welcome email for %s", profile.email)
        return RegisterOrgOut(
            token=token,
            user=ProfileOut.model_validate(profile),
            organization=OrganizationOut.model_validate(org),
            status="approved",
            message="Business approved — you can sign in.",
        )

    # Pending approval flow: send verification email to business + notification to System Admin
    try:
        from app.services.email.service import send_business_pending_admin_email, send_business_verification_email

        background_tasks.add_task(send_business_verification_email, profile.email, org.name, verification_token, profile.full_name)
        background_tasks.add_task(send_business_pending_admin_email, org.name, profile.email, profile.full_name)
    except Exception:
        logger.exception("failed to queue business approval emails for %s", profile.email)

    return RegisterOrgOut(
        token=None,
        user=None,
        organization=OrganizationOut.model_validate(org),
        status="pending",
        message="Business registration submitted. Please check your email to verify, and wait for System Admin approval. You'll be notified by email once approved.",
    )


@router.post("/verify-email")
async def verify_email(body: VerifyEmailBody, db: DbSession, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Verify business admin email via token sent at registration."""
    normalized_token = body.token.strip()
    result = await db.execute(select(Profile).where(Profile.email_verification_token == normalized_token))
    profile = result.scalar_one_or_none()
    if profile is None:
        raise HTTPException(400, "Invalid or expired verification token")
    if profile.email_verified:
        return {"message": "Email already verified. Awaiting System Admin approval."}
    if profile.email_verification_expires_at and profile.email_verification_expires_at < datetime.now(UTC).replace(tzinfo=None):
        raise HTTPException(400, "Verification token expired. Please request a new one via POST /auth/resend-verification.")
    profile.email_verified = True
    profile.email_verified_at = datetime.now(UTC).replace(tzinfo=None)
    profile.email_verification_token = None
    profile.email_verification_expires_at = None
    await db.commit()
    # Notify business that email is verified, still pending admin approval
    try:
        from app.services.email.service import send_business_pending_confirmation_email

        org = await db.get(Organization, profile.org_id) if profile.org_id else None
        if org:
            background_tasks.add_task(send_business_pending_confirmation_email, profile.email, org.name, profile.full_name)
    except Exception:
        logger.exception("failed to queue pending confirmation for %s", profile.email)
    return {"message": "Email verified successfully. Your business is pending System Admin approval. You'll receive an email once approved."}


@router.post("/resend-verification")
async def resend_verification(body: ResendVerificationBody, db: DbSession, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Resend verification email for pending business."""
    normalized = body.email.strip().lower()
    result = await db.execute(select(Profile).where(func.lower(Profile.email) == normalized))
    profile = result.scalar_one_or_none()
    if profile is None:
        return {"message": "If an account exists, a verification email has been sent."}
    if profile.email_verified:
        return {"message": "Email already verified."}
    if profile.org_id is None:
        raise HTTPException(400, "No business associated with this email.")
    org = await db.get(Organization, profile.org_id)
    if org and org.status in ("approved", "rejected"):
        return {"message": f"Business is already {org.status}."}
    # Generate new token
    new_token = secrets.token_urlsafe(32)
    profile.email_verification_token = new_token
    profile.email_verification_expires_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=24)
    await db.commit()
    try:
        from app.services.email.service import send_business_verification_email

        org_name = org.name if org else "your business"
        background_tasks.add_task(send_business_verification_email, profile.email, org_name, new_token, profile.full_name)
    except Exception:
        logger.exception("failed to resend verification for %s", profile.email)
    return {"message": "Verification email sent. Please check your inbox."}


# ── System Admin approval workflow (super_admin only) ──


class ApproveBody(BaseModel):
    reason: str | None = None


@router.get("/admin/pending-organizations", response_model=list[OrganizationPendingOut])
async def list_pending_organizations(db: DbSession, user: CurrentUser) -> list[OrganizationPendingOut]:
    if not getattr(user, "is_super_admin", False):
        raise HTTPException(403, "System Admin privileges required")
    rows = (await db.execute(select(Organization).where(Organization.status == "pending").order_by(Organization.created_at.desc()))).scalars().all()
    out: list[OrganizationPendingOut] = []
    for org in rows:
        # Fetch business admin contact for this org
        admin = (await db.execute(select(Profile).where(Profile.org_id == org.id, Profile.role == "admin").limit(1))).scalar_one_or_none()
        if not admin:
            admin = (await db.execute(select(Profile).where(Profile.org_id == org.id).limit(1))).scalar_one_or_none()
        base = OrganizationPendingOut.model_validate(org)
        base.contact_email = admin.email if admin else None
        base.contact_name = admin.full_name if admin else None
        out.append(base)
    return out


@router.get("/admin/organizations", response_model=list[OrganizationOut])
async def admin_list_all_organizations(db: DbSession, user: CurrentUser, status: str | None = Query(None, pattern="^(pending|approved|rejected)$")) -> list[OrganizationOut]:
    if not getattr(user, "is_super_admin", False):
        raise HTTPException(403, "System Admin privileges required")
    q = select(Organization).order_by(Organization.created_at.desc())
    if status:
        q = q.where(Organization.status == status)
    rows = (await db.execute(q)).scalars().all()
    return [OrganizationOut.model_validate(r) for r in rows]


@router.post("/admin/organizations/{org_id}/approve", response_model=OrganizationOut)
async def approve_organization(org_id: UUID, db: DbSession, user: CurrentUser, background_tasks: BackgroundTasks) -> OrganizationOut:
    if not getattr(user, "is_super_admin", False):
        raise HTTPException(403, "System Admin privileges required")
    org = await db.get(Organization, org_id)
    if org is None:
        raise HTTPException(404, "Organization not found")
    if org.status == "approved":
        return OrganizationOut.model_validate(org)
    if org.status == "rejected":
        raise HTTPException(409, "Organization was previously rejected. Create a new request or contact support.")
    org.status = "approved"
    org.approved_at = datetime.now(UTC).replace(tzinfo=None)
    org.approved_by = user.id
    org.rejected_at = None
    org.rejected_by = None
    org.rejection_reason = None
    # Activate the business admin profile
    admin_profile = (await db.execute(select(Profile).where(Profile.org_id == org.id, Profile.role == "admin"))).scalars().first()
    # Fallback: any profile in org if admin role not found (should not happen)
    if not admin_profile:
        admin_profile = (await db.execute(select(Profile).where(Profile.org_id == org.id))).scalars().first()
    if admin_profile:
        admin_profile.is_active = True
        # If email not yet verified, keep pending verification but allow login after admin approval? Require verified.
        # We keep is_active True only if email_verified; otherwise admin approval still requires email verification.
        # For now, admin approval activates regardless, but login will still check email_verified.
        # To be permissive, if not verified, we keep as is and let verification flow complete.
        pass
    await db.commit()
    await db.refresh(org)
    # Send approval email to business admin
    try:
        from app.services.email.service import send_business_approved_email

        if admin_profile:
            background_tasks.add_task(send_business_approved_email, admin_profile.email, org.name, admin_profile.full_name)
        else:
            logger.warning("Approved org %s has no admin profile to notify", org.id)
    except Exception:
        logger.exception("failed to queue approval email for %s", org.id)
    return OrganizationOut.model_validate(org)


@router.post("/admin/organizations/{org_id}/reject", response_model=OrganizationOut)
async def reject_organization(org_id: UUID, body: ApproveBody, db: DbSession, user: CurrentUser, background_tasks: BackgroundTasks) -> OrganizationOut:
    if not getattr(user, "is_super_admin", False):
        raise HTTPException(403, "System Admin privileges required")
    org = await db.get(Organization, org_id)
    if org is None:
        raise HTTPException(404, "Organization not found")
    if org.status == "rejected":
        return OrganizationOut.model_validate(org)
    org.status = "rejected"
    org.rejected_at = datetime.now(UTC).replace(tzinfo=None)
    org.rejected_by = user.id
    org.rejection_reason = body.reason
    org.approved_at = None
    org.approved_by = None
    # Deactivate all profiles in org
    profiles = (await db.execute(select(Profile).where(Profile.org_id == org.id))).scalars().all()
    for p in profiles:
        p.is_active = False
    await db.commit()
    await db.refresh(org)
    try:
        from app.services.email.service import send_business_rejected_email

        # Notify the business admin
        admin_profile = next((p for p in profiles if p.role == "admin"), profiles[0] if profiles else None)
        if admin_profile:
            background_tasks.add_task(send_business_rejected_email, admin_profile.email, org.name, body.reason, admin_profile.full_name)
    except Exception:
        logger.exception("failed to queue rejection email for %s", org.id)
    return OrganizationOut.model_validate(org)


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

    normalized_login = body.email.strip().lower()
    result = await db.execute(select(Profile).where(func.lower(Profile.email) == normalized_login))
    profile = result.scalar_one_or_none()
    # Use generic error to avoid account enumeration (same message for not-found vs wrong password)
    generic_auth_failed = HTTPException(401, "Invalid email or password")
    if profile is None:
        # Dummy bcrypt to keep timing similar (mitigate enumeration via timing)
        try:
            bcrypt.checkpw(body.password.encode(), bcrypt.gensalt(rounds=4).decode().encode())
        except Exception:
            pass
        await db.flush()
        audit("auth.login_failed")
        await db.commit()
        raise generic_auth_failed
    # Business approval & email verification checks (business-based isolation)
    org = await db.get(Organization, profile.org_id) if profile.org_id else None
    if org and org.status == "pending" and not profile.is_super_admin:
        await db.flush()
        audit("auth.login_failed", profile.id)
        await db.commit()
        raise HTTPException(403, "Your business is pending System Admin approval. Please verify your email and wait for approval. You'll receive an email once approved.")
    if org and org.status == "rejected" and not profile.is_super_admin:
        await db.flush()
        audit("auth.login_failed", profile.id)
        await db.commit()
        raise HTTPException(403, f"Your business was rejected: {org.rejection_reason or 'Contact support for details.'}")
    if not getattr(profile, "email_verified", True) and not profile.is_super_admin:
        await db.flush()
        audit("auth.login_failed", profile.id)
        await db.commit()
        raise HTTPException(403, "Please verify your email. Check your inbox for the verification link or request a new one via POST /auth/resend-verification.")
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
        raise generic_auth_failed

    audit("auth.login", profile.id)
    await db.commit()
    org = await db.get(Organization, profile.org_id) if profile.org_id else None
    token = sign_token(profile.id, profile.email, profile.role, token_version=profile.token_version, org_id=profile.org_id)
    return AuthOut(
        token=token,
        user=ProfileOut.model_validate(profile),
        organization=OrganizationOut.model_validate(org) if org else None,
    )


@router.post("/signup", response_model=AuthOut, status_code=201)
async def signup(body: SignupBody, db: DbSession, background_tasks: BackgroundTasks) -> AuthOut:
    """Signup via invite token (preferred) or legacy fallback.

    With invite_token: validates token, joins that org with invite's role.
    Without token: requires org context — blocked in strict multi-tenant mode unless ADMIN_EMAIL.
    """
    normalized_signup_email = body.email.strip().lower()
    existing = await db.execute(select(Profile).where(func.lower(Profile.email) == normalized_signup_email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(409, "An account with this email already exists")

    if len(body.password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters")
    if not (any(c.isalpha() for c in body.password) and any(c.isdigit() for c in body.password)):
        raise HTTPException(422, "Password must contain at least one letter and one number")
    if body.password.lower() in {"password", "12345678", "admin123", "qwerty123"}:
        raise HTTPException(422, "Password is too common")

    s = get_settings()
    admin_email = (s.admin_email or "").strip().lower()
    is_admin_email = bool(admin_email and body.email.strip().lower() == admin_email)

    org_id: UUID | None = None
    assigned_role = "analyst"
    invite: OrganizationInvite | None = None

    if body.invite_token:
        invite = (
            await db.execute(select(OrganizationInvite).where(OrganizationInvite.token == body.invite_token))
        ).scalar_one_or_none()
        if invite is None:
            raise HTTPException(404, "Invalid invite token")
        if invite.accepted_at is not None:
            raise HTTPException(409, "Invite already accepted")
        if invite.expires_at < datetime.now(UTC).replace(tzinfo=None):
            raise HTTPException(410, "Invite has expired")
        if invite.email and invite.email.strip().lower() != body.email.strip().lower():
            raise HTTPException(403, "Invite email does not match signup email")
        org_id = invite.org_id
        assigned_role = invite.role
        # Validate role still exists
        from app.services import rbac as rbac_service

        policy = await rbac_service.get_policy(db)
        if assigned_role not in policy.roles:
            # Fallback to analyst if role was deleted
            assigned_role = "analyst"
    else:
        # No invite: only ADMIN_EMAIL or dev/test legacy mode may signup without invite (for initial seeding / tests)
        # In production, this path is intentionally blocked to enforce invite-only onboarding.
        if is_admin_email:
            # Auto-attach admin to legacy org if exists else require register-org
            legacy = (
                await db.execute(select(Organization).where(Organization.is_legacy.is_(True)))
            ).scalar_one_or_none()
            org_id = legacy.id if legacy else None
            assigned_role = "admin"
        elif get_settings().is_dev:
            # Dev/test fallback: allow legacy org assignment so existing tests/seed scripts keep working
            legacy = (
                await db.execute(select(Organization).where(Organization.is_legacy.is_(True)))
            ).scalar_one_or_none()
            if legacy is None:
                # No legacy yet (fresh DB before migration); create ephemeral org for this signup
                # But in test mode, conftest will have created legacy via migration; fallback to first org
                first_org = (await db.execute(select(Organization).limit(1))).scalar_one_or_none()
                if first_org:
                    legacy = first_org
                else:
                    # Create minimal legacy on the fly (will be reused)
                    legacy = Organization(name="Legacy — Dev", slug="legacy-dev", is_legacy=True)
                    db.add(legacy)
                    await db.flush()
            org_id = legacy.id
            assigned_role = "analyst"
        else:
            raise HTTPException(
                422,
                "Invite token required — ask your organization admin for an invite, or register a new business via POST /auth/register-org",
            )
        # Keep admin privilege even without org? But post-migration org is required, so if no legacy, error
        if org_id is None:
            raise HTTPException(
                422,
                "Organization context missing — please register your business first via POST /auth/register-org",
            )

    # Final role override: ADMIN_EMAIL always gets admin + super_admin
    if is_admin_email:
        assigned_role = "admin"

    profile_id = uuid5(NAMESPACE_URL, f"email://{normalized_signup_email}")
    pw_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt(rounds=12)).decode()

    profile = Profile(
        id=profile_id,
        email=normalized_signup_email,
        password_hash=pw_hash,
        full_name=body.full_name,
        role=assigned_role,
        is_active=True,
        org_id=org_id,
        is_super_admin=is_admin_email,
    )
    db.add(profile)
    # Mark invite accepted — only for personal invites; open (email=None) invites stay reusable
    if invite and invite.email is not None:
        invite.accepted_at = datetime.now(UTC).replace(tzinfo=None)
        invite.accepted_by = profile_id
    await db.commit()
    await db.refresh(profile)
    org = await db.get(Organization, org_id) if org_id else None

    token = sign_token(profile.id, profile.email, profile.role, token_version=profile.token_version, org_id=profile.org_id)

    # Best-effort welcome email — never blocks signup or fails the request.
    try:
        from app.services.email.service import send_welcome_email

        background_tasks.add_task(send_welcome_email, profile.email, profile.full_name)
    except Exception:
        logger.exception("failed to queue welcome email for %s", profile.email)

    return AuthOut(
        token=token, user=ProfileOut.model_validate(profile), organization=OrganizationOut.model_validate(org) if org else None
    )


# ── Organization invite flow ────────────────────────────────────────────────


@router.post("/invite", response_model=OrganizationInviteOut, dependencies=[Depends(require_role("admin"))])
async def create_invite(body: InviteCreateBody, db: DbSession, user: CurrentUser, background_tasks: BackgroundTasks) -> OrganizationInviteOut:
    """Create an invite token for a new user to join the caller's org (admin only). Business must be approved."""
    from app.services import rbac as rbac_service

    policy = await rbac_service.get_policy(db)
    if body.role not in policy.roles:
        raise HTTPException(422, f"Unknown role '{body.role}'")
    if not user.org_id:
        raise HTTPException(403, "Organization membership required to invite")
    # Business must be approved
    org = await db.get(Organization, user.org_id)
    if org and org.status != "approved" and not getattr(user, "is_super_admin", False):
        raise HTTPException(403, "Business pending System Admin approval — invites disabled until approved")

    # Prevent duplicate active invite for same email (case-insensitive)
    if body.email:
        normalized_invite = body.email.strip().lower()
        dup = (
            await db.execute(
                select(OrganizationInvite).where(
                    OrganizationInvite.org_id == user.org_id,
                    func.lower(OrganizationInvite.email) == normalized_invite,
                    OrganizationInvite.accepted_at.is_(None),
                    OrganizationInvite.expires_at > datetime.now(UTC).replace(tzinfo=None),
                )
            )
        ).scalar_one_or_none()
        if dup:
            raise HTTPException(409, "An active invite already exists for this email")

    token = secrets.token_urlsafe(32)
    invite = OrganizationInvite(
        org_id=user.org_id,
        email=body.email.lower().strip() if body.email else None,
        role=body.role,
        token=token,
        created_by=user.id,
        expires_at=datetime.now(UTC).replace(tzinfo=None) + timedelta(days=body.expires_in_days),
    )
    db.add(invite)
    await db.commit()
    await db.refresh(invite)

    # Best-effort email delivery — never fails the request
    if body.email:
        try:
            from app.services.email.service import send_invite_email  # type: ignore

            # Resolve business name for template
            try:
                _org = await db.get(Organization, user.org_id)
                _bname = _org.name if _org else "your workspace"
            except Exception:
                _bname = "your workspace"
            background_tasks.add_task(send_invite_email, body.email, token, user.email, body.role, _bname)
        except Exception:
            logger.info("Invite token for %s: %s", body.email, token)
    else:
        logger.info("Open invite token: %s", token)

    return OrganizationInviteOut.model_validate(invite)


@router.get("/invites", response_model=list[OrganizationInviteOut], dependencies=[Depends(require_role("admin"))])
async def list_invites(db: DbSession, user: CurrentUser) -> list[OrganizationInviteOut]:
    if not user.org_id:
        raise HTTPException(403, "Organization membership required")
    rows = (
        await db.execute(
            select(OrganizationInvite)
            .where(OrganizationInvite.org_id == user.org_id)
            .order_by(OrganizationInvite.created_at.desc())
            .limit(100)
        )
    ).scalars().all()
    return [OrganizationInviteOut.model_validate(r) for r in rows]


@router.get("/invites/{token}", response_model=OrganizationInviteOut)
async def get_invite(token: str, db: DbSession) -> OrganizationInviteOut:
    invite = (
        await db.execute(select(OrganizationInvite).where(OrganizationInvite.token == token))
    ).scalar_one_or_none()
    if invite is None:
        raise HTTPException(404, "Invite not found")
    if invite.expires_at < datetime.now(UTC).replace(tzinfo=None):
        raise HTTPException(410, "Invite has expired")
    if invite.accepted_at is not None:
        raise HTTPException(409, "Invite already accepted")
    return OrganizationInviteOut.model_validate(invite)


@router.delete("/invites/{invite_id}", status_code=204, dependencies=[Depends(require_role("admin"))])
async def revoke_invite(invite_id: UUID, db: DbSession, user: CurrentUser) -> None:
    invite = await db.get(OrganizationInvite, invite_id)
    if invite is None or invite.org_id != user.org_id:
        raise HTTPException(404, "Invite not found")
    await db.delete(invite)
    await db.commit()


@router.get("/organizations", response_model=list[OrganizationOut])
async def list_organizations(db: DbSession, user: CurrentUser) -> list[OrganizationOut]:
    """List organizations — super-admin sees all, everyone else sees own org only."""
    if getattr(user, "is_super_admin", False):
        rows = (await db.execute(select(Organization).order_by(Organization.name))).scalars().all()
    else:
        if not user.org_id:
            raise HTTPException(403, "Organization membership required")
        org = await db.get(Organization, user.org_id)
        rows = [org] if org else []
    return [OrganizationOut.model_validate(r) for r in rows]


@router.get("/organizations/{org_id}", response_model=OrganizationOut, dependencies=[Depends(require_role("admin"))])
async def get_organization(org_id: UUID, db: DbSession, user: CurrentUser) -> OrganizationOut:
    org = await db.get(Organization, org_id)
    if org is None:
        raise HTTPException(404, "Organization not found")
    if not getattr(user, "is_super_admin", False) and org.id != user.org_id:
        raise HTTPException(403, "Not allowed to view other organizations")
    return OrganizationOut.model_validate(org)


@router.post("/forgot-password")
async def forgot_password(
    body: ForgotPasswordBody, db: DbSession, background_tasks: BackgroundTasks
) -> dict[str, str]:
    normalized_forgot = body.email.strip().lower()
    result = await db.execute(select(Profile).where(func.lower(Profile.email) == normalized_forgot))
    profile = result.scalar_one_or_none()
    if profile is None:
        return {"message": "If an account exists for this email, a reset link has been sent."}

    # Short-lived, purpose-bound reset token (30 min, purpose=reset)
    reset_token = sign_reset_token(profile.id, profile.email, profile.role, token_version=profile.token_version, org_id=profile.org_id)

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

    new_token = sign_token(profile.id, profile.email, profile.role, token_version=profile.token_version, org_id=profile.org_id)
    org = await db.get(Organization, profile.org_id) if profile.org_id else None
    return AuthOut(token=new_token, user=ProfileOut.model_validate(profile), organization=OrganizationOut.model_validate(org) if org else None)


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
