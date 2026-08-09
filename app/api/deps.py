from collections.abc import Callable
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.core.database import get_db
from app.core.security import AuthError, verify_token
from app.models import Profile
from app.services import rbac

bearer_scheme = HTTPBearer(auto_error=False)

DbSession = Annotated[AsyncSession, Depends(get_db)]

# Static fallback hierarchy. The live ladder lives in the admin-editable
# ``roles`` table and is read through app.services.rbac; this dict only backs
# code paths that have no database session at hand.
ROLE_RANK = {"analyst": 1, "manager": 2, "admin": 3}


async def get_current_user(
    request: Request,
    db: DbSession,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> Profile:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    try:
        claims = verify_token(credentials.credentials)
    except AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, e.detail) from e

    profile = await db.get(Profile, claims.user_id)
    if profile is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "No profile for this user")
    if not profile.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account is disabled")
    # Session revocation: every JWT carries a "ver" claim minted from
    # profiles.token_version at sign-in. Role downgrades / account flips bump
    # token_version, so pre-downgrade tokens stop working on the very next
    # request — no reliance on JWT expiry. Tokens minted without a ver claim
    # (legacy/3rd-party) are accepted as version 0.
    if (
        claims.token_version is not None
        and profile.token_version != claims.token_version
    ):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Session revoked — your token was invalidated by a permissions change. Sign in again.",
        )
    # DB is authoritative for role (JWT app_metadata may lag a role change)
    request.state.user = profile
    request.state.org_id = profile.org_id
    return profile


CurrentUser = Annotated[Profile, Depends(get_current_user)]


def require_role(minimum: str) -> Callable[..., Profile]:
    """Gate on the role hierarchy, using the admin-editable rank ladder.

    Ranks come from the ``roles`` table (cached), so promoting a custom role
    above "manager" immediately widens what it can reach. If the role catalog
    has not been seeded yet the built-in ``ROLE_RANK`` applies.
    """

    async def checker(user: CurrentUser, db: DbSession) -> Profile:
        policy = await rbac.get_policy(db)
        required = policy.rank(minimum) or ROLE_RANK.get(minimum, 0)
        if policy.rank(user.role) < required:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Requires role '{minimum}' or higher",
            )
        return user

    return checker  # type: ignore[return-value]


def require_permission(*permissions: str) -> Callable[..., Profile]:
    """Gate on a capability rather than a role.

    Preferred over :func:`require_role` for anything an admin should be able to
    re-delegate: the check reads the live grant matrix, so moving
    ``reports:generate`` from manager to analyst needs no code change. All
    listed permissions are required (AND).
    """

    async def checker(user: CurrentUser, db: DbSession) -> Profile:
        policy = await rbac.get_policy(db)
        held = policy.permissions_for(user.role)
        missing = [p for p in permissions if p not in held]
        if missing:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Missing permission: {', '.join(missing)}",
            )
        return user

    return checker  # type: ignore[return-value]


# ── attribute-based access helpers ────────────────────────────────────────

def user_org_id(user: Profile) -> UUID | None:
    """Tenant scope of the caller (None = default org, multi-tenancy off)."""
    return user.org_id


def org_predicate(column: ColumnElement, org_id: UUID | None) -> ColumnElement:
    """Row-level tenant filter: the row's org must match the caller's org.

    Rows with NULL org_id are legacy/default-org rows and remain visible to
    everyone (single-tenant history), which keeps the existing warehouse data
    fully readable after the migration.
    """
    if org_id is None:
        return column.is_(None)  # legacy mode: only unassigned rows exist
    return column.is_(None) | (column == org_id)


SENSITIVE_FIELDS = ("total_amount", "discount", "unit_price", "amount")


def redact_sensitive(user: Profile, record: dict[str, Any]) -> dict[str, Any]:
    """Field-level redaction: analysts see aggregates, never line-level figures.

    Line-level financial values (per-transaction money fields) are the
    sensitive surface the role hierarchy must gate; managers and admins keep
    full detail, analysts get the record with money fields nulled out.
    """
    if user.role in ("manager", "admin"):
        return record
    out = dict(record)
    for field in SENSITIVE_FIELDS:
        if field in out:
            out[field] = None
    out["redacted"] = True
    return out