from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import AuthError, verify_token
from app.models import Profile

bearer_scheme = HTTPBearer(auto_error=False)

DbSession = Annotated[AsyncSession, Depends(get_db)]

# role hierarchy: an endpoint requiring "analyst" admits everyone authenticated
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
    # DB is authoritative for role (JWT app_metadata may lag a role change)
    request.state.user = profile
    return profile


CurrentUser = Annotated[Profile, Depends(get_current_user)]


def require_role(minimum: str) -> Callable[..., Profile]:
    async def checker(user: CurrentUser) -> Profile:
        if ROLE_RANK.get(user.role, 0) < ROLE_RANK[minimum]:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Requires role '{minimum}' or higher",
            )
        return user

    return checker  # type: ignore[return-value]
