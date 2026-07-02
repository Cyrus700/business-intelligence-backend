from fastapi import APIRouter

from app.api.deps import CurrentUser
from app.schemas.identity import ProfileOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me", response_model=ProfileOut)
async def me(user: CurrentUser) -> ProfileOut:
    return ProfileOut.model_validate(user)
