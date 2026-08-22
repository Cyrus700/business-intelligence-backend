from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession, require_role
from app.models import DataSource
from app.schemas.integration import DataSourceIn, DataSourceOut, DataSourceUpdate

router = APIRouter(
    prefix="/data-sources",
    tags=["data-integration"],
    dependencies=[Depends(require_role("admin"))],
)


@router.get("", response_model=list[DataSourceOut])
async def list_sources(db: DbSession, user: CurrentUser) -> list[DataSourceOut]:
    from app.api.deps import org_predicate, user_org_id

    rows = (
        await db.execute(
            select(DataSource)
            .where(org_predicate(DataSource.org_id, user_org_id(user)))
            .order_by(DataSource.created_at)
        )
    ).scalars().all()
    return [DataSourceOut.model_validate(r) for r in rows]


@router.post("", response_model=DataSourceOut, status_code=status.HTTP_201_CREATED)
async def create_source(body: DataSourceIn, db: DbSession, user: CurrentUser) -> DataSourceOut:
    from app.api.deps import user_org_id
    from app.services.etl.ssrf import validate_public_http_url

    if body.kind == "rest_api":
        try:
            validate_public_http_url(str(body.config.get("url") or ""))
        except ValueError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    existing = await db.execute(select(DataSource).where(DataSource.name == body.name))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "A source with this name already exists")
    source = DataSource(**body.model_dump(), org_id=user_org_id(user))
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return DataSourceOut.model_validate(source)


@router.patch("/{source_id}", response_model=DataSourceOut)
async def update_source(
    source_id: UUID, body: DataSourceUpdate, db: DbSession, user: CurrentUser
) -> DataSourceOut:
    from app.api.deps import org_predicate, user_org_id
    from app.services.etl.ssrf import validate_public_http_url

    payload = body.model_dump(exclude_unset=True)
    config_url = str((payload.get("config") or {}).get("url") or "")
    if config_url:
        try:
            validate_public_http_url(config_url)
        except ValueError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e

    source = (
        await db.execute(
            select(DataSource).where(
                DataSource.id == source_id, org_predicate(DataSource.org_id, user_org_id(user))
            )
        )
    ).scalar_one_or_none()
    if source is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Data source not found")
    for field, value in payload.items():
        setattr(source, field, value)
    await db.commit()
    await db.refresh(source)
    return DataSourceOut.model_validate(source)
