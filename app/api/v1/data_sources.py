from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from app.api.deps import DbSession, require_role
from app.models import DataSource
from app.schemas.integration import DataSourceIn, DataSourceOut, DataSourceUpdate

router = APIRouter(
    prefix="/data-sources",
    tags=["data-integration"],
    dependencies=[Depends(require_role("admin"))],
)


@router.get("", response_model=list[DataSourceOut])
async def list_sources(db: DbSession) -> list[DataSourceOut]:
    rows = (await db.execute(select(DataSource).order_by(DataSource.created_at))).scalars().all()
    return [DataSourceOut.model_validate(r) for r in rows]


@router.post("", response_model=DataSourceOut, status_code=status.HTTP_201_CREATED)
async def create_source(body: DataSourceIn, db: DbSession) -> DataSourceOut:
    existing = await db.execute(select(DataSource).where(DataSource.name == body.name))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "A source with this name already exists")
    source = DataSource(**body.model_dump())
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return DataSourceOut.model_validate(source)


@router.patch("/{source_id}", response_model=DataSourceOut)
async def update_source(source_id: UUID, body: DataSourceUpdate, db: DbSession) -> DataSourceOut:
    source = await db.get(DataSource, source_id)
    if source is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Data source not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(source, field, value)
    await db.commit()
    await db.refresh(source)
    return DataSourceOut.model_validate(source)
