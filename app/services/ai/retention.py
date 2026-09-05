"""AI history retention — auto-flush with admin-configurable TTL.

- Global retention_days (0 = keep forever, 7 = week, 30 = month, etc.)
- Daily job deletes conversations whose updated_at < now - retention_days.
- Lazily seeds default row (30 days) if table empty so fresh installs work
  before the migration seed or on SQLite tests.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ai import AiRetentionSetting, Conversation

logger = logging.getLogger(__name__)

DEFAULT_DAYS = 30
MIN_DAYS = 0
MAX_DAYS = 365
ALLOWED_CHOICES = [0, 7, 14, 30, 60, 90, 180, 365]


async def get_retention_days(db: AsyncSession) -> int:
    row = (await db.execute(select(AiRetentionSetting).limit(1))).scalars().first()
    if row is None:
        # lazy seed for tests / fresh DB before migration
        row = AiRetentionSetting(retention_days=DEFAULT_DAYS)
        db.add(row)
        await db.flush()
        logger.info("seeded ai_retention_settings with %d days", DEFAULT_DAYS)
        return DEFAULT_DAYS
    return int(row.retention_days)


async def set_retention_days(db: AsyncSession, days: int, updated_by: uuid.UUID | None = None) -> AiRetentionSetting:
    if not isinstance(days, int) or days < MIN_DAYS or days > MAX_DAYS:
        raise ValueError(f"retention_days must be between {MIN_DAYS} and {MAX_DAYS}, got {days}")
    row = (await db.execute(select(AiRetentionSetting).limit(1))).scalars().first()
    if row is None:
        row = AiRetentionSetting(retention_days=days, updated_by=updated_by)
        db.add(row)
    else:
        row.retention_days = days
        row.updated_by = updated_by
        # updated_at auto via onupdate, but ensure for SQLite
        row.updated_at = datetime.now(UTC).replace(tzinfo=None)
    await db.flush()
    logger.info("ai retention set to %d days by %s", days, updated_by)
    return row


async def flush_expired_conversations(db: AsyncSession) -> int:
    """Delete conversations older than retention. Returns count deleted.

    Uses Conversation.updated_at (last activity) — a weekly chat that is still
    active stays, an idle thread expires. 0 = disabled (keep forever).
    """
    days = await get_retention_days(db)
    if days == 0:
        logger.debug("ai retention disabled (0 days) — skip flush")
        return 0
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
    # Find expired ids to log
    expired = (await db.execute(select(Conversation.id).where(Conversation.updated_at < cutoff))).scalars().all()
    if not expired:
        return 0
    # Delete via ORM so cascade deletes messages
    for cid in expired:
        obj = await db.get(Conversation, cid)
        if obj:
            await db.delete(obj)
    await db.flush()
    logger.info(
        "auto-flushed %d ai conversations older than %d days (cutoff %s)", len(expired), days, cutoff.date().isoformat()
    )
    return len(expired)


async def retention_status(db: AsyncSession) -> dict:
    days = await get_retention_days(db)
    row = (await db.execute(select(AiRetentionSetting).limit(1))).scalars().first()
    return {
        "retention_days": days,
        "updated_at": row.updated_at.isoformat() if row and row.updated_at else None,
        "updated_by": str(row.updated_by) if row and row.updated_by else None,
        "choices": ALLOWED_CHOICES,
        "is_disabled": days == 0,
    }
