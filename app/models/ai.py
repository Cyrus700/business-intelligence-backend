import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, uuid_pk


class Conversation(Base, TimestampMixin):
    __tablename__ = "ai_conversations"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), index=True
    )
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    title: Mapped[str] = mapped_column(default="New conversation")
    # token-budget-aware rolling summary of the conversation's oldest turns
    # (services/ai/memory.py); None when the whole history still fits the budget.
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint("LENGTH(title) > 0", name="ck_conversation_title_not_empty"),
        Index("ix_conversations_org_id", "org_id"),
    )


class Message(Base):
    __tablename__ = "ai_messages"

    id: Mapped[uuid.UUID] = uuid_pk()
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ai_conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str]
    content: Mapped[str] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="ck_message_valid_role"),
        CheckConstraint("LENGTH(content) > 0", name="ck_message_content_not_empty"),
        Index("ix_messages_org_id", "org_id"),
        # Index for loading conversation messages in order
    )


class AiRetentionSetting(Base):
    """Global AI history retention — auto-flush TTL controlled by system admin."""

    __tablename__ = "ai_retention_settings"

    id: Mapped[uuid.UUID] = uuid_pk()
    retention_days: Mapped[int] = mapped_column(default=30)
    # Nullable — global setting, optionally scoped to org in future
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint("retention_days >= 0 AND retention_days <= 365", name="ck_retention_valid_days"),
        # 0 = never auto-flush (keep forever); 7=week 30=month etc.
    )
