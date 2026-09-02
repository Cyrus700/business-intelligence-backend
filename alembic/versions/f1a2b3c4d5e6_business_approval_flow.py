"""business approval flow: org status + email verification

Revision ID: f1a2b3c4d5e6
Revises: d0def9feeedd
Create Date: 2026-09-02
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "f1a2b3c4d5e6"
down_revision = ("d0def9feeedd", "29aa60c95764")
branch_labels = None
depends_on = None


def upgrade():
    # Organizations: approval workflow
    op.add_column("organizations", sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"))
    op.add_column("organizations", sa.Column("approved_at", sa.DateTime(), nullable=True))
    op.add_column("organizations", sa.Column("approved_by", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("organizations", sa.Column("rejected_at", sa.DateTime(), nullable=True))
    op.add_column("organizations", sa.Column("rejected_by", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("organizations", sa.Column("rejection_reason", sa.Text(), nullable=True))
    op.create_foreign_key("fk_organizations_approved_by", "organizations", "profiles", ["approved_by"], ["id"], ondelete="SET NULL")
    op.create_foreign_key("fk_organizations_rejected_by", "organizations", "profiles", ["rejected_by"], ["id"], ondelete="SET NULL")
    op.create_index("ix_organizations_status", "organizations", ["status"])

    # Existing orgs (legacy and already created) are considered approved
    op.execute("UPDATE organizations SET status='approved', approved_at=now() WHERE status='pending'")

    # Profiles: email verification
    op.add_column("profiles", sa.Column("email_verified", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    op.add_column("profiles", sa.Column("email_verification_token", sa.String(length=128), nullable=True))
    op.add_column("profiles", sa.Column("email_verification_expires_at", sa.DateTime(), nullable=True))
    op.add_column("profiles", sa.Column("email_verified_at", sa.DateTime(), nullable=True))
    op.create_index("ix_profiles_email_verification_token", "profiles", ["email_verification_token"], unique=True)

    # Backfill: existing active profiles are considered email verified
    op.execute("UPDATE profiles SET email_verified=true, email_verified_at=now() WHERE is_active=true")

    # Drop server_default after backfill (keep column default false for new rows)
    op.alter_column("organizations", "status", server_default=None)
    op.alter_column("profiles", "email_verified", server_default=None)


def downgrade():
    op.drop_index("ix_profiles_email_verification_token", table_name="profiles")
    op.drop_column("profiles", "email_verified_at")
    op.drop_column("profiles", "email_verification_expires_at")
    op.drop_column("profiles", "email_verification_token")
    op.drop_column("profiles", "email_verified")
    op.drop_index("ix_organizations_status", table_name="organizations")
    op.drop_constraint("fk_organizations_rejected_by", "organizations", type_="foreignkey")
    op.drop_constraint("fk_organizations_approved_by", "organizations", type_="foreignkey")
    op.drop_column("organizations", "rejection_reason")
    op.drop_column("organizations", "rejected_by")
    op.drop_column("organizations", "rejected_at")
    op.drop_column("organizations", "approved_by")
    op.drop_column("organizations", "approved_at")
    op.drop_column("organizations", "status")
