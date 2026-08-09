"""dynamic rbac catalog

Moves the role/permission policy out of code and into three admin-editable
tables (``roles``, ``permissions``, ``role_permissions``), seeded with exactly
the matrix that used to be hard-coded in ``app/api/deps.py`` and the frontend's
``lib/permissions.ts`` — so the upgrade is behaviour-preserving.

Also drops the ``valid_role`` CHECK on ``profiles``: admins can now define
custom roles, and validity is enforced by the API against the ``roles`` table.

Revision ID: c5e81f0a9b62
Revises: b7c1a4e8d2f3
Create Date: 2026-08-09
"""

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op
from app.core.rbac_defaults import DEFAULT_GRANTS, DEFAULT_PERMISSIONS, DEFAULT_ROLES

revision: str = "c5e81f0a9b62"
down_revision: str | None = "b7c1a4e8d2f3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ROLE_CLAIM = "(auth.jwt() -> 'app_metadata' ->> 'role')"
ANY_ROLE = f"{ROLE_CLAIM} IN ('admin', 'manager', 'analyst')"
ADMIN = f"{ROLE_CLAIM} = 'admin'"

RLS_POLICIES: dict[str, list[tuple[str, str, str, str | None]]] = {
    # every authenticated role may read the catalog (the UI renders it);
    # only admins may edit it
    "roles": [
        ("rbac_read_any_role", "SELECT", ANY_ROLE, None),
        ("rbac_admin_write", "ALL", ADMIN, ADMIN),
    ],
    "permissions": [
        ("rbac_read_any_role", "SELECT", ANY_ROLE, None),
        ("rbac_admin_write", "ALL", ADMIN, ADMIN),
    ],
    "role_permissions": [
        ("rbac_read_any_role", "SELECT", ANY_ROLE, None),
        ("rbac_admin_write", "ALL", ADMIN, ADMIN),
    ],
}


def upgrade() -> None:
    roles = op.create_table(
        "roles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("color", sa.String(), nullable=False, server_default="slate"),
        sa.Column("is_system", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("name", name="uq_roles_name"),
        sa.UniqueConstraint("rank", name="uq_roles_rank"),
        sa.CheckConstraint("rank > 0", name="ck_roles_positive_rank"),
        sa.CheckConstraint("name ~ '^[a-z][a-z0-9_-]{1,31}$'", name="ck_roles_slug_name"),
    )

    permissions = op.create_table(
        "permissions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("group_label", sa.String(), nullable=False, server_default="General"),
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_system", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("key", name="uq_permissions_key"),
        sa.CheckConstraint(
            "key ~ '^[a-z][a-z0-9-]*:[a-z][a-z0-9-]*$'", name="ck_permissions_key_format"
        ),
    )
    op.create_index("ix_permissions_group_label", "permissions", ["group_label"])

    role_permissions = op.create_table(
        "role_permissions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("role_id", UUID(as_uuid=True), nullable=False),
        sa.Column("permission_id", UUID(as_uuid=True), nullable=False),
        sa.Column("granted_by", UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["role_id"],
            ["roles.id"],
            name="fk_role_permissions_role_id_roles",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["permission_id"],
            ["permissions.id"],
            name="fk_role_permissions_permission_id_permissions",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["granted_by"],
            ["profiles.id"],
            name="fk_role_permissions_granted_by_profiles",
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint("role_id", "permission_id", name="uq_role_permissions_role_id"),
    )
    op.create_index("ix_role_permissions_permission_id", "role_permissions", ["permission_id"])

    # ── seed the shipped policy ───────────────────────────────────────────
    role_ids = {r["name"]: uuid.uuid4() for r in DEFAULT_ROLES}
    perm_ids = {p["key"]: uuid.uuid4() for p in DEFAULT_PERMISSIONS}

    op.bulk_insert(
        roles,
        [
            {
                "id": role_ids[r["name"]],
                "name": r["name"],
                "label": r["label"],
                "description": r["description"],
                "rank": r["rank"],
                "color": r["color"],
                "is_system": True,
                "is_active": True,
            }
            for r in DEFAULT_ROLES
        ],
    )
    op.bulk_insert(
        permissions,
        [
            {
                "id": perm_ids[p["key"]],
                "key": p["key"],
                "label": p["label"],
                "description": p["description"],
                "group_label": p["group_label"],
                "sort_order": p["sort_order"],
                "is_system": True,
            }
            for p in DEFAULT_PERMISSIONS
        ],
    )
    op.bulk_insert(
        role_permissions,
        [
            {
                "id": uuid.uuid4(),
                "role_id": role_ids[role],
                "permission_id": perm_ids[key],
                "granted_by": None,
            }
            for role, keys in DEFAULT_GRANTS.items()
            for key in keys
        ],
    )

    # custom roles are now legal values for profiles.role
    op.execute("ALTER TABLE profiles DROP CONSTRAINT IF EXISTS ck_profiles_valid_role")
    op.execute("ALTER TABLE profiles DROP CONSTRAINT IF EXISTS valid_role")

    for table, policies in RLS_POLICIES.items():
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        for name, command, using, check in policies:
            with_check = f" WITH CHECK ({check})" if check else ""
            op.execute(
                f"CREATE POLICY {name} ON {table} FOR {command} "
                f"TO PUBLIC USING ({using}){with_check}"
            )


def downgrade() -> None:
    for table, policies in RLS_POLICIES.items():
        for name, _, _, _ in policies:
            op.execute(f"DROP POLICY IF EXISTS {name} ON {table}")
    op.drop_index("ix_role_permissions_permission_id", table_name="role_permissions")
    op.drop_table("role_permissions")
    op.drop_index("ix_permissions_group_label", table_name="permissions")
    op.drop_table("permissions")
    op.drop_table("roles")
    op.execute(
        "ALTER TABLE profiles ADD CONSTRAINT ck_profiles_valid_role "
        "CHECK (role IN ('admin', 'manager', 'analyst'))"
    )
