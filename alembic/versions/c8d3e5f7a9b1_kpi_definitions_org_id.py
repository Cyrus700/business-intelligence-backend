"""kpi_definitions.org_id — column the model expects but the multitenant
isolation migration (29aa60c95764) never added, because kpi_definitions was
missing from its tenant_tables list. Caused every read of KpiDefinition
(GET /kpis/summary, /insights?scope=dashboard) to 500 with
"column kpi_definitions.org_id does not exist".

NULL org_id keeps its existing meaning here: a global default definition
visible to every org (see app/services/analytics/queries.kpi_summary), so
this migration does not backfill or set NOT NULL.

Revision ID: c8d3e5f7a9b1
Revises: a3f5c81d6e07, b4c8d9e0f1a2
Create Date: 2026-09-07
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "c8d3e5f7a9b1"
down_revision = ("a3f5c81d6e07", "b4c8d9e0f1a2")
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    cols = [c["name"] for c in insp.get_columns("kpi_definitions")]
    if "org_id" not in cols:
        op.add_column("kpi_definitions", sa.Column("org_id", UUID(as_uuid=True), nullable=True))
        op.create_foreign_key(
            "fk_kpi_definitions_org_id_organizations",
            "kpi_definitions",
            "organizations",
            ["org_id"],
            ["id"],
            ondelete="CASCADE",
        )
        op.create_index("ix_kpi_definitions_org_id", "kpi_definitions", ["org_id"])


def downgrade() -> None:
    op.drop_index("ix_kpi_definitions_org_id", table_name="kpi_definitions")
    op.drop_constraint("fk_kpi_definitions_org_id_organizations", "kpi_definitions", type_="foreignkey")
    op.drop_column("kpi_definitions", "org_id")
