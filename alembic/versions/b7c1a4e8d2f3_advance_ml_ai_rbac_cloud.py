"""advance ml ai rbac cloud

Schema for the production-grade advance:
- multi-tenant: organizations + org_id columns (profiles, data_sources, etl_jobs,
  raw_uploads), token_version on profiles for instant JWT revocation.
- ML: anomalies.correlation_id (correlated events), ModelDrift (auditable live
  accuracy checkpoints), AnomalyFeedback (detector calibration signal).
- Decision/cloud: RecommendationFeedback (ranking signal), BackgroundJob
  (durable in-process job queue — the SQS-less dispatch path).

RLS for the new tables mirrors d0def9feeedd (defence-in-depth via the JWT
app_metadata role claim; the service-role FastAPI connection bypasses RLS).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "b7c1a4e8d2f3"
down_revision: str | None = "a7c3f9e2b1d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ROLE = "(auth.jwt() -> 'app_metadata' ->> 'role')"
ANY_ROLE = f"{ROLE} IN ('admin', 'manager', 'analyst')"
ADMIN = f"{ROLE} = 'admin'"
# org-scoped users see only their own org's data; unset org_id = default org
ORG_ORG_JWT = (
    "(auth.jwt() -> 'app_metadata' ->> 'org_id') IS NULL OR org_id::text = (auth.jwt() -> 'app_metadata' ->> 'org_id')"
)


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_organizations")),
        sa.UniqueConstraint("name", name=op.f("uq_organizations_name")),
    )

    op.add_column("profiles", sa.Column("org_id", sa.UUID(), nullable=True))
    op.create_foreign_key(op.f("fk_profiles_org_id_organizations"), "profiles", "organizations", ["org_id"], ["id"])
    op.create_index(op.f("ix_profiles_org_id"), "profiles", ["org_id"], unique=False)
    op.add_column("profiles", sa.Column("token_version", sa.Integer(), server_default="0", nullable=False))

    op.add_column("data_sources", sa.Column("org_id", sa.UUID(), nullable=True))
    op.create_foreign_key(
        op.f("fk_data_sources_org_id_organizations"),
        "data_sources",
        "organizations",
        ["org_id"],
        ["id"],
    )
    op.add_column("etl_jobs", sa.Column("org_id", sa.UUID(), nullable=True))
    op.create_foreign_key(op.f("fk_etl_jobs_org_id_organizations"), "etl_jobs", "organizations", ["org_id"], ["id"])
    op.add_column("raw_uploads", sa.Column("org_id", sa.UUID(), nullable=True), schema="staging")
    op.create_foreign_key(
        op.f("fk_raw_uploads_org_id_organizations"),
        "raw_uploads",
        "organizations",
        ["org_id"],
        ["id"],
        source_schema="staging",
    )

    op.add_column("ai_conversations", sa.Column("summary", sa.Text(), nullable=True))

    op.add_column("anomalies", sa.Column("correlation_id", sa.UUID(), nullable=True))
    op.create_index(op.f("ix_anomalies_correlation_id"), "anomalies", ["correlation_id"], unique=False)

    op.create_table(
        "anomaly_feedback",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("anomaly_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("disposition", sa.String(), nullable=False),
        sa.Column("note", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "disposition IN ('false_positive', 'confirmed')",
            name=op.f("ck_anomaly_feedback_valid_disposition"),
        ),
        sa.ForeignKeyConstraint(
            ["anomaly_id"], ["anomalies.id"], name=op.f("fk_anomaly_feedback_anomaly_id_anomalies"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["profiles.id"], name=op.f("fk_anomaly_feedback_user_id_profiles"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_anomaly_feedback")),
    )
    op.create_index(op.f("ix_anomaly_feedback_anomaly_id"), "anomaly_feedback", ["anomaly_id"], unique=False)
    op.create_index(
        op.f("ix_anomaly_feedback_anomaly_created"),
        "anomaly_feedback",
        ["anomaly_id", "created_at"],
        unique=False,
    )

    op.create_table(
        "model_drift",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("model_id", sa.UUID(), nullable=False),
        sa.Column("target", sa.String(), nullable=False),
        sa.Column("measured_on", sa.Date(), nullable=False),
        sa.Column("window_days", sa.Integer(), nullable=False),
        sa.Column("live_mape", sa.Numeric(18, 4), nullable=True),
        sa.Column("holdout_mape", sa.Numeric(18, 4), nullable=True),
        sa.Column("threshold_mape", sa.Numeric(18, 4), nullable=True),
        sa.Column("triggered", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_id"], ["ml_models.id"], name=op.f("fk_model_drift_model_id_ml_models"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_drift")),
    )
    op.create_index(op.f("ix_model_drift_model_id"), "model_drift", ["model_id"], unique=False)
    op.create_index(op.f("ix_model_drift_measured_on"), "model_drift", ["measured_on"], unique=False)

    op.create_table(
        "recommendation_feedback",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("rec_key", sa.String(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("action IN ('accepted', 'dismissed')", name=op.f("ck_recommendation_feedback_valid_action")),
        sa.CheckConstraint("LENGTH(rec_key) > 0", name=op.f("ck_recommendation_feedback_ck_rec_key_not_empty")),
        sa.ForeignKeyConstraint(
            ["user_id"], ["profiles.id"], name=op.f("fk_recommendation_feedback_user_id_profiles"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_recommendation_feedback")),
    )
    op.create_index(
        op.f("ix_rec_feedback_key_created"), "recommendation_feedback", ["rec_key", "created_at"], unique=False
    )
    op.create_index(op.f("ix_recommendation_feedback_rec_key"), "recommendation_feedback", ["rec_key"], unique=False)

    op.create_table(
        "background_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("run_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'succeeded', 'failed')",
            name=op.f("ck_background_jobs_valid_status"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_background_jobs")),
    )
    op.create_index(op.f("ix_background_jobs_pending"), "background_jobs", ["status", "run_at"], unique=False)

    # ---- RLS for the new tables (defence-in-depth, same mechanism as Phase 6) ----
    op.execute("ALTER TABLE organizations ENABLE ROW LEVEL SECURITY")
    op.execute("CREATE POLICY org_read ON organizations FOR SELECT TO PUBLIC USING (true)")
    op.execute("CREATE POLICY org_admin_write ON organizations FOR ALL USING (true) WITH CHECK (true)")
    op.execute("DROP POLICY IF EXISTS profiles_read_own_or_admin ON profiles")
    op.execute(
        "CREATE POLICY profiles_read_own_or_admin ON profiles FOR SELECT TO PUBLIC "
        f"USING ((id = auth.uid()) OR {ADMIN} OR {ORG_ORG_JWT})"
    )
    op.execute("ALTER TABLE anomaly_feedback ENABLE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY feedback_read_any_role ON anomaly_feedback FOR SELECT TO PUBLIC USING ({ANY_ROLE})")
    op.execute(
        f"CREATE POLICY feedback_insert_any_role ON anomaly_feedback FOR INSERT TO PUBLIC WITH CHECK ({ANY_ROLE})"
    )
    op.execute("ALTER TABLE model_drift ENABLE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY drift_read_any_role ON model_drift FOR SELECT TO PUBLIC USING ({ANY_ROLE})")
    op.execute("ALTER TABLE recommendation_feedback ENABLE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY recfb_read_own ON recommendation_feedback FOR SELECT TO PUBLIC "
        f"USING ((user_id = auth.uid()) OR {ADMIN})"
    )
    op.execute(f"CREATE POLICY recfb_insert ON recommendation_feedback FOR INSERT TO PUBLIC WITH CHECK ({ANY_ROLE})")
    op.execute("ALTER TABLE background_jobs ENABLE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY jobs_admin ON background_jobs FOR ALL TO PUBLIC USING ({ADMIN})")


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS profiles_read_own_or_admin ON profiles")
    op.execute("DROP POLICY IF EXISTS feedback_insert_any_role ON anomaly_feedback")
    op.execute("DROP POLICY IF EXISTS feedback_read_any_role ON anomaly_feedback")
    op.execute("DROP POLICY IF EXISTS drift_read_any_role ON model_drift")
    op.execute("DROP POLICY IF EXISTS recfb_insert ON recommendation_feedback")
    op.execute("DROP POLICY IF EXISTS recfb_read_own ON recommendation_feedback")
    op.execute("DROP POLICY IF EXISTS jobs_admin ON background_jobs")
    op.execute("ALTER TABLE organizations DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE anomaly_feedback DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE model_drift DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE recommendation_feedback DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE background_jobs DISABLE ROW LEVEL SECURITY")

    op.drop_table("background_jobs")
    op.drop_table("recommendation_feedback")
    op.drop_table("model_drift")
    op.drop_table("anomaly_feedback")
    op.drop_index(op.f("ix_anomalies_correlation_id"), table_name="anomalies")
    op.drop_column("anomalies", "correlation_id")
    op.drop_column("ai_conversations", "summary")
    op.drop_column("raw_uploads", "org_id")
    op.drop_column("etl_jobs", "org_id")
    op.drop_column("data_sources", "org_id")
    op.drop_index(op.f("ix_profiles_org_id"), table_name="profiles")
    op.drop_column("profiles", "token_version")
    op.drop_column("profiles", "org_id")
    op.drop_table("organizations")
