"""forecast segments (dimensions column on ml_models/forecasts)

Revision ID: f3a1c9b7d2e4
Revises: c5e81f0a9b62
Create Date: 2026-08-09 18:30:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = 'f3a1c9b7d2e4'
down_revision: str | None = 'c5e81f0a9b62'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'ml_models',
        sa.Column(
            'dimensions',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column(
        'forecasts',
        sa.Column(
            'dimensions',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.drop_constraint('uq_model_version', 'ml_models', type_='unique')
    op.create_unique_constraint(
        'uq_model_version', 'ml_models', ['model_type', 'target', 'dimensions', 'version']
    )
    op.drop_constraint('uq_forecast_point', 'forecasts', type_='unique')
    op.create_unique_constraint(
        'uq_forecast_point', 'forecasts', ['model_id', 'target', 'dimensions', 'forecast_date']
    )


def downgrade() -> None:
    op.drop_constraint('uq_forecast_point', 'forecasts', type_='unique')
    op.create_unique_constraint(
        'uq_forecast_point', 'forecasts', ['model_id', 'target', 'forecast_date']
    )
    op.drop_constraint('uq_model_version', 'ml_models', type_='unique')
    op.create_unique_constraint(
        'uq_model_version', 'ml_models', ['model_type', 'target', 'version']
    )
    op.drop_column('forecasts', 'dimensions')
    op.drop_column('ml_models', 'dimensions')
