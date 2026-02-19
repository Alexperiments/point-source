"""create raw cleaned processed schemas and arxiv snapshot table

Revision ID: b44b067c666d
Revises: 7b1f2f9c4e2a
Create Date: 2026-02-19 22:25:23.177420

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "b44b067c666d"
down_revision = "7b1f2f9c4e2a"
branch_labels = None
depends_on = None

RAW_SCHEMA = "raw"
CLEANED_SCHEMA = "cleaned"
PROCESSED_SCHEMA = "processed"
RAW_TABLE_NAME = "arxiv-metadata-oai-snapshot-2026-02-19"


def upgrade() -> None:
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}"))
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {CLEANED_SCHEMA}"))
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {PROCESSED_SCHEMA}"))

    op.execute(
        sa.text(
            f"""
            CREATE TABLE IF NOT EXISTS {RAW_SCHEMA}."{RAW_TABLE_NAME}" (
                raw_json JSONB NOT NULL,
                ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
    )


def downgrade() -> None:
    op.execute(sa.text(f'DROP TABLE IF EXISTS {RAW_SCHEMA}."{RAW_TABLE_NAME}"'))
    op.execute(sa.text(f"DROP SCHEMA IF EXISTS {PROCESSED_SCHEMA}"))
    op.execute(sa.text(f"DROP SCHEMA IF EXISTS {CLEANED_SCHEMA}"))
    op.execute(sa.text(f"DROP SCHEMA IF EXISTS {RAW_SCHEMA}"))
