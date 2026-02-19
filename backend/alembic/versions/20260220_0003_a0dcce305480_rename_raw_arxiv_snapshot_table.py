"""rename raw arxiv snapshot table

Revision ID: a0dcce305480
Revises: b44b067c666d
Create Date: 2026-02-20 00:03:52.530912

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "a0dcce305480"
down_revision = "b44b067c666d"
branch_labels = None
depends_on = None

RAW_SCHEMA = "raw"
OLD_TABLE_NAME = "arxiv-metadata-oai-snapshot-2026-02-19"
NEW_TABLE_NAME = "arxiv_metadata_oai_snapshot_2026_02_19"


def upgrade() -> None:
    op.execute(
        sa.text(
            f"""
            DO $$
            BEGIN
                IF to_regclass('{RAW_SCHEMA}."{OLD_TABLE_NAME}"') IS NOT NULL
                   AND to_regclass('{RAW_SCHEMA}.{NEW_TABLE_NAME}') IS NULL THEN
                    EXECUTE 'ALTER TABLE {RAW_SCHEMA}."{OLD_TABLE_NAME}" RENAME TO {NEW_TABLE_NAME}';
                END IF;
            END $$;
            """
        )
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            f"""
            DO $$
            BEGIN
                IF to_regclass('{RAW_SCHEMA}.{NEW_TABLE_NAME}') IS NOT NULL
                   AND to_regclass('{RAW_SCHEMA}."{OLD_TABLE_NAME}"') IS NULL THEN
                    EXECUTE 'ALTER TABLE {RAW_SCHEMA}.{NEW_TABLE_NAME} RENAME TO "{OLD_TABLE_NAME}"';
                END IF;
            END $$;
            """
        )
    )
