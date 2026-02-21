"""make processed.document_chunks.document_id not null

Revision ID: 2b7e4c9d5a61
Revises: 6d2ac9e4bf31
Create Date: 2026-02-20 00:13:00.000000

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "2b7e4c9d5a61"
down_revision = "6d2ac9e4bf31"
branch_labels = None
depends_on = None

PROCESSED_SCHEMA = "processed"
TABLE_NAME = "document_chunks"
COLUMN_NAME = "document_id"


def upgrade() -> None:
    op.execute(
        sa.text(
            f"""
            DO $$
            DECLARE
                null_count BIGINT;
            BEGIN
                SELECT COUNT(*) INTO null_count
                FROM {PROCESSED_SCHEMA}.{TABLE_NAME}
                WHERE {COLUMN_NAME} IS NULL;

                IF null_count > 0 THEN
                    RAISE EXCEPTION
                        '{PROCESSED_SCHEMA}.{TABLE_NAME}.{COLUMN_NAME} contains % NULL values; cannot enforce NOT NULL',
                        null_count;
                END IF;
            END
            $$;
            """
        )
    )

    op.alter_column(
        TABLE_NAME,
        COLUMN_NAME,
        existing_type=sa.Text(),
        nullable=False,
        schema=PROCESSED_SCHEMA,
    )


def downgrade() -> None:
    op.alter_column(
        TABLE_NAME,
        COLUMN_NAME,
        existing_type=sa.Text(),
        nullable=True,
        schema=PROCESSED_SCHEMA,
    )
