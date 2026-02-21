"""add url to processed.documents

Revision ID: 9e3f7a2b1c4d
Revises: 2b7e4c9d5a61
Create Date: 2026-02-20 00:14:00.000000

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "9e3f7a2b1c4d"
down_revision = "2b7e4c9d5a61"
branch_labels = None
depends_on = None

PROCESSED_SCHEMA = "processed"
TABLE_NAME = "documents"
COLUMN_NAME = "url"
ARXIV_ABS_PREFIX = "https://arxiv.org/abs/"


def upgrade() -> None:
    op.add_column(
        TABLE_NAME,
        sa.Column(COLUMN_NAME, sa.Text(), nullable=True),
        schema=PROCESSED_SCHEMA,
    )

    op.execute(
        sa.text(
            f"""
            UPDATE {PROCESSED_SCHEMA}.{TABLE_NAME}
            SET {COLUMN_NAME} = :url_prefix || id
            WHERE id IS NOT NULL
            """
        ).bindparams(url_prefix=ARXIV_ABS_PREFIX)
    )

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
    op.drop_column(TABLE_NAME, COLUMN_NAME, schema=PROCESSED_SCHEMA)
