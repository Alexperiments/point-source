"""add primary key to processed.documents

Revision ID: 8e1b9c74a6d2
Revises: 1f6a2bd3c4e5
Create Date: 2026-02-20 00:10:00.000000

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "8e1b9c74a6d2"
down_revision = "1f6a2bd3c4e5"
branch_labels = None
depends_on = None

PROCESSED_SCHEMA = "processed"
TABLE_NAME = "documents"
PK_CONSTRAINT_NAME = "processed_documents_pkey"


def upgrade() -> None:
    op.execute(
        sa.text(
            f"""
            DO $$
            DECLARE
                null_count BIGINT;
                duplicate_count BIGINT;
            BEGIN
                SELECT COUNT(*) INTO null_count
                FROM {PROCESSED_SCHEMA}.{TABLE_NAME}
                WHERE id IS NULL;

                IF null_count > 0 THEN
                    RAISE EXCEPTION
                        '{PROCESSED_SCHEMA}.{TABLE_NAME}.id contains % NULL values; cannot add primary key',
                        null_count;
                END IF;

                SELECT COUNT(*) INTO duplicate_count
                FROM (
                    SELECT id
                    FROM {PROCESSED_SCHEMA}.{TABLE_NAME}
                    GROUP BY id
                    HAVING COUNT(*) > 1
                ) AS duplicate_ids;

                IF duplicate_count > 0 THEN
                    RAISE EXCEPTION
                        '{PROCESSED_SCHEMA}.{TABLE_NAME}.id contains % duplicate values; cannot add primary key',
                        duplicate_count;
                END IF;
            END
            $$;
            """
        )
    )

    op.create_primary_key(
        PK_CONSTRAINT_NAME,
        TABLE_NAME,
        ["id"],
        schema=PROCESSED_SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint(
        PK_CONSTRAINT_NAME,
        TABLE_NAME,
        schema=PROCESSED_SCHEMA,
        type_="primary",
    )
