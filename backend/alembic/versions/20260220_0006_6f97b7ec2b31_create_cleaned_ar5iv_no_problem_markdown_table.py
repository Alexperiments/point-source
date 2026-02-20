"""create cleaned ar5iv no problem markdown table

Revision ID: 6f97b7ec2b31
Revises: 4e0f71f2c9ad
Create Date: 2026-02-20 00:06:00.000000

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "6f97b7ec2b31"
down_revision = "4e0f71f2c9ad"
branch_labels = None
depends_on = None

RAW_SCHEMA = "raw"
CLEANED_SCHEMA = "cleaned"
RAW_TABLE_NAME = "ar5iv_no_problem_markdown"
CLEANED_TABLE_NAME = "ar5iv_no_problem_markdown"


def upgrade() -> None:
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {CLEANED_SCHEMA}"))

    op.create_table(
        CLEANED_TABLE_NAME,
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        schema=CLEANED_SCHEMA,
    )

    op.execute(
        sa.text(
            f"""
            WITH source AS (
                SELECT
                    regexp_replace(
                        regexp_replace(id, '^.*/', ''),
                        '\\.html$',
                        ''
                    ) AS extracted_id,
                    text
                FROM {RAW_SCHEMA}.{RAW_TABLE_NAME}
                WHERE NULLIF(trim(id), '') IS NOT NULL
            ),
            deduped AS (
                SELECT DISTINCT ON (extracted_id)
                    extracted_id AS id,
                    text
                FROM source
                WHERE NULLIF(trim(extracted_id), '') IS NOT NULL
                ORDER BY extracted_id
            )
            INSERT INTO {CLEANED_SCHEMA}.{CLEANED_TABLE_NAME} (id, text)
            SELECT id, text
            FROM deduped
            ON CONFLICT (id) DO NOTHING
            """
        )
    )


def downgrade() -> None:
    op.drop_table(CLEANED_TABLE_NAME, schema=CLEANED_SCHEMA)
