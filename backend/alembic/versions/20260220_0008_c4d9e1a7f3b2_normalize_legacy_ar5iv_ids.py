"""normalize legacy ar5iv ids in cleaned markdown table

Revision ID: c4d9e1a7f3b2
Revises: b1e4c70d5f2a
Create Date: 2026-02-20 00:08:00.000000

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "c4d9e1a7f3b2"
down_revision = "b1e4c70d5f2a"
branch_labels = None
depends_on = None

RAW_SCHEMA = "raw"
CLEANED_SCHEMA = "cleaned"
RAW_TABLE_NAME = "ar5iv_no_problem_markdown"
MARKDOWN_TABLE_NAME = "ar5iv_no_problem_markdown"
METADATA_TABLE_NAME = "arxiv_metadata_oai_snapshot_2026_02_19"
JOINED_TABLE_NAME = "ar5iv_no_problem_markdown_with_metadata"


def _refresh_joined_table() -> None:
    op.execute(sa.text(f"TRUNCATE TABLE {CLEANED_SCHEMA}.{JOINED_TABLE_NAME}"))

    op.execute(
        sa.text(
            f"""
            INSERT INTO {CLEANED_SCHEMA}.{JOINED_TABLE_NAME} (
                id,
                authors,
                title,
                comments,
                journal_ref,
                doi,
                report_no,
                categories,
                license,
                created,
                last_updated,
                text
            )
            SELECT
                metadata.id,
                metadata.authors,
                metadata.title,
                metadata.comments,
                metadata.journal_ref,
                metadata.doi,
                metadata.report_no,
                metadata.categories,
                metadata.license,
                metadata.created,
                metadata.last_updated,
                markdown.text
            FROM {CLEANED_SCHEMA}.{METADATA_TABLE_NAME} AS metadata
            JOIN {CLEANED_SCHEMA}.{MARKDOWN_TABLE_NAME} AS markdown
              ON markdown.id = metadata.id
            """
        )
    )


def upgrade() -> None:
    op.execute(sa.text(f"TRUNCATE TABLE {CLEANED_SCHEMA}.{MARKDOWN_TABLE_NAME}"))

    op.execute(
        sa.text(
            f"""
            WITH raw_source AS (
                SELECT
                    regexp_replace(
                        regexp_replace(id, '^.*/', ''),
                        '\\.html$',
                        ''
                    ) AS base_id,
                    text
                FROM {RAW_SCHEMA}.{RAW_TABLE_NAME}
                WHERE NULLIF(trim(id), '') IS NOT NULL
            ),
            source AS (
                SELECT
                    CASE
                        WHEN base_id ~ '^[0-9]{{4}}\\.[0-9]{{4,5}}(v[0-9]+)?$' THEN base_id
                        WHEN base_id ~ '^.+[0-9]{{7}}(v[0-9]+)?$' THEN
                            regexp_replace(
                                base_id,
                                '^(.+?)([0-9]{{7}}(v[0-9]+)?)$',
                                '\\1/\\2'
                            )
                        ELSE base_id
                    END AS id,
                    text
                FROM raw_source
            ),
            deduped AS (
                SELECT DISTINCT ON (id)
                    id,
                    text
                FROM source
                WHERE NULLIF(trim(id), '') IS NOT NULL
                ORDER BY id
            )
            INSERT INTO {CLEANED_SCHEMA}.{MARKDOWN_TABLE_NAME} (id, text)
            SELECT id, text
            FROM deduped
            """
        )
    )

    _refresh_joined_table()


def downgrade() -> None:
    op.execute(sa.text(f"TRUNCATE TABLE {CLEANED_SCHEMA}.{MARKDOWN_TABLE_NAME}"))

    op.execute(
        sa.text(
            f"""
            WITH source AS (
                SELECT
                    regexp_replace(
                        regexp_replace(id, '^.*/', ''),
                        '\\.html$',
                        ''
                    ) AS id,
                    text
                FROM {RAW_SCHEMA}.{RAW_TABLE_NAME}
                WHERE NULLIF(trim(id), '') IS NOT NULL
            ),
            deduped AS (
                SELECT DISTINCT ON (id)
                    id,
                    text
                FROM source
                WHERE NULLIF(trim(id), '') IS NOT NULL
                ORDER BY id
            )
            INSERT INTO {CLEANED_SCHEMA}.{MARKDOWN_TABLE_NAME} (id, text)
            SELECT id, text
            FROM deduped
            """
        )
    )

    _refresh_joined_table()
