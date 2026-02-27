"""refresh joined metadata and processed documents, fix doi_url format

Revision ID: 3f2c8e4b9d10
Revises: 0b9dfab351b1
Create Date: 2026-02-27 16:00:00.000000

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "3f2c8e4b9d10"
down_revision = "0b9dfab351b1"
branch_labels = None
depends_on = None

CLEANED_SCHEMA = "cleaned"
PROCESSED_SCHEMA = "processed"
METADATA_TABLE_NAME = "arxiv_metadata_oai_snapshot_2026_02_19"
MARKDOWN_TABLE_NAME = "ar5iv_no_problem_markdown"
JOINED_TABLE_NAME = "ar5iv_no_problem_markdown_with_metadata"
DOCUMENTS_TABLE_NAME = "documents"
CHUNKS_TABLE_NAME = "document_chunks"
DOI_PREFIX = "https://www.doi.org/"
ARXIV_DOI_PREFIX = "10.48550/arXiv."


def _sync_joined_table() -> None:
    op.execute(
        sa.text(
            f"""
            ALTER TABLE {CLEANED_SCHEMA}.{JOINED_TABLE_NAME}
            ADD COLUMN IF NOT EXISTS number_of_authors INTEGER NOT NULL DEFAULT 0
            """
        )
    )

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
                text,
                number_of_authors
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
                markdown.text,
                metadata.number_of_authors
            FROM {CLEANED_SCHEMA}.{METADATA_TABLE_NAME} AS metadata
            JOIN {CLEANED_SCHEMA}.{MARKDOWN_TABLE_NAME} AS markdown
              ON markdown.id = metadata.id
            ON CONFLICT (id) DO NOTHING
            """
        )
    )

    op.execute(
        sa.text(
            f"""
            ALTER TABLE {CLEANED_SCHEMA}.{JOINED_TABLE_NAME}
            ALTER COLUMN number_of_authors DROP DEFAULT
            """
        )
    )


def _sync_processed_documents() -> None:
    op.execute(
        sa.text(
            f"""
            ALTER TABLE {PROCESSED_SCHEMA}.{DOCUMENTS_TABLE_NAME}
            ADD COLUMN IF NOT EXISTS number_of_authors INTEGER NOT NULL DEFAULT 0
            """
        )
    )

    op.execute(
        sa.text(
            f"""
            WITH source AS (
                SELECT
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
                    text,
                    number_of_authors,
                    CASE
                        WHEN NULLIF(trim(doi), '') IS NOT NULL
                            THEN :doi_prefix || trim(doi)
                        ELSE :doi_prefix || :arxiv_doi_prefix || id
                    END AS doi_url
                FROM {CLEANED_SCHEMA}.{JOINED_TABLE_NAME}
                WHERE categories ILIKE '%astro-ph%'
            )
            INSERT INTO {PROCESSED_SCHEMA}.{DOCUMENTS_TABLE_NAME} (
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
                text,
                doi_url,
                number_of_authors
            )
            SELECT
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
                text,
                doi_url,
                number_of_authors
            FROM source
            ON CONFLICT (id) DO UPDATE SET
                authors = EXCLUDED.authors,
                title = EXCLUDED.title,
                comments = EXCLUDED.comments,
                journal_ref = EXCLUDED.journal_ref,
                doi = EXCLUDED.doi,
                report_no = EXCLUDED.report_no,
                categories = EXCLUDED.categories,
                license = EXCLUDED.license,
                created = EXCLUDED.created,
                last_updated = EXCLUDED.last_updated,
                text = EXCLUDED.text,
                doi_url = EXCLUDED.doi_url,
                number_of_authors = EXCLUDED.number_of_authors
            """
        ).bindparams(
            doi_prefix=DOI_PREFIX,
            arxiv_doi_prefix=ARXIV_DOI_PREFIX,
        )
    )

    op.execute(
        sa.text(
            f"""
            WITH source AS (
                SELECT id
                FROM {CLEANED_SCHEMA}.{JOINED_TABLE_NAME}
                WHERE categories ILIKE '%astro-ph%'
            )
            DELETE FROM {PROCESSED_SCHEMA}.{DOCUMENTS_TABLE_NAME} AS d
            WHERE NOT EXISTS (
                SELECT 1
                FROM source AS s
                WHERE s.id = d.id
            )
            AND NOT EXISTS (
                SELECT 1
                FROM {PROCESSED_SCHEMA}.{CHUNKS_TABLE_NAME} AS c
                WHERE c.document_id = d.id
            )
            """
        )
    )

    op.execute(
        sa.text(
            f"""
            UPDATE {PROCESSED_SCHEMA}.{DOCUMENTS_TABLE_NAME}
            SET doi_url = CASE
                WHEN NULLIF(trim(doi), '') IS NOT NULL
                    THEN :doi_prefix || trim(doi)
                ELSE :doi_prefix || :arxiv_doi_prefix || id
            END
            """
        ).bindparams(
            doi_prefix=DOI_PREFIX,
            arxiv_doi_prefix=ARXIV_DOI_PREFIX,
        )
    )

    op.execute(
        sa.text(
            f"""
            ALTER TABLE {PROCESSED_SCHEMA}.{DOCUMENTS_TABLE_NAME}
            ALTER COLUMN number_of_authors DROP DEFAULT
            """
        )
    )


def upgrade() -> None:
    _sync_joined_table()
    _sync_processed_documents()


def downgrade() -> None:
    op.drop_column(
        DOCUMENTS_TABLE_NAME,
        "number_of_authors",
        schema=PROCESSED_SCHEMA,
    )
    op.drop_column(
        JOINED_TABLE_NAME,
        "number_of_authors",
        schema=CLEANED_SCHEMA,
    )
