"""create analytics view

Revision ID: 577d2073d7a4
Revises: 9a2336d40053
Create Date: 2026-01-30 15:37:22.702782

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '577d2073d7a4'
down_revision = '9a2336d40053'
branch_labels = None
depends_on = None

VIEW_NAME = "analytics_documents_and_chunks_summary"

def upgrade() -> None:
    # 1) Create / replace the view (single command => OK)
    op.execute(sa.text(f"""
    CREATE OR REPLACE VIEW {VIEW_NAME} AS
    WITH per_document_aggregation AS (
        SELECT
            d.id AS document_id,
            COUNT(c.id) AS chunks_count_for_document,
            COUNT(c.id) FILTER (WHERE c.text <> '') AS chunks_with_non_empty_text_count_for_document
        FROM documents AS d
        LEFT JOIN document_chunks AS c
            ON c.document_id = d.id
        GROUP BY d.id
    ),
    overall_document_aggregation AS (
        SELECT
            COUNT(*)::bigint AS total_documents_count,

            COALESCE(SUM(chunks_count_for_document), 0)::bigint AS total_chunks_count,
            COALESCE(SUM(chunks_with_non_empty_text_count_for_document), 0)::bigint
                AS total_chunks_with_non_empty_text_count,

            AVG(chunks_count_for_document::numeric) AS average_chunks_count_per_document,
            AVG(chunks_with_non_empty_text_count_for_document::numeric)
                AS average_chunks_with_non_empty_text_count_per_document,

            MIN(chunks_count_for_document)::bigint AS minimum_chunks_count_per_document,
            MAX(chunks_count_for_document)::bigint AS maximum_chunks_count_per_document,

            MIN(chunks_with_non_empty_text_count_for_document)::bigint
                AS minimum_chunks_with_non_empty_text_count_per_document,
            MAX(chunks_with_non_empty_text_count_for_document)::bigint
                AS maximum_chunks_with_non_empty_text_count_per_document,

            COUNT(*) FILTER (WHERE chunks_count_for_document = 0)::bigint
                AS documents_with_zero_chunks_count,
            COUNT(*) FILTER (WHERE chunks_with_non_empty_text_count_for_document = 0)::bigint
                AS documents_with_zero_non_empty_text_chunks_count
        FROM per_document_aggregation
    ),
    chunk_quality_aggregation AS (
        SELECT
            COUNT(*) FILTER (WHERE c.text = '')::bigint AS total_chunks_with_empty_text_count,

            AVG(LENGTH(c.text))::numeric AS average_chunk_text_length,
            AVG(LENGTH(c.text)) FILTER (WHERE c.text <> '')::numeric
                AS average_non_empty_chunk_text_length,

            COUNT(*) FILTER (WHERE c.document_id IS NULL)::bigint
                AS chunks_with_null_document_identifier_count,
            COUNT(*) FILTER (WHERE c.document_id IS NOT NULL AND d.id IS NULL)::bigint
                AS chunks_pointing_to_missing_document_count
        FROM document_chunks AS c
        LEFT JOIN documents AS d
            ON d.id = c.document_id
    )
    SELECT
        o.total_documents_count,
        o.total_chunks_count,
        o.total_chunks_with_non_empty_text_count,

        o.average_chunks_count_per_document,
        o.average_chunks_with_non_empty_text_count_per_document,

        o.minimum_chunks_count_per_document,
        o.maximum_chunks_count_per_document,
        o.minimum_chunks_with_non_empty_text_count_per_document,
        o.maximum_chunks_with_non_empty_text_count_per_document,

        o.documents_with_zero_chunks_count,
        o.documents_with_zero_non_empty_text_chunks_count,

        q.total_chunks_with_empty_text_count,
        q.average_chunk_text_length,
        q.average_non_empty_chunk_text_length,

        q.chunks_with_null_document_identifier_count,
        q.chunks_pointing_to_missing_document_count,

        CASE
            WHEN o.total_chunks_count = 0 THEN 0
            ELSE (o.total_chunks_with_non_empty_text_count::numeric / o.total_chunks_count::numeric)
        END AS fraction_of_chunks_with_non_empty_text
    FROM overall_document_aggregation AS o
    CROSS JOIN chunk_quality_aggregation AS q
    ;
    """))

    # 2) Comment on view (single command => OK)
    op.execute(sa.text(f"""
        COMMENT ON VIEW {VIEW_NAME} IS
        'One-row operational analytics summary: documents and their chunks, including non-empty text coverage and orphan detection.';
    """))

    # 3) Column comments: one statement per execute (asyncpg-safe)
    comments: dict[str, str] = {
        "total_documents_count": "Total number of rows in documents.",
        "total_chunks_count": "Total number of rows in document_chunks (including empty-text chunks and orphan chunks).",
        "total_chunks_with_non_empty_text_count": "Total number of chunks whose text is not the empty string ''.",

        "average_chunks_count_per_document": "Mean number of chunks per document, counting documents with zero chunks.",
        "average_chunks_with_non_empty_text_count_per_document": "Mean number of non-empty-text chunks per document, counting documents with zero chunks.",

        "minimum_chunks_count_per_document": "Minimum chunk count across all documents (includes zero).",
        "maximum_chunks_count_per_document": "Maximum chunk count across all documents.",
        "minimum_chunks_with_non_empty_text_count_per_document": "Minimum non-empty-text chunk count across all documents (includes zero).",
        "maximum_chunks_with_non_empty_text_count_per_document": "Maximum non-empty-text chunk count across all documents.",

        "documents_with_zero_chunks_count": "Number of documents that currently have zero associated chunks.",
        "documents_with_zero_non_empty_text_chunks_count": "Number of documents that have zero associated chunks with non-empty text (could still have empty chunks).",

        "total_chunks_with_empty_text_count": "Number of chunks whose text equals the empty string ''.",
        "average_chunk_text_length": "Mean LENGTH(text) across all chunks (empty string counts as 0).",
        "average_non_empty_chunk_text_length": "Mean LENGTH(text) for chunks with non-empty text only.",

        "chunks_with_null_document_identifier_count": "Number of chunks whose document_id is NULL (unassigned to any document).",
        "chunks_pointing_to_missing_document_count": "Number of chunks whose document_id is not NULL but does not match any documents.id (broken foreign key / data inconsistency).",

        "fraction_of_chunks_with_non_empty_text": "total_chunks_with_non_empty_text_count / total_chunks_count; 0 if there are no chunks.",
    }

    for column_name, column_comment in comments.items():
        escaped_comment = column_comment.replace("'", "''")
        op.execute(
            sa.text(
                f"COMMENT ON COLUMN {VIEW_NAME}.{column_name} IS '{escaped_comment}'"
            )
        )


def downgrade() -> None:
    op.execute(sa.text(f"DROP VIEW IF EXISTS {VIEW_NAME};"))
