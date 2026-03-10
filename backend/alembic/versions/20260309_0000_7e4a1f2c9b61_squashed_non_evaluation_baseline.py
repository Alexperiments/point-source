"""squashed non-evaluation baseline

Revision ID: 7e4a1f2c9b61
Revises:
Create Date: 2026-03-09 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import HALFVEC, VECTOR


# revision identifiers, used by Alembic.
revision = "7e4a1f2c9b61"
down_revision = None
branch_labels = None
depends_on = None

RAW_SCHEMA = "raw"
CLEANED_SCHEMA = "cleaned"
PROCESSED_SCHEMA = "processed"

RAW_METADATA_TABLE = "arxiv_metadata_oai_snapshot_2026_02_19"
RAW_MARKDOWN_TABLE = "ar5iv_no_problem_markdown"
CLEANED_METADATA_TABLE = "arxiv_metadata_oai_snapshot_2026_02_19"
CLEANED_MARKDOWN_TABLE = "ar5iv_no_problem_markdown"
CLEANED_JOINED_TABLE = "ar5iv_no_problem_markdown_with_metadata"
PROCESSED_DOCUMENTS_TABLE = "documents"
PROCESSED_CHUNKS_TABLE = "document_chunks"

ANALYTICS_VIEW_NAME = "analytics_documents_and_chunks_summary"
AR5IV_DATASET_NAME = "marin-community/ar5iv-no-problem-markdown"
DOI_PREFIX = "https://www.doi.org/"
ARXIV_DOI_PREFIX = "10.48550/arXiv."

PUBLIC_DOCUMENTS_TABLE = "documents"
PUBLIC_CHUNKS_TABLE = "document_chunks"

PUBLIC_CHUNKS_EMBEDDING_HNSW = "ix_document_chunks_embedding_hnsw"
PUBLIC_DOCUMENTS_EMBEDDING_HNSW = "ix_documents_embedding_hnsw"
PUBLIC_CHUNKS_TEXT_GIN = "ix_document_chunks_text_tsv_gin"
PUBLIC_DOCUMENTS_TEXT_GIN = "ix_documents_text_tsv_gin"
PUBLIC_CHUNKS_METADATA_GIN = "ix_document_chunks_node_metadata_gin"
PUBLIC_CHUNKS_DOCUMENT_ID = "ix_document_chunks_document_id"
PUBLIC_CHUNKS_PREV_ID = "ix_document_chunks_prev_id"

PROCESSED_CHUNKS_EMBEDDING_HNSW = "ix_processed_document_chunks_embedding_hnsw"
PROCESSED_CHUNKS_TEXT_GIN = "ix_processed_document_chunks_text_tsv_gin"
PROCESSED_CHUNKS_METADATA_GIN = "ix_processed_document_chunks_node_metadata_gin"
PROCESSED_CHUNKS_DOCUMENT_ID = "ix_processed_document_chunks_document_id"
PROCESSED_CHUNKS_PREV_ID = "ix_processed_document_chunks_prev_id"
PROCESSED_CHUNKS_PARENT_ID = "ix_processed_document_chunks_parent_id"
PROCESSED_CHUNKS_UNEMBEDDED_ID = "ix_processed_document_chunks_unembedded_id"

PROCESSED_DOCUMENTS_PK = "processed_documents_pkey"


def _author_item_expr(index: str) -> str:
    if index == "1":
        return f"""
        NULLIF(
            trim(
                regexp_replace(
                    author_item->>{index},
                    '[[:space:]]*-[[:space:]]*',
                    '-',
                    'g'
                )
            ),
            ''
        )
        """
    return f"NULLIF(trim(author_item->>{index}), '')"


def _authors_expression(*, order: str) -> str:
    return f"""
    CASE
        WHEN jsonb_typeof(payload->'authors_parsed') = 'array' THEN (
            SELECT string_agg(author_name, ', ' ORDER BY ord)
            FROM (
                SELECT
                    NULLIF(
                        trim(
                            regexp_replace(
                                concat_ws(
                                    ' ',
                                    {_author_item_expr(order[0])},
                                    {_author_item_expr(order[1])},
                                    {_author_item_expr(order[2])}
                                ),
                                '[[:space:]]+',
                                ' ',
                                'g'
                            )
                        ),
                        ''
                    ) AS author_name,
                    ord
                FROM jsonb_array_elements(payload->'authors_parsed')
                WITH ORDINALITY AS author_parts(author_item, ord)
            ) AS normalized
            WHERE author_name IS NOT NULL
        )
        ELSE NULL
    END
    """


def _number_of_authors_expression(*, order: str) -> str:
    return f"""
    CASE
        WHEN jsonb_typeof(payload->'authors_parsed') = 'array' THEN (
            SELECT COUNT(*)
            FROM (
                SELECT
                    NULLIF(
                        trim(
                            regexp_replace(
                                concat_ws(
                                    ' ',
                                    {_author_item_expr(order[0])},
                                    {_author_item_expr(order[1])},
                                    {_author_item_expr(order[2])}
                                ),
                                '[[:space:]]+',
                                ' ',
                                'g'
                            )
                        ),
                        ''
                    ) AS author_name
                FROM jsonb_array_elements(payload->'authors_parsed')
                WITH ORDINALITY AS author_parts(author_item, ord)
            ) AS normalized
            WHERE author_name IS NOT NULL
        )::integer
        ELSE 0
    END
    """


def _create_analytics_view() -> None:
    op.execute(
        sa.text(
            f"""
            CREATE OR REPLACE VIEW {ANALYTICS_VIEW_NAME} AS
            WITH per_document_aggregation AS (
                SELECT
                    d.id AS document_id,
                    COUNT(c.id) AS chunks_count_for_document,
                    COUNT(c.id) FILTER (WHERE c.text <> '') AS chunks_with_non_empty_text_count_for_document
                FROM {PUBLIC_DOCUMENTS_TABLE} AS d
                LEFT JOIN {PUBLIC_CHUNKS_TABLE} AS c
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
                FROM {PUBLIC_CHUNKS_TABLE} AS c
                LEFT JOIN {PUBLIC_DOCUMENTS_TABLE} AS d
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
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            COMMENT ON VIEW {ANALYTICS_VIEW_NAME} IS
            'One-row operational analytics summary: documents and their chunks, including non-empty text coverage and orphan detection.'
            """
        )
    )

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
                f"COMMENT ON COLUMN {ANALYTICS_VIEW_NAME}.{column_name} IS '{escaped_comment}'"
            )
        )


def _drop_hybrid_retrieve_functions() -> None:
    op.execute(
        """
        DROP FUNCTION IF EXISTS processed.hybrid_retrieve(
            text,
            vector,
            integer,
            integer,
            integer,
            integer,
            double precision,
            double precision
        )
        """
    )
    op.execute(
        """
        DROP FUNCTION IF EXISTS processed.hybrid_retrieve(
            text,
            halfvec,
            integer,
            integer,
            integer,
            integer,
            double precision,
            double precision
        )
        """
    )


def _create_hybrid_retrieve_function() -> None:
    _drop_hybrid_retrieve_functions()
    op.execute(
        sa.text(
            f"""
            CREATE OR REPLACE FUNCTION {PROCESSED_SCHEMA}.hybrid_retrieve(
                p_query text,
                p_embedding halfvec(1024),
                p_text_k integer,
                p_vec_k integer,
                p_fused_k integer,
                p_rrf_k integer,
                p_text_weight double precision,
                p_vector_weight double precision
            )
            RETURNS TABLE (
                chunk_id uuid,
                document_id text,
                doi_url text,
                chunk_text text,
                node_metadata json,
                text_rank integer,
                vector_rank integer,
                rrf_score double precision
            )
            LANGUAGE sql
            STABLE
            AS $$
            WITH
            params AS (
                SELECT websearch_to_tsquery('simple', p_query) AS ts_query
            ),
            text_candidates AS (
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    d.doi_url,
                    c.text AS chunk_text,
                    c.node_metadata,
                    row_number() OVER (
                        ORDER BY
                            ts_rank_cd(to_tsvector('simple', c.text), p.ts_query) DESC,
                            c.id
                    ) AS text_rank
                FROM {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_TABLE} c
                JOIN {PROCESSED_SCHEMA}.{PROCESSED_DOCUMENTS_TABLE} d
                  ON d.id = c.document_id
                CROSS JOIN params p
                WHERE c.text <> ''
                  AND to_tsvector('simple', c.text) @@ p.ts_query
                ORDER BY
                    ts_rank_cd(to_tsvector('simple', c.text), p.ts_query) DESC,
                    c.id
                LIMIT GREATEST(p_text_k, 0)
            ),
            vector_candidates AS (
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    d.doi_url,
                    c.text AS chunk_text,
                    c.node_metadata,
                    row_number() OVER (
                        ORDER BY
                            (c.embedding <=> p_embedding),
                            c.id
                    ) AS vector_rank
                FROM {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_TABLE} c
                JOIN {PROCESSED_SCHEMA}.{PROCESSED_DOCUMENTS_TABLE} d
                  ON d.id = c.document_id
                WHERE c.text <> ''
                  AND c.embedding IS NOT NULL
                ORDER BY
                    (c.embedding <=> p_embedding),
                    c.id
                LIMIT GREATEST(p_vec_k, 0)
            ),
            fused AS (
                SELECT
                    COALESCE(t.chunk_id, v.chunk_id) AS chunk_id,
                    COALESCE(t.document_id, v.document_id) AS document_id,
                    COALESCE(t.doi_url, v.doi_url) AS doi_url,
                    COALESCE(t.chunk_text, v.chunk_text) AS chunk_text,
                    COALESCE(t.node_metadata, v.node_metadata) AS node_metadata,
                    t.text_rank,
                    v.vector_rank,
                    (
                        CASE
                            WHEN t.text_rank IS NULL THEN 0.0
                            ELSE p_text_weight / ((GREATEST(p_rrf_k, 0) + t.text_rank)::double precision)
                        END
                    ) + (
                        CASE
                            WHEN v.vector_rank IS NULL THEN 0.0
                            ELSE p_vector_weight / ((GREATEST(p_rrf_k, 0) + v.vector_rank)::double precision)
                        END
                    ) AS rrf_score
                FROM text_candidates t
                FULL OUTER JOIN vector_candidates v
                    ON v.chunk_id = t.chunk_id
            )
            SELECT
                chunk_id,
                document_id,
                doi_url,
                chunk_text,
                node_metadata,
                text_rank,
                vector_rank,
                rrf_score
            FROM fused
            ORDER BY
                rrf_score DESC,
                chunk_id
            LIMIT GREATEST(p_fused_k, 0)
            $$;
            """
        )
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "prompts",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("slug", sa.String(length=100), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_prompts_slug", "prompts", ["slug"], unique=True)

    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("is_superuser", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("is_premium", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_is_deleted", "users", ["is_deleted"], unique=False)

    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("prompt_id", sa.UUID(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("commit_message", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("created_by_id", sa.UUID(), nullable=True),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["prompt_id"], ["prompts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("prompt_id", "version_number", name="uq_prompt_version_number"),
    )
    op.create_index("ix_prompt_versions_prompt_id", "prompt_versions", ["prompt_id"], unique=False)

    op.create_table(
        "threads",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("is_archived", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_threads_user_id", "threads", ["user_id"], unique=False)

    op.create_table(
        "messages",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("thread_id", sa.UUID(), nullable=False),
        sa.Column("sequence_num", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("role", sa.Enum("USER", "ASSISTANT", "SYSTEM", name="message_role"), nullable=False),
        sa.Column("parts", sa.JSON(), server_default=sa.text("'[]'::json"), nullable=False),
        sa.Column("extra_data", sa.JSON(), server_default=sa.text("'{}'::json"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["thread_id"], ["threads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_message_thread_seq", "messages", ["thread_id", "sequence_num"], unique=False)
    op.create_index("ix_messages_thread_id", "messages", ["thread_id"], unique=False)

    op.create_table(
        PUBLIC_DOCUMENTS_TABLE,
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("embedding", VECTOR(dim=1024), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("node_metadata", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id"),
    )
    op.create_table(
        PUBLIC_CHUNKS_TABLE,
        sa.Column("document_id", sa.UUID(), nullable=True),
        sa.Column("parent_id", sa.UUID(), nullable=True),
        sa.Column("prev_id", sa.UUID(), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("embedding", VECTOR(dim=1024), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("node_metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], [f"{PUBLIC_DOCUMENTS_TABLE}.id"]),
        sa.ForeignKeyConstraint(["parent_id"], [f"{PUBLIC_CHUNKS_TABLE}.id"]),
        sa.ForeignKeyConstraint(["prev_id"], [f"{PUBLIC_CHUNKS_TABLE}.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {PUBLIC_CHUNKS_EMBEDDING_HNSW} "
        f"ON {PUBLIC_CHUNKS_TABLE} USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 128)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {PUBLIC_DOCUMENTS_EMBEDDING_HNSW} "
        f"ON {PUBLIC_DOCUMENTS_TABLE} USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 128)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {PUBLIC_CHUNKS_TEXT_GIN} "
        f"ON {PUBLIC_CHUNKS_TABLE} USING gin (to_tsvector('simple', text))"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {PUBLIC_DOCUMENTS_TEXT_GIN} "
        f"ON {PUBLIC_DOCUMENTS_TABLE} USING gin (to_tsvector('simple', text))"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {PUBLIC_CHUNKS_METADATA_GIN} "
        f"ON {PUBLIC_CHUNKS_TABLE} USING gin ((node_metadata::jsonb))"
    )
    op.create_index(PUBLIC_CHUNKS_DOCUMENT_ID, PUBLIC_CHUNKS_TABLE, ["document_id"], unique=False)
    op.create_index(PUBLIC_CHUNKS_PREV_ID, PUBLIC_CHUNKS_TABLE, ["prev_id"], unique=False)

    _create_analytics_view()

    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}"))
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {CLEANED_SCHEMA}"))
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {PROCESSED_SCHEMA}"))

    op.create_table(
        RAW_METADATA_TABLE,
        sa.Column("raw_json", sa.JSON(), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        schema=RAW_SCHEMA,
    )

    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS ai CASCADE"))
    op.execute(
        sa.text(
            f"""
            DO $$
            import site
            version = plpy.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'ai' LIMIT 1"
            )[0]["extversion"]
            site.addsitedir("/usr/local/lib/pgai/" + version)
            import datasets
            split_expr = "train[" + ":" + "1]"
            datasets.load_dataset(
                "{AR5IV_DATASET_NAME}",
                split=split_expr,
                streaming=False,
            )
            $$ LANGUAGE plpython3u;
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            SELECT ai.load_dataset(
                '{AR5IV_DATASET_NAME}',
                table_name => '{RAW_MARKDOWN_TABLE}',
                schema_name => '{RAW_SCHEMA}'
            );
            """
        )
    )

    op.create_table(
        CLEANED_METADATA_TABLE,
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("authors", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("comments", sa.Text(), nullable=True),
        sa.Column("journal_ref", sa.Text(), nullable=True),
        sa.Column("doi", sa.Text(), nullable=True),
        sa.Column("report_no", sa.Text(), nullable=True),
        sa.Column("categories", sa.Text(), nullable=True),
        sa.Column("license", sa.Text(), nullable=True),
        sa.Column("created", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_updated", sa.DateTime(timezone=True), nullable=True),
        sa.Column("number_of_authors", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema=CLEANED_SCHEMA,
    )
    op.execute(
        sa.text(
            f"""
            WITH source AS (
                SELECT COALESCE(raw_json->'root', raw_json) AS payload
                FROM {RAW_SCHEMA}.{RAW_METADATA_TABLE}
            )
            INSERT INTO {CLEANED_SCHEMA}.{CLEANED_METADATA_TABLE} (
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
                number_of_authors
            )
            SELECT
                payload->>'id' AS id,
                {_authors_expression(order="012")} AS authors,
                payload->>'title' AS title,
                payload->>'comments' AS comments,
                payload->>'journal-ref' AS journal_ref,
                payload->>'doi' AS doi,
                payload->>'report-no' AS report_no,
                payload->>'categories' AS categories,
                payload->>'license' AS license,
                CASE
                    WHEN jsonb_typeof(payload->'versions') = 'array'
                     AND jsonb_array_length(payload->'versions') > 0 THEN
                        ((payload->'versions')->0->>'created')::timestamptz
                    ELSE NULL
                END AS created,
                CASE
                    WHEN jsonb_typeof(payload->'versions') = 'array'
                     AND jsonb_array_length(payload->'versions') > 0 THEN (
                        SELECT (version_item->>'created')::timestamptz
                        FROM jsonb_array_elements(payload->'versions')
                        WITH ORDINALITY AS version_parts(version_item, ord)
                        ORDER BY ord DESC
                        LIMIT 1
                    )
                    ELSE NULL
                END AS last_updated,
                {_number_of_authors_expression(order="012")} AS number_of_authors
            FROM source
            WHERE NULLIF(trim(payload->>'id'), '') IS NOT NULL
            ON CONFLICT (id) DO NOTHING
            """
        )
    )

    op.create_table(
        CLEANED_MARKDOWN_TABLE,
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        schema=CLEANED_SCHEMA,
    )
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
                FROM {RAW_SCHEMA}.{RAW_MARKDOWN_TABLE}
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
            INSERT INTO {CLEANED_SCHEMA}.{CLEANED_MARKDOWN_TABLE} (id, text)
            SELECT id, text
            FROM deduped
            ON CONFLICT (id) DO NOTHING
            """
        )
    )

    op.create_table(
        CLEANED_JOINED_TABLE,
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("authors", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("comments", sa.Text(), nullable=True),
        sa.Column("journal_ref", sa.Text(), nullable=True),
        sa.Column("doi", sa.Text(), nullable=True),
        sa.Column("report_no", sa.Text(), nullable=True),
        sa.Column("categories", sa.Text(), nullable=True),
        sa.Column("license", sa.Text(), nullable=True),
        sa.Column("created", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_updated", sa.DateTime(timezone=True), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("number_of_authors", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema=CLEANED_SCHEMA,
    )
    op.execute(
        sa.text(
            f"""
            INSERT INTO {CLEANED_SCHEMA}.{CLEANED_JOINED_TABLE} (
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
            FROM {CLEANED_SCHEMA}.{CLEANED_METADATA_TABLE} AS metadata
            JOIN {CLEANED_SCHEMA}.{CLEANED_MARKDOWN_TABLE} AS markdown
              ON markdown.id = metadata.id
            ON CONFLICT (id) DO NOTHING
            """
        )
    )

    op.create_table(
        PROCESSED_DOCUMENTS_TABLE,
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("authors", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("comments", sa.Text(), nullable=True),
        sa.Column("journal_ref", sa.Text(), nullable=True),
        sa.Column("doi", sa.Text(), nullable=True),
        sa.Column("report_no", sa.Text(), nullable=True),
        sa.Column("categories", sa.Text(), nullable=True),
        sa.Column("license", sa.Text(), nullable=True),
        sa.Column("created", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_updated", sa.DateTime(timezone=True), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("doi_url", sa.Text(), nullable=False),
        sa.Column("number_of_authors", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=PROCESSED_DOCUMENTS_PK),
        schema=PROCESSED_SCHEMA,
    )
    op.execute(
        sa.text(
            f"""
            INSERT INTO {PROCESSED_SCHEMA}.{PROCESSED_DOCUMENTS_TABLE} (
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
                CASE
                    WHEN NULLIF(trim(doi), '') IS NOT NULL
                        THEN :doi_prefix || trim(doi)
                    ELSE :doi_prefix || :arxiv_doi_prefix || id
                END AS doi_url,
                number_of_authors
            FROM {CLEANED_SCHEMA}.{CLEANED_JOINED_TABLE}
            WHERE categories ILIKE '%astro-ph%'
            """
        ).bindparams(
            doi_prefix=DOI_PREFIX,
            arxiv_doi_prefix=ARXIV_DOI_PREFIX,
        )
    )

    op.create_table(
        PROCESSED_CHUNKS_TABLE,
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("parent_id", sa.UUID(), nullable=True),
        sa.Column("prev_id", sa.UUID(), nullable=True),
        sa.Column("embedding", HALFVEC(dim=1024), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("node_metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], [f"{PROCESSED_SCHEMA}.{PROCESSED_DOCUMENTS_TABLE}.id"]),
        sa.ForeignKeyConstraint(["parent_id"], [f"{PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_TABLE}.id"]),
        sa.ForeignKeyConstraint(["prev_id"], [f"{PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_TABLE}.id"]),
        sa.PrimaryKeyConstraint("id"),
        schema=PROCESSED_SCHEMA,
    )

    op.execute(
        f"CREATE INDEX IF NOT EXISTS {PROCESSED_CHUNKS_TEXT_GIN} "
        f"ON {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_TABLE} "
        "USING gin (to_tsvector('simple', text))"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {PROCESSED_CHUNKS_METADATA_GIN} "
        f"ON {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_TABLE} "
        "USING gin ((node_metadata::jsonb))"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {PROCESSED_CHUNKS_DOCUMENT_ID} "
        f"ON {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_TABLE} (document_id)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {PROCESSED_CHUNKS_PREV_ID} "
        f"ON {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_TABLE} (prev_id)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {PROCESSED_CHUNKS_PARENT_ID} "
        f"ON {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_TABLE} (parent_id)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {PROCESSED_CHUNKS_UNEMBEDDED_ID} "
        f"ON {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_TABLE} (id) "
        "WHERE embedding IS NULL AND text <> ''"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {PROCESSED_CHUNKS_EMBEDDING_HNSW} "
        f"ON {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_TABLE} "
        "USING hnsw (embedding halfvec_cosine_ops) "
        "WITH (m = 16, ef_construction = 128)"
    )

    _create_hybrid_retrieve_function()


def downgrade() -> None:
    _drop_hybrid_retrieve_functions()

    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_EMBEDDING_HNSW}")
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_UNEMBEDDED_ID}")
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_PARENT_ID}")
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_PREV_ID}")
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_DOCUMENT_ID}")
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_METADATA_GIN}")
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{PROCESSED_CHUNKS_TEXT_GIN}")
    op.drop_table(PROCESSED_CHUNKS_TABLE, schema=PROCESSED_SCHEMA)
    op.drop_table(PROCESSED_DOCUMENTS_TABLE, schema=PROCESSED_SCHEMA)
    op.execute(sa.text(f"DROP SCHEMA IF EXISTS {PROCESSED_SCHEMA}"))

    op.drop_table(CLEANED_JOINED_TABLE, schema=CLEANED_SCHEMA)
    op.drop_table(CLEANED_MARKDOWN_TABLE, schema=CLEANED_SCHEMA)
    op.drop_table(CLEANED_METADATA_TABLE, schema=CLEANED_SCHEMA)
    op.execute(sa.text(f"DROP SCHEMA IF EXISTS {CLEANED_SCHEMA}"))

    op.execute(sa.text(f'DROP TABLE IF EXISTS {RAW_SCHEMA}."{RAW_MARKDOWN_TABLE}"'))
    op.drop_table(RAW_METADATA_TABLE, schema=RAW_SCHEMA)
    op.execute(sa.text("DROP EXTENSION IF EXISTS ai"))
    op.execute(sa.text(f"DROP SCHEMA IF EXISTS {RAW_SCHEMA}"))

    op.execute(sa.text(f"DROP VIEW IF EXISTS {ANALYTICS_VIEW_NAME}"))

    op.drop_index(PUBLIC_CHUNKS_PREV_ID, table_name=PUBLIC_CHUNKS_TABLE)
    op.drop_index(PUBLIC_CHUNKS_DOCUMENT_ID, table_name=PUBLIC_CHUNKS_TABLE)
    op.execute(f"DROP INDEX IF EXISTS {PUBLIC_CHUNKS_METADATA_GIN}")
    op.execute(f"DROP INDEX IF EXISTS {PUBLIC_DOCUMENTS_TEXT_GIN}")
    op.execute(f"DROP INDEX IF EXISTS {PUBLIC_CHUNKS_TEXT_GIN}")
    op.execute(f"DROP INDEX IF EXISTS {PUBLIC_DOCUMENTS_EMBEDDING_HNSW}")
    op.execute(f"DROP INDEX IF EXISTS {PUBLIC_CHUNKS_EMBEDDING_HNSW}")
    op.drop_table(PUBLIC_CHUNKS_TABLE)
    op.drop_table(PUBLIC_DOCUMENTS_TABLE)

    op.drop_index("ix_messages_thread_id", table_name="messages")
    op.drop_index("ix_message_thread_seq", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_threads_user_id", table_name="threads")
    op.drop_table("threads")
    op.drop_index("ix_prompt_versions_prompt_id", table_name="prompt_versions")
    op.drop_table("prompt_versions")
    op.drop_index("ix_users_is_deleted", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
    op.drop_index("ix_prompts_slug", table_name="prompts")
    op.drop_table("prompts")
    op.execute("DROP TYPE IF EXISTS message_role")
