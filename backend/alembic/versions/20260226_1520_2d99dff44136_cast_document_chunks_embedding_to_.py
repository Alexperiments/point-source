"""cast_document_chunks_embedding_to_halfvec

Revision ID: 2d99dff44136
Revises: e1b2c3d4f5a6
Create Date: 2026-02-26 15:20:04.812601

"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "2d99dff44136"
down_revision = "e1b2c3d4f5a6"
branch_labels = None
depends_on = None

_DROP_VECTOR_HYBRID_RETRIEVE_SQL = """
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

_DROP_HALFVEC_HYBRID_RETRIEVE_SQL = """
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


def _create_hybrid_retrieve_sql(embedding_type: str) -> str:
    return f"""
    CREATE OR REPLACE FUNCTION processed.hybrid_retrieve(
        p_query text,
        p_embedding {embedding_type},
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
        url text,
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
            d.url,
            c.text AS chunk_text,
            c.node_metadata,
            row_number() OVER (
                ORDER BY
                    ts_rank_cd(to_tsvector('simple', c.text), p.ts_query) DESC,
                    c.id
            ) AS text_rank
        FROM processed.document_chunks c
        JOIN processed.documents d
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
            d.url,
            c.text AS chunk_text,
            c.node_metadata,
            row_number() OVER (
                ORDER BY
                    (c.embedding <=> p_embedding),
                    c.id
            ) AS vector_rank
        FROM processed.document_chunks c
        JOIN processed.documents d
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
            COALESCE(t.url, v.url) AS url,
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
        url,
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


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("DROP INDEX IF EXISTS processed.ix_processed_document_chunks_embedding_hnsw")
    op.execute(
        """
        ALTER TABLE processed.document_chunks
        ALTER COLUMN embedding TYPE halfvec(1024)
        USING embedding::halfvec(1024)
        """,
    )
    op.execute(_DROP_VECTOR_HYBRID_RETRIEVE_SQL)
    op.execute(_create_hybrid_retrieve_sql("halfvec(1024)"))


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS processed.ix_processed_document_chunks_embedding_hnsw")
    op.execute(
        """
        ALTER TABLE processed.document_chunks
        ALTER COLUMN embedding TYPE vector(1024)
        USING embedding::vector(1024)
        """,
    )
    op.execute(_DROP_HALFVEC_HYBRID_RETRIEVE_SQL)
    op.execute(_create_hybrid_retrieve_sql("vector(1024)"))
