"""create processed hybrid retrieval sql function

Revision ID: c8a1d4f6e2b9
Revises: b2c947ab7d22
Create Date: 2026-02-24 14:00:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "c8a1d4f6e2b9"
down_revision = "b2c947ab7d22"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        """
        CREATE OR REPLACE FUNCTION processed.hybrid_retrieve(
            p_query text,
            p_embedding vector(1024),
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
    )


def downgrade() -> None:
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
