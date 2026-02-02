"""sync embedding eval schema

Revision ID: d1e7c8b1f9aa
Revises: 8f2d9f6a4b2e
Create Date: 2026-01-31 09:05:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "d1e7c8b1f9aa"
down_revision = "8f2d9f6a4b2e"
branch_labels = None
depends_on = None

SCHEMA_NAME = "evaluation"

EMBEDDING_MODEL_TABLES: list[tuple[str, int]] = [
    ("qwen3_embedding_0_6b", 1024),
    ("embeddinggemma_300m", 768),
    ("e5_large_v2", 1024),
    ("gte_large_en_v1_5", 1024),
    ("bge_m3", 1024),
    ("text_embedding_3_small", 1536),
    ("voyage_3_5", 1024),
]


def _conditional_rename(old_name: str, new_name: str) -> None:
    qualified_old = f"{SCHEMA_NAME}.{old_name}"
    qualified_new = f"{SCHEMA_NAME}.{new_name}"
    op.execute(
        sa.text(
            f"""
            DO $$
            BEGIN
                IF to_regclass('{qualified_old}') IS NOT NULL
                   AND to_regclass('{qualified_new}') IS NULL THEN
                    EXECUTE 'ALTER TABLE {qualified_old} RENAME TO {new_name}';
                END IF;
            END $$;
            """
        )
    )


def _create_base_tables() -> None:
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}"))

    # Clean up any legacy tables accidentally created in public schema.
    op.execute(sa.text("DROP TABLE IF EXISTS public.embedding_eval_documents CASCADE"))
    op.execute(sa.text("DROP TABLE IF EXISTS public.embedding_eval_queries CASCADE"))
    op.execute(sa.text("DROP TABLE IF EXISTS public.embedding_eval_pair_metrics CASCADE"))

    for model_key, _ in EMBEDDING_MODEL_TABLES:
        op.execute(
            sa.text(
                "DROP TABLE IF EXISTS public.embedding_eval_document_embeddings_"
                f"{model_key} CASCADE"
            )
        )
        op.execute(
            sa.text(
                "DROP TABLE IF EXISTS public.embedding_eval_query_embeddings_"
                f"{model_key} CASCADE"
            )
        )

    _conditional_rename("embedding_eval_documents", "documents")
    _conditional_rename("embedding_eval_queries", "queries")
    _conditional_rename("embedding_eval_pair_metrics", "pair_metrics")

    op.execute(
        sa.text(
            f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.documents (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                dataset_name text NOT NULL,
                source_id text NULL,
                text text NOT NULL,
                document_metadata json NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            );
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.queries (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                dataset_name text NOT NULL,
                text text NOT NULL,
                query_metadata json NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            );
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.pair_metrics (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                dataset_name text NOT NULL,
                run_name text NOT NULL,
                model_name text NOT NULL,
                quantization text NOT NULL,
                query_id uuid NOT NULL REFERENCES {SCHEMA_NAME}.queries(id) ON DELETE CASCADE,
                document_id uuid NOT NULL REFERENCES {SCHEMA_NAME}.documents(id) ON DELETE CASCADE,
                retrieval_rank integer NULL,
                retrieval_score double precision NULL,
                rerank_rank integer NULL,
                rerank_score double precision NULL,
                relevance_grade integer NULL,
                retrieval_latency_ms double precision NULL,
                rerank_latency_ms double precision NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                CONSTRAINT uq_pair_metrics UNIQUE (
                    run_name,
                    model_name,
                    quantization,
                    query_id,
                    document_id
                )
            );
            """
        )
    )

    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_documents_dataset_name "
            f"ON {SCHEMA_NAME}.documents (dataset_name);"
        )
    )
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_documents_source_id "
            f"ON {SCHEMA_NAME}.documents (source_id);"
        )
    )
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_queries_dataset_name "
            f"ON {SCHEMA_NAME}.queries (dataset_name);"
        )
    )
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_pair_metrics_dataset_name "
            f"ON {SCHEMA_NAME}.pair_metrics (dataset_name);"
        )
    )
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_pair_metrics_document_id "
            f"ON {SCHEMA_NAME}.pair_metrics (document_id);"
        )
    )
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_pair_metrics_model_name "
            f"ON {SCHEMA_NAME}.pair_metrics (model_name);"
        )
    )
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_pair_metrics_query_id "
            f"ON {SCHEMA_NAME}.pair_metrics (query_id);"
        )
    )
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_pair_metrics_quantization "
            f"ON {SCHEMA_NAME}.pair_metrics (quantization);"
        )
    )
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_pair_metrics_run_name "
            f"ON {SCHEMA_NAME}.pair_metrics (run_name);"
        )
    )


def _create_embedding_tables(kind: str, table_prefix: str, fk_table: str) -> None:
    for model_key, dim in EMBEDDING_MODEL_TABLES:
        old_table = f"embedding_eval_{table_prefix}_embeddings_{model_key}"
        new_table = f"{table_prefix}_embeddings_{model_key}"

        _conditional_rename(old_table, new_table)

        op.execute(
            sa.text(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.{new_table} (
                    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    {kind}_id uuid NOT NULL REFERENCES {SCHEMA_NAME}.{fk_table}(id) ON DELETE CASCADE,
                    quantization text NOT NULL DEFAULT 'fp32',
                    embedding vector({dim}) NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    CONSTRAINT uq_{table_prefix}_embeddings_{model_key} UNIQUE (
                        {kind}_id,
                        quantization
                    )
                );
                """
            )
        )
        op.execute(
            sa.text(
                f"CREATE INDEX IF NOT EXISTS ix_{table_prefix}_embeddings_{model_key}_{kind}_id "
                f"ON {SCHEMA_NAME}.{new_table} ({kind}_id);"
            )
        )
        op.execute(
            sa.text(
                f"CREATE INDEX IF NOT EXISTS ix_{table_prefix}_embeddings_{model_key}_quantization "
                f"ON {SCHEMA_NAME}.{new_table} (quantization);"
            )
        )
        op.execute(
            sa.text(
                f"ALTER TABLE {SCHEMA_NAME}.{new_table} "
                f"ALTER COLUMN embedding DROP NOT NULL;"
            )
        )
        op.execute(
            sa.text(
                f"ALTER TABLE {SCHEMA_NAME}.{new_table} "
                f"ADD COLUMN IF NOT EXISTS embedding_halfvec halfvec({dim});"
            )
        )
        op.execute(
            sa.text(
                f"ALTER TABLE {SCHEMA_NAME}.{new_table} "
                f"ADD COLUMN IF NOT EXISTS embedding_bit bit({dim});"
            )
        )


def upgrade() -> None:
    _create_base_tables()

    op.execute(
        sa.text(
            f"DROP TABLE IF EXISTS {SCHEMA_NAME}.embedding_eval_document_embeddings_snowflake_arctic_embed_l_v2_0 CASCADE;"
        )
    )
    op.execute(
        sa.text(
            f"DROP TABLE IF EXISTS {SCHEMA_NAME}.embedding_eval_query_embeddings_snowflake_arctic_embed_l_v2_0 CASCADE;"
        )
    )
    op.execute(
        sa.text(
            f"DROP TABLE IF EXISTS {SCHEMA_NAME}.document_embeddings_snowflake_arctic_embed_l_v2_0 CASCADE;"
        )
    )
    op.execute(
        sa.text(
            f"DROP TABLE IF EXISTS {SCHEMA_NAME}.query_embeddings_snowflake_arctic_embed_l_v2_0 CASCADE;"
        )
    )

    _create_embedding_tables("document", "document", "documents")
    _create_embedding_tables("query", "query", "queries")


def downgrade() -> None:
    # No-op: this migration is intended to be idempotent repair.
    pass
