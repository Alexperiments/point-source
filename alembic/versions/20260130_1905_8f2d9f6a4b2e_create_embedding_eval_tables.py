"""create embedding eval tables

Revision ID: 8f2d9f6a4b2e
Revises: 577d2073d7a4
Create Date: 2026-01-30 19:05:00.000000

"""
from alembic import op
import sqlalchemy as sa
import pgvector


# revision identifiers, used by Alembic.
revision = "8f2d9f6a4b2e"
down_revision = "577d2073d7a4"
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


def _create_embedding_tables(kind: str) -> None:
    fk_table = "documents" if kind == "document" else "queries"
    for model_key, dim in EMBEDDING_MODEL_TABLES:
        table_name = f"{kind}_embeddings_{model_key}"
        op.create_table(
            table_name,
            sa.Column(
                "id",
                sa.UUID(),
                server_default=sa.text("gen_random_uuid()"),
                nullable=False,
            ),
            sa.Column(f"{kind}_id", sa.UUID(), nullable=False),
            sa.Column(
                "quantization",
                sa.Text(),
                nullable=False,
                server_default=sa.text("'fp32'"),
            ),
            sa.Column(
                "embedding",
                pgvector.sqlalchemy.vector.VECTOR(dim=dim),
                nullable=True,
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(
                [f"{kind}_id"],
                [f"{SCHEMA_NAME}.{fk_table}.id"],
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                f"{kind}_id",
                "quantization",
                name=f"uq_{kind}_embeddings_{model_key}",
            ),
            schema=SCHEMA_NAME,
        )
        op.create_index(
            op.f(f"ix_{kind}_embeddings_{model_key}_{kind}_id"),
            table_name,
            [f"{kind}_id"],
            unique=False,
            schema=SCHEMA_NAME,
        )
        op.create_index(
            op.f(
                f"ix_{kind}_embeddings_{model_key}_quantization"
            ),
            table_name,
            ["quantization"],
            unique=False,
            schema=SCHEMA_NAME,
        )
        op.execute(
            sa.text(
                "ALTER TABLE "
                f"{SCHEMA_NAME}.{table_name} "
                f"ADD COLUMN IF NOT EXISTS embedding_halfvec halfvec({dim})"
            )
        )
        op.execute(
            sa.text(
                "ALTER TABLE "
                f"{SCHEMA_NAME}.{table_name} "
                f"ADD COLUMN IF NOT EXISTS embedding_bit bit({dim})"
            )
        )


def _drop_embedding_tables(kind: str) -> None:
    for model_key, _ in reversed(EMBEDDING_MODEL_TABLES):
        table_name = f"{kind}_embeddings_{model_key}"
        op.drop_index(
            op.f(
                f"ix_{kind}_embeddings_{model_key}_quantization"
            ),
            table_name=table_name,
            schema=SCHEMA_NAME,
        )
        op.drop_index(
            op.f(f"ix_{kind}_embeddings_{model_key}_{kind}_id"),
            table_name=table_name,
            schema=SCHEMA_NAME,
        )
        op.drop_table(table_name, schema=SCHEMA_NAME)


def upgrade() -> None:
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}"))

    op.create_table(
        "documents",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("dataset_name", sa.Text(), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("document_metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA_NAME,
    )
    op.create_index(
        op.f("ix_documents_dataset_name"),
        "documents",
        ["dataset_name"],
        unique=False,
        schema=SCHEMA_NAME,
    )
    op.create_index(
        op.f("ix_documents_source_id"),
        "documents",
        ["source_id"],
        unique=False,
        schema=SCHEMA_NAME,
    )

    op.create_table(
        "queries",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("dataset_name", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("query_metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA_NAME,
    )
    op.create_index(
        op.f("ix_queries_dataset_name"),
        "queries",
        ["dataset_name"],
        unique=False,
        schema=SCHEMA_NAME,
    )

    op.create_table(
        "pair_metrics",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("dataset_name", sa.Text(), nullable=False),
        sa.Column("run_name", sa.Text(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("quantization", sa.Text(), nullable=False),
        sa.Column("query_id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("retrieval_rank", sa.Integer(), nullable=True),
        sa.Column("retrieval_score", sa.Float(), nullable=True),
        sa.Column("rerank_rank", sa.Integer(), nullable=True),
        sa.Column("rerank_score", sa.Float(), nullable=True),
        sa.Column("relevance_grade", sa.Integer(), nullable=True),
        sa.Column("retrieval_latency_ms", sa.Float(), nullable=True),
        sa.Column("rerank_latency_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            [f"{SCHEMA_NAME}.documents.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["query_id"],
            [f"{SCHEMA_NAME}.queries.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_name",
            "model_name",
            "quantization",
            "query_id",
            "document_id",
            name="uq_pair_metrics",
        ),
        schema=SCHEMA_NAME,
    )
    op.create_index(
        op.f("ix_pair_metrics_dataset_name"),
        "pair_metrics",
        ["dataset_name"],
        unique=False,
        schema=SCHEMA_NAME,
    )
    op.create_index(
        op.f("ix_pair_metrics_document_id"),
        "pair_metrics",
        ["document_id"],
        unique=False,
        schema=SCHEMA_NAME,
    )
    op.create_index(
        op.f("ix_pair_metrics_model_name"),
        "pair_metrics",
        ["model_name"],
        unique=False,
        schema=SCHEMA_NAME,
    )
    op.create_index(
        op.f("ix_pair_metrics_query_id"),
        "pair_metrics",
        ["query_id"],
        unique=False,
        schema=SCHEMA_NAME,
    )
    op.create_index(
        op.f("ix_pair_metrics_quantization"),
        "pair_metrics",
        ["quantization"],
        unique=False,
        schema=SCHEMA_NAME,
    )
    op.create_index(
        op.f("ix_pair_metrics_run_name"),
        "pair_metrics",
        ["run_name"],
        unique=False,
        schema=SCHEMA_NAME,
    )

    _create_embedding_tables("document")
    _create_embedding_tables("query")


def downgrade() -> None:
    _drop_embedding_tables("query")
    _drop_embedding_tables("document")

    op.drop_index(
        op.f("ix_pair_metrics_run_name"),
        table_name="pair_metrics",
        schema=SCHEMA_NAME,
    )
    op.drop_index(
        op.f("ix_pair_metrics_quantization"),
        table_name="pair_metrics",
        schema=SCHEMA_NAME,
    )
    op.drop_index(
        op.f("ix_pair_metrics_query_id"),
        table_name="pair_metrics",
        schema=SCHEMA_NAME,
    )
    op.drop_index(
        op.f("ix_pair_metrics_model_name"),
        table_name="pair_metrics",
        schema=SCHEMA_NAME,
    )
    op.drop_index(
        op.f("ix_pair_metrics_document_id"),
        table_name="pair_metrics",
        schema=SCHEMA_NAME,
    )
    op.drop_index(
        op.f("ix_pair_metrics_dataset_name"),
        table_name="pair_metrics",
        schema=SCHEMA_NAME,
    )
    op.drop_table("pair_metrics", schema=SCHEMA_NAME)

    op.drop_index(
        op.f("ix_queries_dataset_name"),
        table_name="queries",
        schema=SCHEMA_NAME,
    )
    op.drop_table("queries", schema=SCHEMA_NAME)

    op.drop_index(
        op.f("ix_documents_source_id"),
        table_name="documents",
        schema=SCHEMA_NAME,
    )
    op.drop_index(
        op.f("ix_documents_dataset_name"),
        table_name="documents",
        schema=SCHEMA_NAME,
    )
    op.drop_table("documents", schema=SCHEMA_NAME)

    op.execute(sa.text(f"DROP SCHEMA IF EXISTS {SCHEMA_NAME}"))
