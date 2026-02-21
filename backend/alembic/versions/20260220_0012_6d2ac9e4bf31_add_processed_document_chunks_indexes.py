"""add retrieval indexes for processed.document_chunks

Revision ID: 6d2ac9e4bf31
Revises: af5d3c1e9b72
Create Date: 2026-02-20 00:12:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "6d2ac9e4bf31"
down_revision = "af5d3c1e9b72"
branch_labels = None
depends_on = None

PROCESSED_SCHEMA = "processed"
CHUNKS_TABLE_NAME = "document_chunks"

IX_EMBEDDING_HNSW = "ix_processed_document_chunks_embedding_hnsw"
IX_TEXT_TSV_GIN = "ix_processed_document_chunks_text_tsv_gin"
IX_METADATA_GIN = "ix_processed_document_chunks_node_metadata_gin"
IX_DOCUMENT_ID = "ix_processed_document_chunks_document_id"
IX_PREV_ID = "ix_processed_document_chunks_prev_id"
IX_PARENT_ID = "ix_processed_document_chunks_parent_id"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        f"CREATE INDEX IF NOT EXISTS {IX_EMBEDDING_HNSW} "
        f"ON {PROCESSED_SCHEMA}.{CHUNKS_TABLE_NAME} "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 128)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {IX_TEXT_TSV_GIN} "
        f"ON {PROCESSED_SCHEMA}.{CHUNKS_TABLE_NAME} "
        "USING gin (to_tsvector('simple', text))"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {IX_METADATA_GIN} "
        f"ON {PROCESSED_SCHEMA}.{CHUNKS_TABLE_NAME} "
        "USING gin ((node_metadata::jsonb))"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {IX_DOCUMENT_ID} "
        f"ON {PROCESSED_SCHEMA}.{CHUNKS_TABLE_NAME} (document_id)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {IX_PREV_ID} "
        f"ON {PROCESSED_SCHEMA}.{CHUNKS_TABLE_NAME} (prev_id)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {IX_PARENT_ID} "
        f"ON {PROCESSED_SCHEMA}.{CHUNKS_TABLE_NAME} (parent_id)"
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{IX_PARENT_ID}")
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{IX_PREV_ID}")
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{IX_DOCUMENT_ID}")
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{IX_METADATA_GIN}")
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{IX_TEXT_TSV_GIN}")
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{IX_EMBEDDING_HNSW}")
