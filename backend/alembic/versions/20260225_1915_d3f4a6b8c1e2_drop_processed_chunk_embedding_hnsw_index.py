"""drop processed.document_chunks embedding hnsw index

Revision ID: d3f4a6b8c1e2
Revises: c8a1d4f6e2b9
Create Date: 2026-02-25 19:15:00.000000

"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "d3f4a6b8c1e2"
down_revision = "c8a1d4f6e2b9"
branch_labels = None
depends_on = None

PROCESSED_SCHEMA = "processed"
CHUNKS_TABLE_NAME = "document_chunks"
IX_EMBEDDING_HNSW = "ix_processed_document_chunks_embedding_hnsw"


def upgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{IX_EMBEDDING_HNSW}")


def downgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {IX_EMBEDDING_HNSW} "
        f"ON {PROCESSED_SCHEMA}.{CHUNKS_TABLE_NAME} "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 128)"
    )
