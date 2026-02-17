"""add native text search and rag indexes

Revision ID: 3c8a1e9b4d1f
Revises: 6b2f3b7e9c1a
Create Date: 2026-02-09 12:00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "3c8a1e9b4d1f"
down_revision = "6b2f3b7e9c1a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 128)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_documents_embedding_hnsw "
        "ON documents USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 128)"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_text_tsv_gin "
        "ON document_chunks USING gin (to_tsvector('simple', text))"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_documents_text_tsv_gin "
        "ON documents USING gin (to_tsvector('simple', text))"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_node_metadata_gin "
        "ON document_chunks USING gin ((node_metadata::jsonb))"
    )

    op.create_index(
        "ix_document_chunks_document_id",
        "document_chunks",
        ["document_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")

    op.execute("DROP INDEX IF EXISTS ix_document_chunks_node_metadata_gin")
    op.execute("DROP INDEX IF EXISTS ix_documents_text_tsv_gin")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_text_tsv_gin")
    op.execute("DROP INDEX IF EXISTS ix_documents_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
