"""add_processed_halfvec_hnsw_index

Revision ID: ed586ee27f65
Revises: 2d99dff44136
Create Date: 2026-02-26 15:40:43.264502

"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "ed586ee27f65"
down_revision = "2d99dff44136"
branch_labels = None
depends_on = None

PROCESSED_SCHEMA = "processed"
CHUNKS_TABLE_NAME = "document_chunks"
IX_EMBEDDING_HNSW = "ix_processed_document_chunks_embedding_hnsw"
MAINTENANCE_MEMORY_TIERS = (
    "10GB",
    "6GB",
    "4GB",
    "2GB",
    "1GB",
    "512MB",
    "256MB",
    "128MB",
    "64MB",
)
CREATE_HNSW_INDEX_SQL = (
    f"CREATE INDEX IF NOT EXISTS {IX_EMBEDDING_HNSW} "
    f"ON {PROCESSED_SCHEMA}.{CHUNKS_TABLE_NAME} "
    "USING hnsw (embedding halfvec_cosine_ops) "
    "WITH (m = 16, ef_construction = 128)"
)
_MAINTENANCE_MEMORY_TIERS_SQL = ", ".join(f"'{tier}'" for tier in MAINTENANCE_MEMORY_TIERS)
_MAINTENANCE_MEMORY_TIERS_LABEL = ", ".join(MAINTENANCE_MEMORY_TIERS)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        f"""
        DO $$
        DECLARE
            mem text;
        BEGIN
            FOREACH mem IN ARRAY ARRAY[{_MAINTENANCE_MEMORY_TIERS_SQL}] LOOP
                BEGIN
                    EXECUTE format('SET LOCAL maintenance_work_mem = %L', mem);
                    EXECUTE {CREATE_HNSW_INDEX_SQL!r};
                    RETURN;
                EXCEPTION
                    WHEN insufficient_resources OR disk_full THEN
                        CONTINUE;
                END;
            END LOOP;

            RAISE EXCEPTION
                'Unable to create {IX_EMBEDDING_HNSW} with maintenance_work_mem tiers: {_MAINTENANCE_MEMORY_TIERS_LABEL}';
        END
        $$;
        """
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{IX_EMBEDDING_HNSW}")
