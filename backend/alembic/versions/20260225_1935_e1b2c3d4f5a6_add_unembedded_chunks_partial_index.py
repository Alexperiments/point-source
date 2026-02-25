"""add partial index for unembedded processed chunks

Revision ID: e1b2c3d4f5a6
Revises: d3f4a6b8c1e2
Create Date: 2026-02-25 19:35:00.000000

"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "e1b2c3d4f5a6"
down_revision = "d3f4a6b8c1e2"
branch_labels = None
depends_on = None

PROCESSED_SCHEMA = "processed"
CHUNKS_TABLE = "document_chunks"
IX_UNEMBEDDED_ID = "ix_processed_document_chunks_unembedded_id"


def upgrade() -> None:
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {IX_UNEMBEDDED_ID} "
        f"ON {PROCESSED_SCHEMA}.{CHUNKS_TABLE} (id) "
        "WHERE embedding IS NULL AND text <> ''"
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {PROCESSED_SCHEMA}.{IX_UNEMBEDDED_ID}")
