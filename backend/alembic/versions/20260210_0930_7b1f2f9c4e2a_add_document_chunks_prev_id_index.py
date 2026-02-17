"""add document_chunks prev_id index

Revision ID: 7b1f2f9c4e2a
Revises: 3c8a1e9b4d1f
Create Date: 2026-02-10 09:30:00

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "7b1f2f9c4e2a"
down_revision = "3c8a1e9b4d1f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_document_chunks_prev_id",
        "document_chunks",
        ["prev_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_prev_id", table_name="document_chunks")
