"""create processed document_chunks table

Revision ID: af5d3c1e9b72
Revises: 8e1b9c74a6d2
Create Date: 2026-02-20 00:11:00.000000

"""
import pgvector
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "af5d3c1e9b72"
down_revision = "8e1b9c74a6d2"
branch_labels = None
depends_on = None

PROCESSED_SCHEMA = "processed"
TABLE_NAME = "document_chunks"
DOCUMENTS_TABLE = "documents"


def upgrade() -> None:
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {PROCESSED_SCHEMA}"))

    op.create_table(
        TABLE_NAME,
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=True),
        sa.Column("parent_id", sa.UUID(), nullable=True),
        sa.Column("prev_id", sa.UUID(), nullable=True),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=1024),
            nullable=True,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("node_metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(
            ["document_id"],
            [f"{PROCESSED_SCHEMA}.{DOCUMENTS_TABLE}.id"],
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            [f"{PROCESSED_SCHEMA}.{TABLE_NAME}.id"],
        ),
        sa.ForeignKeyConstraint(
            ["prev_id"],
            [f"{PROCESSED_SCHEMA}.{TABLE_NAME}.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=PROCESSED_SCHEMA,
    )


def downgrade() -> None:
    op.drop_table(TABLE_NAME, schema=PROCESSED_SCHEMA)
